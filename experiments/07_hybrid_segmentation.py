#!/usr/bin/env python3
"""Experiment 07: Hybrid segmentation — scout → region-aware extraction.

Improvements over experiment 06:
  1. Scout detects rotation per region (title blocks often 90°, some notes panels too)
  2. Crops are rotated to upright before extraction
  3. Adaptive extraction:
     - Small regions (< TILE_THRESHOLD px on longest side) → direct extraction
     - Large regions (DRAWING_AREA, oversized NOTES, large SCHEDULE) → tiled within bounds
  4. Per-region ground truth via PyMuPDF spatial text (page.get_text("dict"))
     so precision/recall is meaningful per region, not just % of whole page

Key questions being tested:
  - Does rotation correction improve title block and margin note extraction?
  - Does tiling within DRAWING_AREA bounds beat tiling the full page?
  - Is per-region recall higher than the 87% whole-page baseline?
"""

import base64
import io
import json
import re
import time
from pathlib import Path
from PIL import Image
from openai import OpenAI
import fitz  # PyMuPDF

LMSTUDIO_HOST = "http://localhost:1234"
SAMPLES_DIR   = Path(__file__).parent / "samples"
SEGMENTS_DIR  = Path(__file__).parent / "segments" / "exp07"
PDF_DIR       = Path(__file__).parent.parent / "Sample Documents"

SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "google/gemma-3-27b"

# Region crop is tiled if its longest side exceeds this threshold (px)
TILE_THRESHOLD_PX = 2000

# Tile config for large regions: (rows, cols, overlap_pct)
# Adaptive based on region aspect ratio — see tile_config_for_region()
DEFAULT_TILE_OVERLAP = 0.15

TEST_CASES = [
    {
        "pdf":      "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf",
        "page_idx": 47,
        "image":    "PN92107_JSOC_MCC_V3__p048_200dpi.png",
        "type":     "floor_plan",
        "baseline_recall": 0.87,
    },
    {
        "pdf":      "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf",
        "page_idx": 94,
        "image":    "PN92107_JSOC_MCC_V3__p095_200dpi.png",
        "type":     "schedule",
        "baseline_recall": None,
    },
]

# ── Scout prompt ─────────────────────────────────────────────────────────────
SCOUT_PROMPT = """You are analyzing a D-size (34"×22") construction drawing.

Identify every distinct semantic region on this page. Construction drawings frequently have:
- Text panels or title blocks printed SIDEWAYS (rotated 90° CW or CCW) along margins
- Notes sections that run vertically along a side
- A title block in the lower-right corner (sometimes rotated)
- Overlapping smaller regions (key plans, area tables) at the bottom

For each region return a JSON object with:
  "type":     TITLE_BLOCK | NOTES | DRAWING_AREA | LEGEND | SCHEDULE | OTHER
  "label":    short description (e.g. "General Notes", "2nd Floor Fire Suppression Plan")
  "rotation": degrees to rotate the crop CLOCKWISE to make text upright (0, 90, 180, 270)
              — use 0 if text is already upright, 90 if text reads bottom-to-top,
                270 if text reads top-to-bottom
  "x1","y1","x2","y2": region bounds as fractions of image width/height (0.0 to 1.0)

Be precise — do not let regions overlap unless they genuinely overlap in the source.

Return ONLY a JSON array, no other text."""

# ── Per-region extraction prompts ────────────────────────────────────────────
EXTRACT_PROMPTS = {
    "TITLE_BLOCK": (
        "This is the title block from a construction drawing. "
        "Extract EVERY piece of text exactly as written: project name, building name, "
        "address, sheet number, sheet title, revision letters and dates, drawing date, "
        "scale, engineer/architect/firm names, seal text, contract number, and all other fields. "
        "Do NOT paraphrase. Output raw text, one field per line."
    ),
    "NOTES": (
        "This is the General Notes or code notes section of a construction drawing. "
        "Extract ALL text exactly as written. Preserve note numbers and sub-items. "
        "Include every specification reference code (e.g. NFPA 72, IBC, ASTM), "
        "abbreviation definitions, and instruction. "
        "Output raw text, preserving note numbers."
    ),
    "DRAWING_AREA": (
        "This is a section of the main drawing area of a construction drawing. "
        "Extract ALL visible text: room names, room numbers, space labels, "
        "door/window tags, equipment tags and labels, dimensions (all numeric values), "
        "callout text, revision clouds text, north arrow label, scale bar text, "
        "and any annotation text. "
        "Output raw text, one item per line. Do NOT describe what you see — only output the text."
    ),
    "LEGEND": (
        "This is a legend, symbol key, or abbreviation table from a construction drawing. "
        "Extract ALL text: symbol names, abbreviation codes, and their full descriptions. "
        "Preserve pairing (abbreviation = definition) where visible. "
        "Output raw text."
    ),
    "SCHEDULE": (
        "This is a schedule or data table from a construction drawing. "
        "Extract ALL text from every cell: column headers, row labels, and all cell values. "
        "Preserve row structure using | separators if helpful. "
        "Output raw text, do NOT invent or guess any values."
    ),
    "OTHER": (
        "Extract ALL text visible in this section of a construction drawing. "
        "Include every piece of text you can read, exactly as written. "
        "Output raw text only, one item per line."
    ),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def normalize(text: str) -> str:
    text = re.sub(r'[*#_`\-\|>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def word_overlap(gt: set[str], extracted: str) -> dict:
    ex_words = set(normalize(extracted).split())
    if not gt:
        return {"precision": 0, "recall": 0, "f1": 0, "gt_n": 0, "ex_n": 0, "matched": 0, "missed": [], "extra": []}
    matched = gt & ex_words
    p = len(matched) / len(ex_words) if ex_words else 0
    r = len(matched) / len(gt)
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {
        "precision": p, "recall": r, "f1": f1,
        "gt_n": len(gt), "ex_n": len(ex_words), "matched": len(matched),
        "missed": sorted(gt - ex_words)[:15],
        "extra":  sorted(ex_words - gt)[:15],
    }

def spatial_gt_for_bbox(page: fitz.Page, x1n: float, y1n: float, x2n: float, y2n: float,
                         pw: int, ph: int) -> set[str]:
    """Extract PDF embedded text words that fall within a normalized bbox region."""
    clip = fitz.Rect(x1n * pw, y1n * ph, x2n * pw, y2n * ph)
    words = page.get_text("words", clip=clip)  # returns (x0,y0,x1,y1,word,block,line,word_idx)
    result = set()
    for w in words:
        word = normalize(w[4])
        if word:
            result.add(word)
    return result

def tile_config_for_region(w: int, h: int) -> tuple[int, int]:
    """Pick (rows, cols) to keep each tile's longest side under TILE_THRESHOLD_PX."""
    cols = max(1, -(-w // TILE_THRESHOLD_PX))  # ceiling division
    rows = max(1, -(-h // TILE_THRESHOLD_PX))
    return rows, cols

def create_region_tiles(img: Image.Image, rows: int, cols: int,
                        overlap: float, stem: str, rtype: str) -> list[dict]:
    """Tile a region image into overlapping sub-crops. Returns list of tile dicts."""
    w, h = img.size
    tw = w / cols
    th = h / rows
    ow = int(tw * overlap)
    oh = int(th * overlap)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tw) - ow)
            y1 = max(0, int(r * th) - oh)
            x2 = min(w, int((c+1)*tw) + ow)
            y2 = min(h, int((r+1)*th) + oh)
            crop = img.crop((x1, y1, x2, y2))
            path = SEGMENTS_DIR / f"{stem}_{rtype}_t{r}x{c}.png"
            crop.save(str(path))
            tiles.append({"path": path, "row": r, "col": c, "size": f"{x2-x1}x{y2-y1}"})
    return tiles

# ── LMStudio calls ────────────────────────────────────────────────────────────

def scout_regions(client: OpenAI, image_path: Path) -> list[dict]:
    """Downsampled scout pass — returns list of validated region dicts."""
    img = Image.open(image_path)
    w, h = img.size
    # Use 2000px wide scout image — large enough to read all layout details
    scale = min(1.0, 2000 / w)
    thumb = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS) if scale < 1 else img.copy()
    img.close()
    tw, th = thumb.size
    print(f"  Scout thumbnail: {tw}x{th}px")

    b64 = encode_pil(thumb)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": SCOUT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}],
            max_tokens=2048,
            temperature=0.05,
        )
        elapsed = time.time() - start
        raw = resp.choices[0].message.content
        print(f"  Scout response ({elapsed:.1f}s):\n{raw[:800]}\n")

        cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```","").strip()
        regions = json.loads(cleaned)

        valid = []
        for r in regions:
            try:
                x1 = max(0.0, min(1.0, float(r["x1"])))
                y1 = max(0.0, min(1.0, float(r["y1"])))
                x2 = max(0.0, min(1.0, float(r["x2"])))
                y2 = max(0.0, min(1.0, float(r["y2"])))
                rot = int(r.get("rotation", 0))
                if rot not in (0, 90, 180, 270):
                    rot = 0
                if x2 > x1 + 0.01 and y2 > y1 + 0.01:
                    valid.append({
                        "type":     r.get("type","OTHER").upper(),
                        "label":    r.get("label",""),
                        "rotation": rot,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })
            except (KeyError, ValueError, TypeError):
                continue
        return valid
    except json.JSONDecodeError as e:
        print(f"  Scout JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"  Scout error: {e}")
        return []


def extract_text(client: OpenAI, image_path: Path, rtype: str,
                 tile_context: str = "") -> dict:
    """Send a region crop (or tile) to the vision model for text extraction."""
    prompt = EXTRACT_PROMPTS.get(rtype, EXTRACT_PROMPTS["OTHER"])
    if tile_context:
        prompt += f"\n\nNote: {tile_context}"
    b64 = encode_file(image_path)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}],
            max_tokens=4096,
            temperature=0.05,
        )
        elapsed = time.time() - start
        return {"text": resp.choices[0].message.content, "elapsed": elapsed, "error": None}
    except Exception as e:
        return {"text": None, "elapsed": time.time()-start, "error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=f"{LMSTUDIO_HOST}/v1", api_key="not-needed")

    print("=" * 80)
    print("EXPERIMENT 07: HYBRID SEGMENTATION (scout + rotation + region-aware tiling)")
    print(f"Model: {MODEL}  |  Host: {LMSTUDIO_HOST}")
    print(f"Tile threshold: {TILE_THRESHOLD_PX}px — larger regions get tiled within bounds")
    print("=" * 80)

    for case in TEST_CASES:
        image_path = SAMPLES_DIR / case["image"]
        if not image_path.exists():
            print(f"\nSKIP: {case['image']} not found"); continue

        print(f"\n{'='*80}")
        print(f"PAGE: {case['image']}  ({case['type']})")
        if case["baseline_recall"]:
            print(f"2×3 dumb-tile baseline recall: {case['baseline_recall']:.1%}")
        print(f"{'='*80}")

        # Load image dimensions
        img_full = Image.open(image_path)
        iw, ih = img_full.size
        img_full.close()
        print(f"Full image: {iw}×{ih}px")

        # Ground truth — whole page and spatially-indexed
        doc  = fitz.open(str(PDF_DIR / case["pdf"]))
        page = doc[case["page_idx"]]
        # PyMuPDF page size in points; we'll use it for spatial clipping
        prect = page.rect  # (x0,y0,x1,y1) in points
        pw_pt, ph_pt = prect.width, prect.height
        gt_all_text = page.get_text()
        gt_all_words = set(normalize(gt_all_text).split())
        print(f"Ground truth: {len(gt_all_text)} chars, {len(gt_all_words)} unique words")

        # ── Step 1: Scout ────────────────────────────────────────────────────
        print(f"\n── Step 1: Scout pass ──────────────────────────────────────")
        regions = scout_regions(client, image_path)

        if not regions:
            print("  Scout returned no regions — aborting this page"); doc.close(); continue

        print(f"  {len(regions)} region(s) detected:")
        for i, r in enumerate(regions):
            pw = int((r["x2"]-r["x1"]) * iw)
            ph = int((r["y2"]-r["y1"]) * ih)
            rot_note = f" [ROT {r['rotation']}°]" if r["rotation"] else ""
            print(f"    [{i}] {r['type']:12s} {r['rotation']:3d}°  "
                  f"bbox=({r['x1']:.2f},{r['y1']:.2f})-({r['x2']:.2f},{r['y2']:.2f})  "
                  f"{pw}×{ph}px  \"{r['label'][:45]}\"")

        # ── Step 2: Crop, rotate, tile or extract ────────────────────────────
        print(f"\n── Step 2: Crop → rotate → extract/tile ────────────────────")
        stem = image_path.stem
        page_results = []  # {region, texts, elapsed}

        for i, r in enumerate(regions):
            rtype = r["type"]
            rot   = r["rotation"]
            print(f"\n  ── Region [{i}] {rtype}: \"{r['label'][:50]}\"")

            # Spatial ground truth for this region (from PDF text positions)
            region_gt = spatial_gt_for_bbox(page, r["x1"], r["y1"], r["x2"], r["y2"],
                                             pw_pt, ph_pt)
            if region_gt:
                print(f"     Region GT (spatial): {len(region_gt)} unique words")
            else:
                print(f"     Region GT (spatial): none (no embedded text in this region)")

            # Crop from full-res image
            img_full = Image.open(image_path)
            x1p = int(r["x1"] * iw); y1p = int(r["y1"] * ih)
            x2p = int(r["x2"] * iw); y2p = int(r["y2"] * ih)
            crop = img_full.crop((x1p, y1p, x2p, y2p))
            img_full.close()
            cw, ch = crop.size

            # Rotation correction
            if rot:
                # PIL rotate is CCW; to correct a CW rotation we rotate CCW by same amount
                crop = crop.rotate(rot, expand=True)
                cw, ch = crop.size
                print(f"     Rotated {rot}° → now {cw}×{ch}px")

            # Save the (possibly rotated) region crop
            crop_path = SEGMENTS_DIR / f"{stem}_r{i:02d}_{rtype}.png"
            crop.save(str(crop_path))

            longest = max(cw, ch)
            all_texts = []
            total_time = 0.0

            if longest <= TILE_THRESHOLD_PX:
                # ── Small region: direct extraction
                print(f"     {cw}×{ch}px — direct extraction")
                res = extract_text(client, crop_path, rtype)
                if res["error"]:
                    print(f"     ERROR: {res['error']}")
                else:
                    total_time += res["elapsed"]
                    text = res["text"] or ""
                    all_texts.append(text)
                    m = word_overlap(region_gt, text) if region_gt else word_overlap(gt_all_words, text)
                    gt_label = "region GT" if region_gt else "page GT"
                    print(f"     {res['elapsed']:.1f}s | {len(text)} chars | "
                          f"recall={m['recall']:.1%} precision={m['precision']:.1%} [{gt_label}]")
                    preview = text[:180].replace('\n',' ')
                    print(f"     Preview: {preview}")
            else:
                # ── Large region: tile within bounds
                rows, cols = tile_config_for_region(cw, ch)
                n_tiles = rows * cols
                print(f"     {cw}×{ch}px — tiling {rows}×{cols} ({n_tiles} tiles, "
                      f"threshold {TILE_THRESHOLD_PX}px)")
                tiles = create_region_tiles(crop, rows, cols, DEFAULT_TILE_OVERLAP,
                                            f"{stem}_r{i:02d}", rtype)
                for t in tiles:
                    ctx = (f"This is tile row {t['row']}, col {t['col']} of a "
                           f"{rows}×{cols} grid within the {rtype.lower().replace('_',' ')} region. "
                           f"Tile size: {t['size']}px.")
                    res = extract_text(client, t["path"], rtype, tile_context=ctx)
                    if res["error"]:
                        print(f"     Tile [{t['row']},{t['col']}] ERROR: {res['error']}")
                    else:
                        total_time += res["elapsed"]
                        text = res["text"] or ""
                        all_texts.append(text)
                        print(f"     Tile [{t['row']},{t['col']}] {res['elapsed']:.1f}s "
                              f"| {len(text)} chars")

                combined_tile_text = "\n".join(all_texts)
                m = word_overlap(region_gt, combined_tile_text) if region_gt \
                    else word_overlap(gt_all_words, combined_tile_text)
                gt_label = "region GT" if region_gt else "page GT"
                print(f"     Region combined: recall={m['recall']:.1%} "
                      f"precision={m['precision']:.1%} [{gt_label}]  "
                      f"({m['matched']}/{m['gt_n']} words)")
                if m["missed"]:
                    print(f"     Missed sample: {', '.join(m['missed'][:8])}")

            page_results.append({
                "region": r,
                "region_gt": region_gt,
                "texts": all_texts,
                "elapsed": total_time,
            })

        doc.close()

        # ── Step 3: Whole-page combined metrics ──────────────────────────────
        print(f"\n── Step 3: Whole-page combined results ─────────────────────")
        all_combined = "\n".join(
            t for pr in page_results for t in pr["texts"]
        )
        metrics = word_overlap(gt_all_words, all_combined)

        total_elapsed = sum(pr["elapsed"] for pr in page_results)
        print(f"  Total time:   {total_elapsed:.1f}s")
        print(f"  Total chars:  {len(all_combined)}")
        print(f"\n  RECALL:    {metrics['recall']:.1%}")
        print(f"  PRECISION: {metrics['precision']:.1%}")
        print(f"  F1:        {metrics['f1']:.1%}")
        print(f"  Matched:   {metrics['matched']}/{metrics['gt_n']} unique words")

        if case["baseline_recall"]:
            delta = metrics["recall"] - case["baseline_recall"]
            sign = "+" if delta >= 0 else ""
            verdict = "BETTER ✓" if delta > 0.02 else ("WORSE ✗" if delta < -0.02 else "≈ SAME")
            print(f"\n  vs 2×3 dumb-tile baseline: {sign}{delta:.1%}  [{verdict}]")

        if metrics["missed"]:
            print(f"\n  MISSED sample:        {', '.join(metrics['missed'][:12])}")
        if metrics["extra"]:
            print(f"  HALLUCINATED sample:  {', '.join(metrics['extra'][:12])}")

        # Region summary table
        print(f"\n  Region summary:")
        print(f"  {'Idx':>3}  {'Type':<12} {'Rot':>4}  {'GT words':>8}  {'Recall':>7}  {'Strategy'}")
        print(f"  {'-'*3}  {'-'*12} {'-'*4}  {'-'*8}  {'-'*7}  {'-'*20}")
        for i, pr in enumerate(page_results):
            r = pr["region"]
            rgt = pr["region_gt"]
            combined = "\n".join(pr["texts"])
            m = word_overlap(rgt, combined) if rgt else {"recall": 0, "gt_n": 0}
            cw = int((r["x2"]-r["x1"])*iw)
            ch = int((r["y2"]-r["y1"])*ih)
            strategy = "direct" if max(cw,ch) <= TILE_THRESHOLD_PX else \
                       f"tiled {tile_config_for_region(cw,ch)[0]}×{tile_config_for_region(cw,ch)[1]}"
            if r["rotation"]:
                strategy += f" (rot{r['rotation']}°)"
            gt_disp = str(len(rgt)) if rgt else "n/a"
            recall_disp = f"{m['recall']:.1%}" if rgt else " n/a"
            print(f"  {i:>3}  {r['type']:<12} {r['rotation']:>4}°  {gt_disp:>8}  "
                  f"{recall_disp:>7}  {strategy}")

    print(f"\n{'='*80}")
    print("DONE  —  segment tiles saved to experiments/segments/exp07/")
    print("="*80)


if __name__ == "__main__":
    main()
