#!/usr/bin/env python3
"""Experiment 09: PDF-guided region detection + debug visualization.

Problem with exp06-08: vision scout hallucinates layout positions.
Fix: use PyMuPDF text block positions to define region boundaries precisely.

Algorithm:
  1. page.get_text("blocks") → all text block bounding boxes from PDF
  2. Classify blocks: "dense" (multi-line text panels = notes/title block)
     vs "sparse" (short labels scattered inside the drawing area)
  3. Cluster dense blocks to find RIGHT PANEL (notes + title block column)
  4. DRAWING_AREA = everything left of the right panel's left edge
  5. Within right panel: split by horizontal gaps into NOTES vs TITLE_BLOCK
  6. Vision model used only for rotation detection on small border strips

Best extraction config from exp08: 300 DPI, 1500px tile threshold, 25% overlap.

Also outputs: debug PNG with colored bounding boxes for visual verification.
"""

import base64
import io
import json
import re
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import fitz  # PyMuPDF

LMSTUDIO_HOST = "http://localhost:1234"
SAMPLES_DIR   = Path(__file__).parent / "samples"
OUT_DIR       = Path(__file__).parent / "segments" / "exp09"
PDF_DIR       = Path(__file__).parent.parent / "Sample Documents"

OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "google/gemma-3-27b"

# Best config from exp08
RENDER_DPI      = 300
TILE_THRESHOLD  = 1500   # px — longest tile side
TILE_OVERLAP    = 0.25

# Text block classification thresholds
# A block is "dense" (text panel) if it has >= this many newlines or chars per height
DENSE_MIN_LINES  = 4     # blocks with >= 4 lines are "text panel" candidates
DENSE_MIN_CHARS  = 80    # or >= 80 chars
# A block is in the "right panel" if its left edge is past this fraction of page width
# (heuristic — will be overridden by actual clustering)
RIGHT_PANEL_MIN_X = 0.55  # only consider blocks in the right ~45% as right-panel candidates

# Colors for debug visualization (RGB)
COLORS = {
    "DRAWING_AREA": (100, 180, 255),   # blue
    "NOTES":        (100, 220, 100),   # green
    "TITLE_BLOCK":  (255, 180, 60),    # orange
    "SCHEDULE":     (220, 100, 220),   # purple
    "LEGEND":       (255, 100, 100),   # red
    "OTHER":        (180, 180, 180),   # grey
}

TEST_CASES = [
    {
        "pdf":      "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf",
        "page_idx": 47,
        "label":    "p048_floor_plan",
        "baseline_recall": 0.87,   # dumb 2×3 tiling baseline
    },
    {
        "pdf":      "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf",
        "page_idx": 94,
        "label":    "p095_schedule",
        "baseline_recall": None,
    },
]

# Extraction prompts (same as exp07/08)
EXTRACT_PROMPTS = {
    "DRAWING_AREA": (
        "This is a section of the main drawing area of a construction drawing. "
        "Extract ALL visible text exactly as written — every character matters: "
        "room names, room numbers, space labels, door/window tags, equipment tags, "
        "dimensions (ALL numeric values including fractions), callout text, "
        "north arrow label, scale bar text, revision cloud text, "
        "sheet cross-references (e.g. 'SEE SHEET E-101'), grid line labels, "
        "level/floor labels, any stamp or seal text visible. "
        "Do NOT describe what you see. Output raw text only, one item per line. "
        "If text is small or partially obscured, make your best reading and include it."
    ),
    "NOTES": (
        "This is the General Notes or code notes section of a construction drawing. "
        "Extract ALL text exactly as written. Preserve note numbers and sub-items. "
        "Include every specification reference code (e.g. NFPA 72, IBC, ASTM), "
        "abbreviation definitions, and instruction. "
        "Output raw text, preserving note numbers."
    ),
    "TITLE_BLOCK": (
        "This is the title block from a construction drawing. "
        "Extract EVERY piece of text exactly as written: project name, building name, "
        "address, sheet number, sheet title, revision letters and dates, drawing date, "
        "scale, engineer/architect/firm names, seal text, contract number, and all other fields. "
        "Do NOT paraphrase. Output raw text, one field per line."
    ),
    "SCHEDULE": (
        "This is a schedule or data table from a construction drawing. "
        "Extract ALL text from every cell: column headers, row labels, and all cell values. "
        "Preserve row structure using | separators if helpful. "
        "Output raw text, do NOT invent or guess any values."
    ),
    "LEGEND": (
        "This is a legend, symbol key, or abbreviation table from a construction drawing. "
        "Extract ALL text: symbol names, abbreviation codes, and their full descriptions. "
        "Preserve pairing (abbreviation = definition) where visible. "
        "Output raw text."
    ),
    "OTHER": (
        "Extract ALL text visible in this section of a construction drawing. "
        "Include every piece of text you can read, exactly as written. "
        "Output raw text only, one item per line."
    ),
}

# Rotation-detection prompt (sent to vision model on narrow strips only)
ROTATION_PROMPT = """Look at this image strip from a construction drawing.
Is the text in this strip rotated (i.e., you'd need to tilt your head to read it)?
Reply with ONLY a JSON object: {"rotation": N} where N is the degrees to rotate
the image CLOCKWISE to make the text upright (0, 90, 180, or 270).
If already upright, return {"rotation": 0}."""


# ── PyMuPDF region detection ────────────────────────────────────────────────

def classify_text_blocks(page: fitz.Page) -> dict:
    """
    Use PyMuPDF text block positions to identify semantic regions.
    Returns a dict with normalized bbox for each region found.
    """
    pw = page.rect.width
    ph = page.rect.height

    blocks = page.get_text("blocks")
    # blocks: (x0, y0, x1, y1, text, block_no, block_type)
    # block_type: 0=text, 1=image

    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    # Classify each block as dense (text panel) or sparse (drawing label)
    dense, sparse = [], []
    for b in text_blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        lines = text.strip().count('\n') + 1
        chars = len(text.strip())
        if lines >= DENSE_MIN_LINES or chars >= DENSE_MIN_CHARS:
            dense.append(b)
        else:
            sparse.append(b)

    print(f"  Text blocks: {len(text_blocks)} total, {len(dense)} dense, {len(sparse)} sparse")

    # Find right panel: dense blocks in the right portion of the page
    right_panel_blocks = [b for b in dense if b[0] / pw >= RIGHT_PANEL_MIN_X]
    print(f"  Right-panel dense blocks: {len(right_panel_blocks)}")

    if not right_panel_blocks:
        print("  WARNING: no dense right-panel blocks found — falling back to full-page tiling")
        return {
            "DRAWING_AREA": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "rotation": 0}
        }

    # Right panel bounding box
    rp_x0 = min(b[0] for b in right_panel_blocks) / pw   # leftmost edge
    rp_x1 = max(b[2] for b in right_panel_blocks) / pw   # rightmost edge
    rp_y0 = min(b[1] for b in right_panel_blocks) / ph
    rp_y1 = max(b[3] for b in right_panel_blocks) / ph

    print(f"  Right panel bbox (normalized): x={rp_x0:.3f}–{rp_x1:.3f}, y={rp_y0:.3f}–{rp_y1:.3f}")

    # DRAWING_AREA = everything left of the right panel, full height
    # Add a small margin to avoid clipping drawing content at the boundary
    drawing_x2 = max(0.01, rp_x0 - 0.005)

    # Split right panel into NOTES (upper) and TITLE_BLOCK (lower)
    # Find the largest vertical gap between right-panel blocks
    rp_sorted = sorted(right_panel_blocks, key=lambda b: b[1])  # sort by y0
    best_gap_y  = None
    best_gap_sz = 0
    for i in range(len(rp_sorted) - 1):
        gap_top = rp_sorted[i][3]      # bottom of block i
        gap_bot = rp_sorted[i+1][1]    # top of block i+1
        gap_sz  = gap_bot - gap_top
        if gap_sz > best_gap_sz:
            best_gap_sz = gap_sz
            best_gap_y  = (gap_top + gap_bot) / 2 / ph  # normalized midpoint

    print(f"  Largest right-panel gap: {best_gap_sz:.1f}pt at y≈{best_gap_y:.3f}" if best_gap_y else
          "  No gap found in right panel — treating as single NOTES region")

    regions = {}

    # Drawing area
    regions["DRAWING_AREA"] = {
        "x1": 0.0, "y1": 0.0, "x2": drawing_x2, "y2": 1.0, "rotation": 0
    }

    if best_gap_y and best_gap_sz > 5:  # at least 5pt gap to split
        regions["NOTES"] = {
            "x1": rp_x0, "y1": 0.0, "x2": 1.0, "y2": best_gap_y, "rotation": 0
        }
        regions["TITLE_BLOCK"] = {
            "x1": rp_x0, "y1": best_gap_y, "x2": 1.0, "y2": 1.0, "rotation": 0
        }
    else:
        regions["NOTES"] = {
            "x1": rp_x0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "rotation": 0
        }

    # Also find any schedule blocks that span wide portions of the page
    # (e.g. a horizontal schedule band)
    schedule_blocks = [b for b in dense if
                       (b[2] - b[0]) / pw > 0.5 and   # wide block
                       b[0] / pw < 0.5]               # starts in left half
    if schedule_blocks:
        sb = schedule_blocks[0]
        print(f"  Wide schedule-like block detected: x={sb[0]/pw:.3f}–{sb[2]/pw:.3f}")

    return regions


def detect_rotation(client: OpenAI, image_path: Path) -> int:
    """Ask vision model for rotation of a small strip image. Returns 0/90/180/270."""
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":[
                {"type":"text","text":ROTATION_PROMPT},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}
            ]}],
            max_tokens=64, temperature=0.0,
        )
        raw = resp.choices[0].message.content
        cleaned = re.sub(r"```(?:json)?\s*","",raw).replace("```","").strip()
        parsed = json.loads(cleaned)
        rot = int(parsed.get("rotation", 0))
        return rot if rot in (0,90,180,270) else 0
    except Exception:
        return 0


def render_region(pdf_path: Path, page_idx: int, region: dict, dpi: int) -> Image.Image:
    """Re-render a normalized-bbox region from PDF vector data at given DPI."""
    doc  = fitz.open(str(pdf_path))
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    clip = fitz.Rect(
        region["x1"]*pw, region["y1"]*ph,
        region["x2"]*pw, region["y2"]*ph,
    )
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def tile_image(img: Image.Image, threshold: int, overlap: float,
               stem: str) -> tuple[list[dict], int, int]:
    w, h = img.size
    cols = max(1, -(-w // threshold))
    rows = max(1, -(-h // threshold))
    tw, th = w/cols, h/rows
    ow, oh = int(tw*overlap), int(th*overlap)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c*tw)-ow);  y1 = max(0, int(r*th)-oh)
            x2 = min(w, int((c+1)*tw)+ow); y2 = min(h, int((r+1)*th)+oh)
            crop = img.crop((x1,y1,x2,y2))
            path = OUT_DIR / f"{stem}_t{r}x{c}.png"
            crop.save(str(path))
            tiles.append({"path":path,"row":r,"col":c,"size":f"{x2-x1}×{y2-y1}"})
    return tiles, rows, cols


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def extract_text(client: OpenAI, image_path: Path, rtype: str,
                 tile_ctx: str = "") -> dict:
    prompt = EXTRACT_PROMPTS.get(rtype, EXTRACT_PROMPTS["OTHER"])
    if tile_ctx:
        prompt += f"\n\nNote: {tile_ctx}"
    b64 = encode_file(image_path)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}
            ]}],
            max_tokens=4096, temperature=0.05,
        )
        return {"text":resp.choices[0].message.content,
                "elapsed":time.time()-start,"error":None}
    except Exception as e:
        return {"text":None,"elapsed":time.time()-start,"error":str(e)}


def normalize(text: str) -> str:
    text = re.sub(r'[*#_`\-\|>]','',text)
    text = re.sub(r'\s+',' ',text).strip().lower()
    return text


def word_overlap(gt: set, extracted: str) -> dict:
    ex = set(normalize(extracted).split())
    if not gt:
        return {"precision":0,"recall":0,"f1":0,"gt_n":0,"ex_n":0,"matched":0,"missed":[],"extra":[]}
    matched = gt & ex
    p = len(matched)/len(ex) if ex else 0
    r = len(matched)/len(gt)
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {"precision":p,"recall":r,"f1":f1,
            "gt_n":len(gt),"ex_n":len(ex),"matched":len(matched),
            "missed":sorted(gt-ex)[:20],"extra":sorted(ex-gt)[:15]}


def spatial_gt(page: fitz.Page, region: dict) -> set:
    pw, ph = page.rect.width, page.rect.height
    clip = fitz.Rect(region["x1"]*pw, region["y1"]*ph,
                     region["x2"]*pw, region["y2"]*ph)
    words = page.get_text("words", clip=clip)
    return {normalize(w[4]) for w in words if normalize(w[4])}


def save_debug_image(pdf_path: Path, page_idx: int, regions: dict,
                     out_path: Path, dpi: int = 72):
    """Render page at low DPI and overlay colored region bounding boxes."""
    doc  = fitz.open(str(pdf_path))
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    mat  = fitz.Matrix(dpi/72, dpi/72)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    iw, ih = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    for rtype, r in regions.items():
        color = COLORS.get(rtype, COLORS["OTHER"])
        fill  = color + (40,)    # semi-transparent fill
        outline = color + (220,) # solid-ish outline
        x1p = int(r["x1"]*iw); y1p = int(r["y1"]*ih)
        x2p = int(r["x2"]*iw); y2p = int(r["y2"]*ih)
        draw.rectangle([x1p,y1p,x2p,y2p], fill=fill, outline=outline[:3], width=3)

        # Label
        label = f"{rtype}"
        if r.get("rotation"):
            label += f" [{r['rotation']}°]"
        draw.text((x1p+6, y1p+4), label, fill=outline[:3])

    img.save(str(out_path))
    print(f"  Debug image saved: {out_path.name} ({iw}×{ih}px)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=f"{LMSTUDIO_HOST}/v1", api_key="not-needed")

    print("=" * 80)
    print("EXPERIMENT 09: PDF-GUIDED REGION DETECTION + DEBUG VISUALIZATION")
    print(f"Model: {MODEL}  |  Host: {LMSTUDIO_HOST}")
    print(f"Config: {RENDER_DPI} DPI, {TILE_THRESHOLD}px tiles, {TILE_OVERLAP:.0%} overlap")
    print("=" * 80)

    for case in TEST_CASES:
        pdf_path = PDF_DIR / case["pdf"]
        label    = case["label"]
        print(f"\n{'='*80}")
        print(f"PAGE: {label}")
        if case["baseline_recall"]:
            print(f"Dumb-tile baseline: {case['baseline_recall']:.1%}")
        print(f"{'='*80}")

        doc  = fitz.open(str(pdf_path))
        page = doc[case["page_idx"]]
        pw, ph = page.rect.width, page.rect.height
        gt_all_text  = page.get_text()
        gt_all_words = set(normalize(gt_all_text).split())
        doc.close()
        print(f"Whole-page GT: {len(gt_all_words)} unique words")

        # ── Step 1: Detect regions from PDF text blocks ───────────────────
        print(f"\n── Step 1: PDF text-block region detection ─────────────────")
        doc  = fitz.open(str(pdf_path))
        page = doc[case["page_idx"]]
        regions = classify_text_blocks(page)
        doc.close()

        print(f"\n  Regions found ({len(regions)}):")
        for rtype, r in regions.items():
            rw = int((r["x2"]-r["x1"]) * pw * RENDER_DPI/72)
            rh = int((r["y2"]-r["y1"]) * ph * RENDER_DPI/72)
            print(f"    {rtype:14s}  bbox=({r['x1']:.3f},{r['y1']:.3f})-({r['x2']:.3f},{r['y2']:.3f})  "
                  f"≈{rw}×{rh}px @ {RENDER_DPI}dpi")

        # ── Step 2: Debug visualization ───────────────────────────────────
        print(f"\n── Step 2: Debug visualization ─────────────────────────────")
        debug_path = OUT_DIR / f"{label}_debug_regions.png"
        save_debug_image(pdf_path, case["page_idx"], regions, debug_path, dpi=120)

        # ── Step 3: Optional rotation detection for small panels ──────────
        print(f"\n── Step 3: Rotation detection for text panels ───────────────")
        for rtype in ("NOTES", "TITLE_BLOCK"):
            if rtype not in regions:
                continue
            r = regions[rtype]
            # Only run rotation check on narrow strips (width < 20% of page)
            width_frac = r["x2"] - r["x1"]
            if width_frac < 0.25:
                strip_img = render_region(pdf_path, case["page_idx"], r, dpi=150)
                strip_path = OUT_DIR / f"{label}_{rtype}_strip.png"
                strip_img.save(str(strip_path))
                rot = detect_rotation(client, strip_path)
                regions[rtype]["rotation"] = rot
                print(f"  {rtype}: strip width={width_frac:.2f} → rotation detected: {rot}°")
            else:
                regions[rtype]["rotation"] = 0
                print(f"  {rtype}: wide panel (width={width_frac:.2f}) → assumed upright (0°)")

        # ── Step 4: Extract from each region ─────────────────────────────
        print(f"\n── Step 4: Extraction ───────────────────────────────────────")
        all_texts    = []
        total_time   = 0.0
        region_stats = []

        for rtype, r in regions.items():
            print(f"\n  ── {rtype}: bbox=({r['x1']:.3f},{r['y1']:.3f})-({r['x2']:.3f},{r['y2']:.3f})")

            # Ground truth for this region
            doc  = fitz.open(str(pdf_path))
            page = doc[case["page_idx"]]
            rgt  = spatial_gt(page, r)
            doc.close()
            print(f"     Region GT: {len(rgt)} unique words")

            # Render from PDF at full DPI
            region_img  = render_region(pdf_path, case["page_idx"], r, RENDER_DPI)
            rw, rh = region_img.size

            # Apply rotation if detected
            rot = r.get("rotation", 0)
            if rot:
                region_img = region_img.rotate(rot, expand=True)
                rw, rh = region_img.size
                print(f"     Rotated {rot}° → {rw}×{rh}px")

            region_path = OUT_DIR / f"{label}_{rtype}.png"
            region_img.save(str(region_path))

            longest = max(rw, rh)
            region_texts = []
            region_time  = 0.0

            if longest <= TILE_THRESHOLD:
                # Direct extraction
                print(f"     {rw}×{rh}px → direct extraction")
                res = extract_text(client, region_path, rtype)
                if res["error"]:
                    print(f"     ERROR: {res['error']}")
                else:
                    region_time += res["elapsed"]
                    region_texts.append(res["text"] or "")
                    m = word_overlap(rgt, res["text"] or "")
                    print(f"     {res['elapsed']:.1f}s | {len(res['text'] or '')} chars | "
                          f"recall={m['recall']:.1%} prec={m['precision']:.1%} "
                          f"[{m['matched']}/{m['gt_n']} words]")
                    preview = (res["text"] or "")[:200].replace('\n',' ')
                    print(f"     Preview: {preview}")
            else:
                # Tile within region
                tiles, rows, cols = tile_image(region_img, TILE_THRESHOLD, TILE_OVERLAP,
                                               f"{label}_{rtype}")
                print(f"     {rw}×{rh}px → tiling {rows}×{cols} ({len(tiles)} tiles)")
                for t in tiles:
                    ctx = (f"Tile row {t['row']} col {t['col']} of {rows}×{cols} "
                           f"within the {rtype.lower().replace('_',' ')} region.")
                    res = extract_text(client, t["path"], rtype, tile_ctx=ctx)
                    if res["error"]:
                        print(f"     [{t['row']},{t['col']}] ERROR: {res['error']}")
                    else:
                        region_time += res["elapsed"]
                        region_texts.append(res["text"] or "")
                        print(f"     [{t['row']},{t['col']}] {res['elapsed']:.1f}s | "
                              f"{len(res['text'] or '')} chars")
                combined_region = "\n".join(region_texts)
                m = word_overlap(rgt, combined_region)
                print(f"     Region combined: recall={m['recall']:.1%} prec={m['precision']:.1%} "
                      f"[{m['matched']}/{m['gt_n']} words]")
                if m["missed"]:
                    print(f"     Missed: {', '.join(m['missed'][:10])}")

            all_texts.extend(region_texts)
            total_time += region_time
            region_stats.append({
                "type": rtype, "region_gt": len(rgt),
                "texts": region_texts, "time": region_time,
            })

        # ── Step 5: Whole-page results ────────────────────────────────────
        print(f"\n── Step 5: Whole-page combined results ──────────────────────")
        combined = "\n".join(all_texts)
        m = word_overlap(gt_all_words, combined)

        print(f"  Total time:    {total_time:.1f}s")
        print(f"  Regions:       {len(regions)}")
        print(f"  Total chars:   {len(combined)}")
        print(f"\n  RECALL:    {m['recall']:.1%}  ({m['matched']}/{m['gt_n']} words)")
        print(f"  PRECISION: {m['precision']:.1%}")
        print(f"  F1:        {m['f1']:.1%}")

        if case["baseline_recall"]:
            delta  = m["recall"] - case["baseline_recall"]
            sign   = "+" if delta >= 0 else ""
            verdict = "BETTER ✓" if delta > 0.02 else ("WORSE ✗" if delta < -0.02 else "≈ SAME")
            print(f"\n  vs dumb-tile baseline: {sign}{delta:.1%}  [{verdict}]")

        if m["missed"]:
            print(f"\n  MISSED:       {', '.join(m['missed'][:15])}")
        if m["extra"]:
            print(f"  HALLUCINATED: {', '.join(m['extra'][:12])}")

        # Region summary table
        print(f"\n  Region breakdown:")
        print(f"  {'Type':<14} {'GT words':>8}  {'Recall':>7}  {'Time':>6}")
        print(f"  {'-'*14} {'-'*8}  {'-'*7}  {'-'*6}")
        for rs in region_stats:
            reg_combined = "\n".join(rs["texts"])
            rtype = rs["type"]
            rgt   = set()  # rebuild from doc
            doc   = fitz.open(str(pdf_path))
            pg    = doc[case["page_idx"]]
            rgt   = spatial_gt(pg, regions[rtype])
            doc.close()
            rm = word_overlap(rgt, reg_combined) if rgt else {"recall":0}
            recall_str = f"{rm['recall']:.1%}" if rgt else "  n/a"
            print(f"  {rtype:<14} {len(rgt):>8}  {recall_str:>7}  {rs['time']:>5.0f}s")

        print(f"\n  Debug regions image: {OUT_DIR}/{label}_debug_regions.png")

    print(f"\n{'='*80}")
    print("DONE  —  outputs saved to experiments/segments/exp09/")
    print("="*80)


if __name__ == "__main__":
    main()
