#!/usr/bin/env python3
"""Experiment 08: Max recall on DRAWING_AREA.

Uses the scout region bounds from exp07 to crop the drawing area, then
re-renders from the vector PDF at 300 DPI (sharper fine annotations)
and tiles at a smaller threshold with higher overlap.

Compares against exp07 baseline of 90.2% on the drawing area region.

Variables tested:
  - DPI: 200 vs 300 (re-rendered from PDF vector data)
  - Tile threshold: 2000px (exp07) vs 1500px vs 1200px
  - Overlap: 15% (exp07) vs 25%
"""

import base64
import re
import time
from pathlib import Path
from PIL import Image
from openai import OpenAI
import fitz  # PyMuPDF

LMSTUDIO_HOST = "http://localhost:1234"
SAMPLES_DIR   = Path(__file__).parent / "samples"
OUT_DIR       = Path(__file__).parent / "segments" / "exp08"
PDF_DIR       = Path(__file__).parent.parent / "Sample Documents"

OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "google/gemma-3-27b"

# Drawing area bounds from exp07 scout (floor plan p048)
# bbox=(0.35,0.00)-(0.98,1.00) — covers the main fire suppression plan
DRAWING_REGION = {"x1": 0.35, "y1": 0.0, "x2": 0.98, "y2": 1.0}

TEST_CASE = {
    "pdf":      "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf",
    "page_idx": 47,
    "image":    "PN92107_JSOC_MCC_V3__p048_200dpi.png",
    "label":    "floor_plan p048",
}

EXTRACT_PROMPT = (
    "This is a section of the main drawing area of a construction drawing. "
    "Extract ALL visible text exactly as written — every character matters: "
    "room names, room numbers, space labels, door/window tags, equipment tags, "
    "dimensions (ALL numeric values including fractions), callout text, "
    "north arrow label, scale bar text, revision cloud text, "
    "sheet cross-references (e.g. 'SEE SHEET E-101'), grid line labels, "
    "level/floor labels, any stamp or seal text visible. "
    "Do NOT describe what you see. Output raw text only, one item per line. "
    "If text is small or partially obscured, make your best reading and include it."
)

CONFIGS = [
    # (label,          dpi,  threshold, overlap)
    ("200dpi 2000px 15%",  200, 2000, 0.15),   # exp07 baseline
    ("300dpi 2000px 15%",  300, 2000, 0.15),   # higher DPI, same tiles
    ("300dpi 1500px 25%",  300, 1500, 0.25),   # smaller tiles, more overlap
    ("300dpi 1200px 25%",  300, 1200, 0.25),   # finest tiles
]


def render_region_from_pdf(pdf_path: Path, page_idx: int,
                            region: dict, dpi: int) -> Image.Image:
    """Re-render just the drawing area region from the PDF vector data at given DPI."""
    doc  = fitz.open(str(pdf_path))
    page = fitz.open(str(pdf_path))[page_idx]
    pw   = page.rect.width   # points
    ph   = page.rect.height

    # Clip rect in PDF points
    clip = fitz.Rect(
        region["x1"] * pw, region["y1"] * ph,
        region["x2"] * pw, region["y2"] * ph,
    )
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    doc.close()

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def tile_image(img: Image.Image, threshold: int, overlap: float,
               stem: str) -> list[dict]:
    """Tile image keeping longest tile side <= threshold, with overlap."""
    w, h = img.size
    cols = max(1, -(-w // threshold))   # ceiling division
    rows = max(1, -(-h // threshold))
    tw   = w / cols
    th   = h / rows
    ow   = int(tw * overlap)
    oh   = int(th * overlap)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tw) - ow)
            y1 = max(0, int(r * th) - oh)
            x2 = min(w, int((c+1)*tw) + ow)
            y2 = min(h, int((r+1)*th) + oh)
            crop = img.crop((x1, y1, x2, y2))
            path = OUT_DIR / f"{stem}_t{r}x{c}.png"
            crop.save(str(path))
            tiles.append({
                "path": path, "row": r, "col": c,
                "size": f"{x2-x1}×{y2-y1}",
            })
    return tiles, rows, cols


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def extract_text(client: OpenAI, image_path: Path, tile_ctx: str = "") -> dict:
    prompt = EXTRACT_PROMPT
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
            max_tokens=4096,
            temperature=0.05,
        )
        return {"text": resp.choices[0].message.content,
                "elapsed": time.time()-start, "error": None}
    except Exception as e:
        return {"text": None, "elapsed": time.time()-start, "error": str(e)}


def normalize(text: str) -> str:
    text = re.sub(r'[*#_`\-\|>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def word_overlap(gt: set[str], extracted: str) -> dict:
    ex = set(normalize(extracted).split())
    if not gt:
        return {"precision":0,"recall":0,"f1":0,"gt_n":0,"ex_n":0,"matched":0,"missed":[],"extra":[]}
    matched = gt & ex
    p = len(matched)/len(ex) if ex else 0
    r = len(matched)/len(gt)
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {
        "precision":p,"recall":r,"f1":f1,
        "gt_n":len(gt),"ex_n":len(ex),"matched":len(matched),
        "missed":sorted(gt-ex)[:20],
        "extra": sorted(ex-gt)[:15],
    }


def get_region_gt(pdf_path: Path, page_idx: int, region: dict) -> set[str]:
    """Spatial ground truth words within the drawing area bounds."""
    doc  = fitz.open(str(pdf_path))
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    clip = fitz.Rect(region["x1"]*pw, region["y1"]*ph,
                     region["x2"]*pw, region["y2"]*ph)
    words = page.get_text("words", clip=clip)
    doc.close()
    return {normalize(w[4]) for w in words if normalize(w[4])}


def main():
    client = OpenAI(base_url=f"{LMSTUDIO_HOST}/v1", api_key="not-needed")
    pdf_path = PDF_DIR / TEST_CASE["pdf"]

    print("=" * 80)
    print("EXPERIMENT 08: DRAWING_AREA MAX RECALL — DPI × TILE SIZE × OVERLAP")
    print(f"Model: {MODEL}  |  Host: {LMSTUDIO_HOST}")
    print(f"Page: {TEST_CASE['label']}")
    print(f"Drawing region: x={DRAWING_REGION['x1']:.2f}–{DRAWING_REGION['x2']:.2f}, "
          f"y={DRAWING_REGION['y1']:.2f}–{DRAWING_REGION['y2']:.2f}")
    print("=" * 80)

    # Ground truth for the drawing area
    region_gt = get_region_gt(pdf_path, TEST_CASE["page_idx"], DRAWING_REGION)
    print(f"\nDrawing area GT (spatial): {len(region_gt)} unique words")
    print(f"GT sample: {sorted(region_gt)[:20]}")

    results_summary = []

    for label, dpi, threshold, overlap in CONFIGS:
        print(f"\n{'='*80}")
        print(f"CONFIG: {label}  (dpi={dpi}, tile_threshold={threshold}px, overlap={overlap:.0%})")
        print(f"{'='*80}")

        # Render from PDF at target DPI
        region_img = render_region_from_pdf(pdf_path, TEST_CASE["page_idx"],
                                             DRAWING_REGION, dpi)
        rw, rh = region_img.size
        print(f"Rendered: {rw}×{rh}px at {dpi} DPI")

        # Save the region image for reference
        safe_label = label.replace(" ", "_")
        region_path = OUT_DIR / f"drawing_area_{safe_label}.png"
        region_img.save(str(region_path))

        # Tile
        tiles, rows, cols = tile_image(region_img, threshold, overlap, safe_label)
        n_tiles = len(tiles)
        tile_sizes = [t["size"] for t in tiles[:3]]
        print(f"Tiling: {rows}×{cols} = {n_tiles} tiles  (sample sizes: {', '.join(tile_sizes)})")

        # Extract from each tile
        all_texts = []
        total_time = 0.0
        for t in tiles:
            ctx = (f"Tile row {t['row']} col {t['col']} of {rows}×{cols} grid "
                   f"within the main drawing area.")
            res = extract_text(client, t["path"], tile_ctx=ctx)
            if res["error"]:
                print(f"  [{t['row']},{t['col']}] ERROR: {res['error']}")
            else:
                total_time += res["elapsed"]
                text = res["text"] or ""
                all_texts.append(text)
                print(f"  [{t['row']},{t['col']}] {res['elapsed']:.1f}s | {len(text)} chars")

        combined = "\n".join(all_texts)
        m = word_overlap(region_gt, combined)

        print(f"\n  RECALL:    {m['recall']:.1%}  ({m['matched']}/{m['gt_n']} words)")
        print(f"  PRECISION: {m['precision']:.1%}")
        print(f"  F1:        {m['f1']:.1%}")
        print(f"  Total time: {total_time:.1f}s")
        if m["missed"]:
            print(f"  MISSED: {', '.join(m['missed'][:15])}")
        if m["extra"]:
            print(f"  HALLUCINATED sample: {', '.join(m['extra'][:10])}")

        results_summary.append({
            "label": label, "dpi": dpi, "threshold": threshold,
            "overlap": overlap, "tiles": n_tiles,
            "recall": m["recall"], "precision": m["precision"],
            "f1": m["f1"], "time": total_time,
            "missed": m["missed"],
        })

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY — Drawing Area Recall by Config")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'DPI':>4} {'Thresh':>7} {'Ovlp':>5} {'Tiles':>6} "
          f"{'Recall':>7} {'Prec':>7} {'F1':>7} {'Time':>7}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['label']:<25} {r['dpi']:>4} {r['threshold']:>7} "
              f"{r['overlap']:>5.0%} {r['tiles']:>6} "
              f"{r['recall']:>7.1%} {r['precision']:>7.1%} "
              f"{r['f1']:>7.1%} {r['time']:>6.0f}s")

    best = max(results_summary, key=lambda x: x["recall"])
    print(f"\nBest recall: {best['label']} → {best['recall']:.1%}")
    print(f"Remaining missed ({len(best['missed'])} words): {', '.join(best['missed'])}")


if __name__ == "__main__":
    main()
