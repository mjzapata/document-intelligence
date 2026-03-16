#!/usr/bin/env python3
"""Experiment 04: Tiled extraction - slice pages into tiles and extract text from each.

Compares full-page extraction vs tiled extraction against PDF embedded text.
Tests multiple tile grid sizes to find optimal chunking.
"""

import base64
import time
import re
from pathlib import Path
from PIL import Image
from openai import OpenAI
import fitz

LMSTUDIO_HOST = "http://192.168.215.90:1234"
SAMPLES_DIR = Path(__file__).parent / "samples"
TILES_DIR = Path(__file__).parent / "tiles"
PDF_DIR = Path(__file__).parent.parent / "Sample Documents"

TILES_DIR.mkdir(exist_ok=True)

MODEL = "google/gemma-3-27b"

# Test on the floor plan - most challenging page type
TEST_PDF = "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf"
TEST_PAGE_IDX = 47  # page 48 (floor plan)
TEST_IMAGE = "PN92107_JSOC_MCC_V3__p048_200dpi.png"

# Tile configurations to test: (rows, cols, overlap_pct)
TILE_CONFIGS = [
    (2, 3, 0.15),   # 6 tiles - coarse
    (3, 4, 0.20),   # 12 tiles - medium
    (4, 6, 0.20),   # 24 tiles - fine
]


def create_tiles(image_path: Path, rows: int, cols: int, overlap: float) -> list[dict]:
    """Slice image into overlapping tiles. Returns list of {path, row, col, bbox}."""
    img = Image.open(image_path)
    w, h = img.size

    tile_w = w / cols
    tile_h = h / rows
    overlap_w = int(tile_w * overlap)
    overlap_h = int(tile_h * overlap)

    tiles = []
    prefix = f"{image_path.stem}_{rows}x{cols}"

    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tile_w) - overlap_w)
            y1 = max(0, int(r * tile_h) - overlap_h)
            x2 = min(w, int((c + 1) * tile_w) + overlap_w)
            y2 = min(h, int((r + 1) * tile_h) + overlap_h)

            tile = img.crop((x1, y1, x2, y2))
            tile_path = TILES_DIR / f"{prefix}_r{r}c{c}.png"
            tile.save(str(tile_path))

            tiles.append({
                "path": tile_path,
                "row": r, "col": c,
                "bbox": (x1, y1, x2, y2),
                "size_px": f"{x2-x1}x{y2-y1}",
            })

    img.close()
    return tiles


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_vision(client: OpenAI, image_path: Path, context: str = "") -> dict:
    """Extract ALL text from image using vision model."""
    b64 = encode_image(image_path)

    prompt = (
        "Extract ALL text visible in this image section of a construction drawing. "
        "Include every piece of text: room numbers, labels, dimensions, annotations, "
        "notes, symbols, abbreviations — everything you can read. "
        "Output ONLY the raw text, no descriptions or commentary. "
        "Separate distinct text elements with newlines."
    )
    if context:
        prompt += f"\n\nContext: This is a tile from a larger drawing. {context}"

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }],
            max_tokens=2048,
            temperature=0.1,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content
        return {"text": text, "elapsed": elapsed, "error": None}
    except Exception as e:
        return {"text": None, "elapsed": time.time() - start, "error": str(e)}


def normalize(text: str) -> str:
    text = re.sub(r'[*#_`\-\|>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def word_overlap(gt: str, extracted: str) -> dict:
    gt_words = set(normalize(gt).split())
    ex_words = set(normalize(extracted).split())
    if not gt_words:
        return {"precision": 0, "recall": 0, "f1": 0, "gt_count": 0, "ex_count": 0, "matched": 0}
    matched = gt_words & ex_words
    p = len(matched) / len(ex_words) if ex_words else 0
    r = len(matched) / len(gt_words)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": p, "recall": r, "f1": f1, "gt_count": len(gt_words), "ex_count": len(ex_words), "matched": len(matched),
            "missed": list(gt_words - ex_words)[:15], "extra": list(ex_words - gt_words)[:15]}


def main():
    client = OpenAI(base_url=f"{LMSTUDIO_HOST}/v1", api_key="not-needed")

    print("=" * 80)
    print("EXPERIMENT 04: TILED EXTRACTION COMPARISON")
    print(f"Model: {MODEL} | Host: {LMSTUDIO_HOST}")
    print(f"Test page: {TEST_IMAGE} (floor plan)")
    print("=" * 80)

    # Ground truth
    doc = fitz.open(str(PDF_DIR / TEST_PDF))
    gt_text = doc[TEST_PAGE_IDX].get_text()
    doc.close()
    gt_words = set(normalize(gt_text).split())
    print(f"\nGround truth: {len(gt_text)} chars, {len(gt_words)} unique words")

    image_path = SAMPLES_DIR / TEST_IMAGE

    # Test 1: Full page (baseline)
    print(f"\n{'='*80}")
    print("FULL PAGE (baseline)")
    print(f"{'='*80}")

    result = extract_text_vision(client, image_path)
    if result["error"]:
        print(f"ERROR: {result['error']}")
    else:
        metrics = word_overlap(gt_text, result["text"])
        print(f"Time: {result['elapsed']:.1f}s")
        print(f"Recall: {metrics['recall']:.1%} | Precision: {metrics['precision']:.1%} | F1: {metrics['f1']:.1%}")
        print(f"Words: {metrics['matched']}/{metrics['gt_count']} matched")

    # Test 2: Tiled extraction at different grid sizes
    for rows, cols, overlap in TILE_CONFIGS:
        print(f"\n{'='*80}")
        print(f"TILED: {rows}x{cols} grid, {overlap:.0%} overlap")
        print(f"{'='*80}")

        tiles = create_tiles(image_path, rows, cols, overlap)
        print(f"Created {len(tiles)} tiles")
        print(f"Tile sizes: {tiles[0]['size_px']} (first tile)")

        all_text = []
        total_time = 0

        for i, tile in enumerate(tiles):
            result = extract_text_vision(
                client, tile["path"],
                context=f"Tile position: row {tile['row']}, col {tile['col']} of {rows}x{cols} grid."
            )
            if result["error"]:
                print(f"  Tile r{tile['row']}c{tile['col']}: ERROR - {result['error']}")
            else:
                total_time += result["elapsed"]
                text_len = len(result["text"]) if result["text"] else 0
                print(f"  Tile r{tile['row']}c{tile['col']}: {result['elapsed']:.1f}s, {text_len} chars")
                if result["text"]:
                    all_text.append(result["text"])

        # Combine all tile text
        combined = "\n".join(all_text)
        metrics = word_overlap(gt_text, combined)

        print(f"\n  COMBINED RESULTS:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Combined text: {len(combined)} chars")
        print(f"  Recall: {metrics['recall']:.1%} | Precision: {metrics['precision']:.1%} | F1: {metrics['f1']:.1%}")
        print(f"  Words: {metrics['matched']}/{metrics['gt_count']} matched")

        if metrics['missed']:
            print(f"  Sample MISSED: {', '.join(metrics['missed'][:10])}")


if __name__ == "__main__":
    main()
