#!/usr/bin/env python3
"""Experiment 06: Vision-model-guided region detection vs dumb tiling.

Strategy:
  1. Scout pass: send a downsampled thumbnail to the vision model and ask it
     to identify semantic regions (TITLE_BLOCK, NOTES, DRAWING_AREA, LEGEND, SCHEDULE)
     with normalized bounding boxes.
  2. Crop: extract those regions at full resolution from the 200 DPI image.
  3. Extract: run targeted extraction on each crop with region-appropriate prompts.
  4. Compare: measure recall vs the 2x3 tiling baseline (87%).

Hypothesis: semantic crops beat dumb tiles because:
  - Title block is small + dense -> its own isolated crop
  - Notes don't get split across tile borders
  - Empty whitespace is skipped entirely
  - Per-region prompts can be tuned to content type
"""

import base64
import json
import re
import time
from pathlib import Path
from PIL import Image
from openai import OpenAI
import fitz  # PyMuPDF

LMSTUDIO_HOST = "http://localhost:1234"
SAMPLES_DIR = Path(__file__).parent / "samples"
SEGMENTS_DIR = Path(__file__).parent / "segments"
PDF_DIR = Path(__file__).parent.parent / "Sample Documents"

SEGMENTS_DIR.mkdir(exist_ok=True)

MODEL = "google/gemma-3-27b"

# Test pages - same as prior experiments for direct comparison
TEST_CASES = [
    {
        "pdf": "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf",
        "page_idx": 47,
        "image": "PN92107_JSOC_MCC_V3__p048_200dpi.png",
        "type": "floor_plan",
        "tile_2x3_recall": 0.87,  # from experiment 04
    },
    {
        "pdf": "PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf",
        "page_idx": 94,
        "image": "PN92107_JSOC_MCC_V3__p095_200dpi.png",
        "type": "schedule",
        "tile_2x3_recall": None,  # unknown, will measure
    },
]

# Prompt to detect semantic regions — returns normalized bbox coords
SCOUT_PROMPT = """You are analyzing a construction drawing (architectural/engineering document).

Identify the distinct semantic regions on this page. For each region found, return a JSON object.

Region types to look for:
- TITLE_BLOCK: The bordered box (usually lower-right) with project name, sheet number, date, engineer info
- NOTES: A block of numbered or bulleted general notes / specifications (usually upper-left or left side)
- DRAWING_AREA: The main drawing content (floor plan, wiring diagram, elevation, etc.)
- LEGEND: A symbol key or legend table explaining drawing symbols
- SCHEDULE: A data table (equipment schedule, door schedule, panel schedule, etc.)
- OTHER: Any significant region that doesn't fit above

Return ONLY a JSON array. Each item must have:
  "type": one of the region types above
  "label": short description (e.g. "General Notes", "Floor Plan Level 1", "Electrical Panel Schedule")
  "x1": left edge as fraction of image width (0.0 to 1.0)
  "y1": top edge as fraction of image height (0.0 to 1.0)
  "x2": right edge as fraction of image width (0.0 to 1.0)
  "y2": bottom edge as fraction of image height (0.0 to 1.0)

Example format:
[
  {"type": "TITLE_BLOCK", "label": "Project title block", "x1": 0.75, "y1": 0.85, "x2": 1.0, "y2": 1.0},
  {"type": "NOTES", "label": "General Notes", "x1": 0.0, "y1": 0.0, "x2": 0.25, "y2": 0.7},
  {"type": "DRAWING_AREA", "label": "Floor Plan", "x1": 0.25, "y1": 0.0, "x2": 1.0, "y2": 0.85}
]

Only return the JSON array — no other text."""

# Per-region extraction prompts
EXTRACTION_PROMPTS = {
    "TITLE_BLOCK": (
        "This is the title block from a construction drawing. "
        "Extract ALL text exactly as written: project name, address, sheet number, revision, "
        "date, engineer/architect names, drawing title, scale, and any other fields. "
        "Output raw text only, one field per line."
    ),
    "NOTES": (
        "This is the General Notes or specifications section of a construction drawing. "
        "Extract ALL text exactly as written, preserving note numbers and structure. "
        "Include every abbreviation, specification reference, and instruction. "
        "Output raw text only."
    ),
    "DRAWING_AREA": (
        "This is the main drawing area of a construction drawing. "
        "Extract ALL visible text: room names, room numbers, door/window tags, "
        "equipment labels, dimensions, annotations, callouts, and symbol labels. "
        "Output raw text only, one item per line."
    ),
    "LEGEND": (
        "This is a legend or symbol key from a construction drawing. "
        "Extract ALL text: symbol names, abbreviations, and their descriptions. "
        "Preserve the symbol-to-description pairing where possible. "
        "Output raw text only."
    ),
    "SCHEDULE": (
        "This is a schedule or data table from a construction drawing. "
        "Extract ALL text from every cell: headers, row labels, and all values. "
        "Preserve row structure where possible. "
        "Output raw text only."
    ),
    "OTHER": (
        "Extract ALL text visible in this section of a construction drawing. "
        "Include every piece of text you can read. "
        "Output raw text only."
    ),
}


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_image(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_thumbnail(image_path: Path, max_width: int = 2000) -> Image.Image:
    """Downsample for scout pass — still large enough to read all regions clearly."""
    img = Image.open(image_path)
    w, h = img.size
    if w > max_width:
        scale = max_width / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def scout_regions(client: OpenAI, image_path: Path) -> list[dict]:
    """Scout pass: ask vision model to identify semantic regions with bounding boxes."""
    thumb = make_thumbnail(image_path, max_width=1200)
    b64 = encode_pil_image(thumb)
    tw, th = thumb.size

    print(f"  Scout thumbnail: {tw}x{th}px")

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SCOUT_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }],
            max_tokens=2048,
            temperature=0.05,
        )
        elapsed = time.time() - start
        raw = response.choices[0].message.content
        print(f"  Scout response ({elapsed:.1f}s):\n{raw[:600]}")

        # Parse JSON — handle markdown code blocks if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        regions = json.loads(cleaned)

        # Validate and clamp
        valid = []
        for r in regions:
            try:
                x1 = max(0.0, min(1.0, float(r["x1"])))
                y1 = max(0.0, min(1.0, float(r["y1"])))
                x2 = max(0.0, min(1.0, float(r["x2"])))
                y2 = max(0.0, min(1.0, float(r["y2"])))
                if x2 > x1 and y2 > y1:
                    valid.append({
                        "type": r.get("type", "OTHER").upper(),
                        "label": r.get("label", ""),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })
            except (KeyError, ValueError, TypeError):
                continue
        return valid

    except json.JSONDecodeError as e:
        print(f"  Scout JSON parse error: {e}")
        print(f"  Raw response: {raw[:300]}")
        return []
    except Exception as e:
        print(f"  Scout error: {e}")
        return []


def crop_region(image_path: Path, region: dict, save_path: Path) -> Image.Image:
    """Crop a region from the full-res image using normalized coords."""
    img = Image.open(image_path)
    w, h = img.size
    x1 = int(region["x1"] * w)
    y1 = int(region["y1"] * h)
    x2 = int(region["x2"] * w)
    y2 = int(region["y2"] * h)
    crop = img.crop((x1, y1, x2, y2))
    crop.save(str(save_path))
    img.close()
    return crop


def extract_region_text(client: OpenAI, image_path: Path, region_type: str) -> dict:
    """Extract text from a cropped region using a region-appropriate prompt."""
    prompt = EXTRACTION_PROMPTS.get(region_type, EXTRACTION_PROMPTS["OTHER"])
    b64 = encode_image(image_path)

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
            max_tokens=4096,
            temperature=0.05,
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
    return {
        "precision": p, "recall": r, "f1": f1,
        "gt_count": len(gt_words), "ex_count": len(ex_words), "matched": len(matched),
        "missed": list(gt_words - ex_words)[:15],
        "extra": list(ex_words - gt_words)[:15],
    }


def main():
    client = OpenAI(base_url=f"{LMSTUDIO_HOST}/v1", api_key="not-needed")

    print("=" * 80)
    print("EXPERIMENT 06: VISION-MODEL SEMANTIC SEGMENTATION vs DUMB TILING")
    print(f"Model: {MODEL} | Host: {LMSTUDIO_HOST}")
    print("=" * 80)

    for case in TEST_CASES:
        image_path = SAMPLES_DIR / case["image"]
        if not image_path.exists():
            print(f"\nSKIP: {case['image']} not found in {SAMPLES_DIR}")
            continue

        print(f"\n{'='*80}")
        print(f"PAGE: {case['image']} ({case['type']})")
        if case["tile_2x3_recall"]:
            print(f"Baseline (2x3 dumb tiling recall): {case['tile_2x3_recall']:.1%}")
        print(f"{'='*80}")

        # Ground truth
        doc = fitz.open(str(PDF_DIR / case["pdf"]))
        gt_text = doc[case["page_idx"]].get_text()
        doc.close()
        gt_words = set(normalize(gt_text).split())
        print(f"\nGround truth: {len(gt_text)} chars, {len(gt_words)} unique words")

        # Image info
        img = Image.open(image_path)
        iw, ih = img.size
        img.close()
        print(f"Image size: {iw}x{ih}px")

        # --- STEP 1: Scout pass ---
        print(f"\n--- STEP 1: Scout pass (region detection) ---")
        regions = scout_regions(client, image_path)

        if not regions:
            print("  No regions detected — scout pass failed. Check model output above.")
            continue

        print(f"\n  Detected {len(regions)} region(s):")
        for i, r in enumerate(regions):
            px_w = int((r["x2"] - r["x1"]) * iw)
            px_h = int((r["y2"] - r["y1"]) * ih)
            print(f"    [{i}] {r['type']:12s} | {r['label'][:40]:40s} | "
                  f"bbox=({r['x1']:.2f},{r['y1']:.2f})-({r['x2']:.2f},{r['y2']:.2f}) | "
                  f"{px_w}x{px_h}px")

        # --- STEP 2: Crop regions ---
        print(f"\n--- STEP 2: Crop regions at full resolution ---")
        stem = image_path.stem
        for i, r in enumerate(regions):
            save_path = SEGMENTS_DIR / f"{stem}_seg{i:02d}_{r['type']}.png"
            crop_region(image_path, r, save_path)
            print(f"  Saved: {save_path.name}")
            r["crop_path"] = save_path

        # --- STEP 3: Extract text from each region ---
        print(f"\n--- STEP 3: Targeted extraction per region ---")
        all_extracted = []
        total_time = 0.0

        for i, r in enumerate(regions):
            rtype = r["type"]
            print(f"\n  Region [{i}] {rtype}: {r['label']}")
            result = extract_region_text(client, r["crop_path"], rtype)

            if result["error"]:
                print(f"    ERROR: {result['error']}")
                continue

            total_time += result["elapsed"]
            text = result["text"] or ""
            m = word_overlap(gt_text, text)
            print(f"    Time: {result['elapsed']:.1f}s | Chars: {len(text)}")
            print(f"    Region recall: {m['recall']:.1%} | Precision: {m['precision']:.1%}")

            # Preview
            preview = text[:200].replace('\n', ' ')
            print(f"    Preview: {preview}...")

            all_extracted.append(text)

        # --- STEP 4: Combined metrics ---
        print(f"\n--- STEP 4: Combined results ---")
        combined = "\n".join(all_extracted)
        metrics = word_overlap(gt_text, combined)

        print(f"  Total extraction time: {total_time:.1f}s")
        print(f"  Regions processed:     {len(all_extracted)}")
        print(f"  Combined text:         {len(combined)} chars")
        print(f"\n  RECALL:    {metrics['recall']:.1%}  (words found / ground truth words)")
        print(f"  PRECISION: {metrics['precision']:.1%}  (valid words / all extracted words)")
        print(f"  F1:        {metrics['f1']:.1%}")
        print(f"  Matched:   {metrics['matched']}/{metrics['gt_count']} unique words")

        if case["tile_2x3_recall"]:
            delta = metrics["recall"] - case["tile_2x3_recall"]
            sign = "+" if delta >= 0 else ""
            print(f"\n  vs 2x3 tiling baseline: {sign}{delta:.1%}  "
                  f"({'BETTER' if delta >= 0 else 'WORSE'} than dumb tiling)")

        if metrics["missed"]:
            print(f"\n  Sample MISSED: {', '.join(metrics['missed'][:10])}")
        if metrics["extra"]:
            print(f"  Sample HALLUCINATED: {', '.join(metrics['extra'][:10])}")

    print(f"\n{'='*80}")
    print("DONE — segment images saved to experiments/segments/")
    print("Review segments/ folder to check region cropping quality visually")
    print("="*80)


if __name__ == "__main__":
    main()
