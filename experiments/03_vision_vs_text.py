#!/usr/bin/env python3
"""Experiment 03: Vision model vs embedded PDF text comparison.

Extracts embedded text from PDF pages as ground truth, then asks the vision
model to extract ALL text from the same page image. Compares to measure accuracy.
"""

import base64
import time
import difflib
import re
from pathlib import Path
from openai import OpenAI
import fitz  # PyMuPDF

LMSTUDIO_HOST = "http://192.168.215.90:1234"
SAMPLES_DIR = Path(__file__).parent / "samples"
PDF_DIR = Path(__file__).parent.parent / "Sample Documents"

# Test pages: (pdf_filename, page_index_0based, image_filename, page_type)
TEST_PAGES = [
    ("PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf", 0, "PN92107_JSOC_MCC_V3__p001_200dpi.png", "cover_sheet"),
    ("PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf", 47, "PN92107_JSOC_MCC_V3__p048_200dpi.png", "floor_plan"),
    ("PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf", 94, "PN92107_JSOC_MCC_V3__p095_200dpi.png", "schedule"),
    ("PN92107_JSOC_MCC_V3_RTA_W912PM26RA007_UnLocked.pdf", 141, "PN92107_JSOC_MCC_V3__p142_200dpi.png", "diagram"),
    ("PN92107_JSOC_MCC_V4_RTA_W912PM26RA007_UnLocked.pdf", 96, "PN92107_JSOC_MCC_V4__p097_200dpi.png", "abbreviations"),
]

VISION_MODELS = [
    "google/gemma-3-27b",
]


def extract_pdf_text(pdf_name: str, page_idx: int) -> str:
    """Extract embedded text from a specific PDF page."""
    doc = fitz.open(str(PDF_DIR / pdf_name))
    page = doc[page_idx]
    text = page.get_text()
    doc.close()
    return text


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def vision_extract(client: OpenAI, model: str, image_path: Path) -> dict:
    """Ask vision model to extract ALL text from the page."""
    b64 = encode_image(image_path)

    prompt = (
        "Extract ALL text visible in this construction drawing image. "
        "Include every single piece of text you can read: sheet numbers, titles, "
        "room numbers, labels, annotations, notes, dimensions, table contents, "
        "legend entries, abbreviations — everything. "
        "Do NOT summarize or paraphrase. Output the raw text exactly as written. "
        "Organize by region of the page (top-left, center, bottom-right, etc.) "
        "if that helps, but capture EVERY piece of text."
    )

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }],
            max_tokens=4096,
            temperature=0.1,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else None
        return {"text": text, "elapsed": elapsed, "tokens": tokens, "error": None}
    except Exception as e:
        elapsed = time.time() - start
        return {"text": None, "elapsed": elapsed, "tokens": None, "error": str(e)}


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, remove markdown."""
    text = re.sub(r'[*#_`\-\|]', '', text)  # remove markdown chars
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def word_overlap(ground_truth: str, extracted: str) -> dict:
    """Calculate word-level overlap between ground truth and extracted text."""
    gt_words = set(normalize_text(ground_truth).split())
    ex_words = set(normalize_text(extracted).split())

    if not gt_words:
        return {"precision": 0, "recall": 0, "f1": 0, "gt_count": 0, "ex_count": 0}

    matched = gt_words & ex_words
    precision = len(matched) / len(ex_words) if ex_words else 0
    recall = len(matched) / len(gt_words) if gt_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_count": len(gt_words),
        "ex_count": len(ex_words),
        "matched": len(matched),
        "missed_sample": list(gt_words - ex_words)[:20],
        "extra_sample": list(ex_words - gt_words)[:20],
    }


def main():
    client = OpenAI(base_url=f"{LMSTUDIO_HOST}/v1", api_key="not-needed")

    print("=" * 80)
    print("EXPERIMENT 03: VISION MODEL vs EMBEDDED PDF TEXT")
    print(f"Host: {LMSTUDIO_HOST}")
    print("=" * 80)

    for model in VISION_MODELS:
        print(f"\nMODEL: {model}")

        for pdf_name, page_idx, img_name, page_type in TEST_PAGES:
            image_path = SAMPLES_DIR / img_name
            if not image_path.exists():
                print(f"\n  SKIP: {img_name} not found")
                continue

            print(f"\n{'='*80}")
            print(f"  Page: {img_name} ({page_type})")
            print(f"  PDF: {pdf_name}, page {page_idx + 1}")
            print(f"{'='*80}")

            # Extract ground truth from PDF
            gt_text = extract_pdf_text(pdf_name, page_idx)
            print(f"\n  [Embedded PDF Text] ({len(gt_text)} chars, {len(gt_text.split())} words)")
            # Show first 500 chars
            for line in gt_text[:500].split("\n"):
                if line.strip():
                    print(f"    GT: {line.strip()}")
            if len(gt_text) > 500:
                print(f"    ... ({len(gt_text) - 500} more chars)")

            # Vision model extraction
            print(f"\n  [Vision Model Extraction]")
            result = vision_extract(client, model, image_path)

            if result["error"]:
                print(f"  ERROR: {result['error']}")
                continue

            print(f"  Time: {result['elapsed']:.1f}s | Tokens: {result['tokens']}")
            extracted = result["text"]
            print(f"  Extracted: {len(extracted)} chars")
            # Show first 500 chars
            for line in extracted[:500].split("\n"):
                if line.strip():
                    print(f"    VIS: {line.strip()}")
            if len(extracted) > 500:
                print(f"    ... ({len(extracted) - 500} more chars)")

            # Compare
            print(f"\n  [Word-Level Comparison]")
            metrics = word_overlap(gt_text, extracted)
            print(f"  Ground truth unique words: {metrics['gt_count']}")
            print(f"  Vision model unique words: {metrics['ex_count']}")
            print(f"  Matched words:             {metrics['matched']}")
            print(f"  Precision: {metrics['precision']:.1%} (of vision words, how many are in GT)")
            print(f"  Recall:    {metrics['recall']:.1%} (of GT words, how many did vision find)")
            print(f"  F1 Score:  {metrics['f1']:.1%}")

            if metrics['missed_sample']:
                print(f"\n  Sample MISSED words (in GT but not vision):")
                print(f"    {', '.join(metrics['missed_sample'][:15])}")
            if metrics['extra_sample']:
                print(f"\n  Sample EXTRA words (in vision but not GT):")
                print(f"    {', '.join(metrics['extra_sample'][:15])}")


if __name__ == "__main__":
    main()
