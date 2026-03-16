#!/usr/bin/env python3
"""Experiment 02: Basic vision model test - send drawing pages to LMStudio and evaluate."""

import base64
import json
import time
from pathlib import Path
from openai import OpenAI

LMSTUDIO_HOST = "http://192.168.215.90:1234"
SAMPLES_DIR = Path(__file__).parent / "samples"

# Vision models to test
VISION_MODELS = [
    "google/gemma-3-27b",
    # "mistralai/magistral-small-2509",  # Uncomment to compare
]

# Test pages - mix of page types
TEST_PAGES = [
    ("PN92107_JSOC_MCC_V3__p001_200dpi.png", "cover_sheet"),
    ("PN92107_JSOC_MCC_V3__p048_200dpi.png", "floor_plan"),
    ("PN92107_JSOC_MCC_V3__p095_200dpi.png", "schedule"),
    ("PN92107_JSOC_MCC_V3__p142_200dpi.png", "diagram"),
    ("PN92107_JSOC_MCC_V4__p097_200dpi.png", "abbreviations"),
]


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_vision(client: OpenAI, model: str, image_path: Path, prompt: str, max_tokens: int = 2048) -> dict:
    b64 = encode_image(image_path)
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
            max_tokens=max_tokens,
            temperature=0.1,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else None
        return {"text": text, "elapsed": elapsed, "tokens": tokens, "error": None}
    except Exception as e:
        elapsed = time.time() - start
        return {"text": None, "elapsed": elapsed, "tokens": None, "error": str(e)}


def main():
    client = OpenAI(base_url=f"{LMSTUDIO_HOST}/v1", api_key="not-needed")

    print("=" * 80)
    print("EXPERIMENT 02: BASIC VISION MODEL TEST")
    print(f"Host: {LMSTUDIO_HOST}")
    print("=" * 80)

    # Test 1: Basic description
    describe_prompt = (
        "Describe what you see in this construction drawing image. "
        "Include: the sheet number/title if visible, what type of drawing it is "
        "(floor plan, diagram, schedule, cover sheet, etc.), and list any text, "
        "labels, symbols, or annotations you can read."
    )

    # Test 2: Page classification
    classify_prompt = (
        "Classify this construction document page into exactly ONE category:\n"
        "- COVER: Title/cover sheet\n"
        "- INDEX: Sheet list or index\n"
        "- FLOOR_PLAN: Architectural or engineering floor plan with rooms/spaces\n"
        "- SCHEDULE: Tables listing equipment, fixtures, or specifications\n"
        "- DIAGRAM: Wiring diagrams, control diagrams, riser diagrams\n"
        "- DETAIL: Enlarged detail drawings\n"
        "- LEGEND: Symbol legends, abbreviation lists, general notes\n"
        "- SPEC: Text-heavy specification pages\n\n"
        "Respond with ONLY the category name and the sheet number if visible. "
        "Format: CATEGORY | Sheet: <number>"
    )

    for model in VISION_MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")

        for filename, page_type in TEST_PAGES:
            image_path = SAMPLES_DIR / filename
            if not image_path.exists():
                print(f"\n  SKIP: {filename} not found")
                continue

            size_mb = image_path.stat().st_size / 1024 / 1024
            print(f"\n{'─'*80}")
            print(f"  Page: {filename}")
            print(f"  Expected type: {page_type} | Size: {size_mb:.1f}MB")
            print(f"{'─'*80}")

            # Classification test (quick)
            print(f"\n  [Classification]")
            result = test_vision(client, model, image_path, classify_prompt, max_tokens=100)
            if result["error"]:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Time: {result['elapsed']:.1f}s")
                print(f"  Result: {result['text'].strip()}")

            # Description test (detailed)
            print(f"\n  [Description]")
            result = test_vision(client, model, image_path, describe_prompt, max_tokens=2048)
            if result["error"]:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Time: {result['elapsed']:.1f}s | Tokens: {result['tokens']}")
                # Print first 20 lines of response
                lines = result["text"].split("\n")
                for line in lines[:20]:
                    print(f"    {line}")
                if len(lines) > 20:
                    print(f"    ... ({len(lines) - 20} more lines)")


if __name__ == "__main__":
    main()
