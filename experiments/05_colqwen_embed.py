#!/usr/bin/env python3
"""Experiment 05: ColQwen2 visual embeddings for document page retrieval.

Tests whether ColQwen2 can correctly match text queries to the right pages.
"""

import time
import torch
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor

SAMPLES_DIR = Path(__file__).parent / "samples"
MAX_IMAGE_DIM = 1024  # Resize images for embedding - ColQwen2 doesn't need full res

# Pages to embed with descriptions for evaluation
TEST_PAGES = [
    ("PN92107_JSOC_MCC_V3__p001_200dpi.png", "cover_sheet", "Cover sheet - project title"),
    ("PN92107_JSOC_MCC_V3__p048_200dpi.png", "floor_plan", "Fire Suppression 2nd Floor Area C"),
    ("PN92107_JSOC_MCC_V3__p095_200dpi.png", "schedule", "Plumbing Fixture Schedule"),
    ("PN92107_JSOC_MCC_V3__p142_200dpi.png", "diagram", "Emergency Power Off / Ventilation Fan Diagram"),
    ("PN92107_JSOC_MCC___S_p586_200dpi.png", "spec_text", "Specification text page"),
]

# Queries to test retrieval
TEST_QUERIES = [
    "fire suppression floor plan",
    "plumbing fixture schedule",
    "electrical abbreviations list",
    "emergency power off diagram",
    "project cover sheet title page",
    "sheet index list of drawings",
    "concrete specification requirements",
    "ventilation fan control diagram",
    "sprinkler system layout second floor",
    "water meter schedule",
]


def main():
    print("=" * 80)
    print("EXPERIMENT 05: ColQwen2 VISUAL EMBEDDINGS")
    print("=" * 80)

    # Determine device - MPS crashes with ColQwen2, use CPU on macOS
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading ColQwen2 model...")
    start = time.time()
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.float32,  # MPS doesn't support bfloat16
    ).to(device).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Load and embed pages
    print(f"\nEmbedding {len(TEST_PAGES)} pages...")
    images = []
    page_labels = []
    for filename, ptype, desc in TEST_PAGES:
        path = SAMPLES_DIR / filename
        if not path.exists():
            print(f"  SKIP: {filename}")
            continue
        img = Image.open(path).convert("RGB")
        # Resize to manageable size for embedding
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
        images.append(img)
        page_labels.append((filename, ptype, desc))
        print(f"  Loaded: {filename} -> {img.size[0]}x{img.size[1]}")

    # Process images one at a time to avoid OOM
    print("\nGenerating image embeddings...")
    start = time.time()
    all_image_embeddings = []
    for i, img in enumerate(images):
        batch_images = processor.process_images([img]).to(device)
        with torch.no_grad():
            emb = model(**batch_images)
        all_image_embeddings.append(emb)
        print(f"  Embedded page {i+1}/{len(images)}: {emb.shape}")
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    print(f"All embeddings: {image_embeddings.shape} in {time.time() - start:.1f}s")

    # Process queries and compute similarity
    print(f"\nTesting {len(TEST_QUERIES)} queries...")
    print(f"\n{'='*80}")

    for query in TEST_QUERIES:
        batch_queries = processor.process_queries([query]).to(device)
        with torch.no_grad():
            query_embeddings = model(**batch_queries)

        # Compute MaxSim scores (late interaction)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)

        # Rank pages by score
        ranked = sorted(enumerate(scores[0].tolist()), key=lambda x: -x[1])

        print(f"\n  Query: \"{query}\"")
        print(f"  Top 3 matches:")
        for rank, (idx, score) in enumerate(ranked[:3]):
            fname, ptype, desc = page_labels[idx]
            marker = " <<<" if rank == 0 else ""
            print(f"    {rank+1}. [{score:.1f}] {ptype:15s} | {desc}{marker}")

    # Close images
    for img in images:
        img.close()

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
