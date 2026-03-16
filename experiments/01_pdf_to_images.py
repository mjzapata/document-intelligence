#!/usr/bin/env python3
"""Experiment 01: PDF to Images - Extract pages, analyze dimensions, identify page types."""

import fitz  # PyMuPDF
import os
import sys
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent.parent / "Sample Documents"
OUTPUT_DIR = Path(__file__).parent / "samples"
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_pdf(pdf_path: Path):
    """Analyze a PDF: page count, dimensions, embedded text presence."""
    doc = fitz.open(str(pdf_path))
    print(f"\n{'='*80}")
    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(doc)}")
    print(f"File size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*80}")

    page_sizes = {}
    for i in range(len(doc)):
        page = doc[i]
        # Dimensions in points (72 points = 1 inch)
        w_in = page.rect.width / 72
        h_in = page.rect.height / 72
        size_key = f"{w_in:.1f}x{h_in:.1f}"
        page_sizes.setdefault(size_key, []).append(i + 1)

        # Check for embedded text
        text = page.get_text().strip()
        text_chars = len(text)

        if i < 5 or (i % 20 == 0):  # Print first 5 and every 20th
            print(f"  Page {i+1:3d}: {w_in:.1f}\" x {h_in:.1f}\" | "
                  f"Text chars: {text_chars:5d} | "
                  f"{'HAS TEXT' if text_chars > 50 else 'MINIMAL/NO TEXT'}")

    print(f"\nPage size distribution:")
    for size, pages in sorted(page_sizes.items(), key=lambda x: -len(x[1])):
        print(f"  {size}\": {len(pages)} pages (e.g., pages {pages[:5]}{'...' if len(pages) > 5 else ''})")

    count = len(doc)
    doc.close()
    return count


def extract_sample_pages(pdf_path: Path, page_indices: list[int], dpi: int = 200):
    """Extract specific pages as images at given DPI."""
    doc = fitz.open(str(pdf_path))
    pdf_stem = pdf_path.stem[:20]  # Truncate long names

    for page_idx in page_indices:
        if page_idx >= len(doc):
            print(f"  Skipping page {page_idx+1} (only {len(doc)} pages)")
            continue

        page = doc[page_idx]
        # Scale factor: default PDF is 72 DPI
        scale = dpi / 72
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)

        out_name = f"{pdf_stem}_p{page_idx+1:03d}_{dpi}dpi.png"
        out_path = OUTPUT_DIR / out_name
        pix.save(str(out_path))

        w_in = page.rect.width / 72
        h_in = page.rect.height / 72
        print(f"  Saved: {out_name} ({pix.width}x{pix.height}px, {w_in:.1f}\"x{h_in:.1f}\", {out_path.stat().st_size/1024:.0f}KB)")

    doc.close()


def main():
    pdfs = sorted(SAMPLE_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in Sample Documents/")
        sys.exit(1)

    print("=" * 80)
    print("DOCUMENT INTELLIGENCE - PDF ANALYSIS")
    print("=" * 80)

    # Phase 1: Analyze all PDFs
    for pdf in pdfs:
        page_count = analyze_pdf(pdf)

    # Phase 2: Extract sample pages from each PDF
    # Extract first 3 pages + a few from the middle at 200 DPI
    print(f"\n{'='*80}")
    print("EXTRACTING SAMPLE PAGES (200 DPI)")
    print(f"{'='*80}")

    for pdf in pdfs:
        doc = fitz.open(str(pdf))
        total = len(doc)
        doc.close()

        # Pick representative pages: first 3, middle, and a few spread out
        indices = [0, 1, 2]
        if total > 6:
            indices.extend([total // 4, total // 2, 3 * total // 4])
        if total > 1:
            indices.append(total - 1)  # Last page
        indices = sorted(set(i for i in indices if i < total))

        print(f"\n{pdf.name} - extracting pages {[i+1 for i in indices]}:")
        extract_sample_pages(pdf, indices, dpi=200)

    # Phase 3: Multi-DPI comparison on one page
    print(f"\n{'='*80}")
    print("DPI COMPARISON (first drawing PDF, page 3)")
    print(f"{'='*80}")

    # Use the first non-specs PDF for DPI comparison
    drawing_pdfs = [p for p in pdfs if "Specifications" not in p.name]
    if drawing_pdfs:
        test_pdf = drawing_pdfs[0]
        for dpi in [150, 200, 300]:
            print(f"\n{dpi} DPI:")
            extract_sample_pages(test_pdf, [2], dpi=dpi)

    print(f"\n{'='*80}")
    print(f"All samples saved to: {OUTPUT_DIR}")
    print(f"Total files: {len(list(OUTPUT_DIR.glob('*.png')))}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
