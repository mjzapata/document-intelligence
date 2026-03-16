# Document Intelligence — Experiment Analysis

**Date**: 2026-03-13
**Sample Documents**: JSOC MCC Construction Drawing Sets (V3: 189 pages, V4: 192 pages, Specs: 2343 pages)

---

## 1. PDF Structure Analysis

### Page Dimensions
- **Drawing pages (V3, V4)**: All 34" x 22" (D-size) — 381 pages total
- **Specification pages**: 8.5" x 11" (letter) — 2343 pages, some landscape
- At 200 DPI, drawing pages render to **6800 x 4400 pixels** (~1-13MB PNG)

### Embedded Text
- All drawing pages have embedded OCR text (760–11,500 chars/page)
- This embedded text serves as ground truth for evaluating vision model accuracy
- Spec pages are text-rich (1,000–3,000 chars/page)

### Page Types Identified
| Type | Example | Characteristics |
|------|---------|----------------|
| Cover sheet | V3 p1 | Title, project info, minimal content |
| Sheet index | V3 p3 | Dense tabular lists of all sheets |
| Floor plan | V3 p48 | Room layouts, sprinkler symbols, annotations |
| Schedule | V3 p95 | Equipment tables with small text |
| Diagram | V3 p142 | Control diagrams, wiring, parts lists |
| Abbreviations | V4 p97 | Multi-column abbreviation definitions |
| Spec text | Specs p586 | Standard letter-size text documents |

---

## 2. Vision Model: Gemma-3-27b (Q4, LMStudio)

### Classification
- **4/5 correct** on page type classification
- Misclassified abbreviations page as SCHEDULE (understandable — visually similar)
- Classification is fast (~4-5 seconds per page)
- **Verdict**: Vision model is strong for page-type routing

### Full-Page Text Extraction
Comparison against embedded PDF text (word-level overlap):

| Page Type | GT Words | Recall | Precision | F1 | Time |
|-----------|----------|--------|-----------|----|------|
| Cover sheet | 97 | 60.8% | 46.8% | 52.9% | 21s |
| Floor plan | 262 | 30.9% | 32.4% | 31.6% | 50s |
| Schedule | 438 | 16.0% | 37.0% | 22.3% | 93s |
| Diagram | 508 | 21.9% | 36.6% | 27.4% | 112s |
| Abbreviations | 968 | 16.2% | 34.6% | 22.1% | 125s |

### Key Findings
1. **Full-page vision extraction is inadequate for text capture** — recall drops below 30% on complex pages
2. The model *understands* the drawings (correct page types, spatial understanding) but **cannot read small text** at 34"x22" rendered to a single image
3. Precision is always higher than recall — the model is accurate in what it *does* read, it just misses most of it
4. Dense pages (schedules, abbreviations) are worst — too much small text for a single-image pass
5. Response quality favors summarization over exhaustive extraction, even when prompted for raw text

### Lesson Learned
> **The vision model is good for understanding, bad for reading.** At D-size page resolution compressed to a single image, small annotations, room numbers, and table contents are largely invisible to the model. Full-page vision extraction alone is NOT viable for construction drawings.

---

## 3. Tiled Extraction — The Key Breakthrough

Slicing the floor plan (34"x22", 6800x4400px) into overlapping tiles and extracting text from each tile individually:

| Approach | Tile Size (px) | Recall | Precision | F1 | Total Time |
|----------|---------------|--------|-----------|----|------------|
| Full page | 6800x4400 | 30.5% | 60.6% | 40.6% | 40s |
| **2x3 grid** | **~2600x2500** | **86.6%** | **85.3%** | **86.0%** | **72s** |
| 3x4 grid | ~2000x1750 | 92.4% | 81.2% | 86.4% | 153s |
| 4x6 grid | ~1350x1300 | 96.9% | 42.5% | 59.1% | 293s |

### Key Findings
1. **2x3 tiling is the sweet spot** — recall jumps from 31% → 87% with only 1.8x the time
2. Going finer (3x4) adds 6% recall but doubles the time — diminishing returns
3. **4x6 is too fine** — precision drops to 42% because tiles lack context and the model hallucinates/duplicates text. Also takes 5 minutes
4. Tile size of ~2500x2500px appears optimal for Gemma-3-27b Q4 on construction drawings
5. For a 34"x22" page at 200 DPI, 2x3 grid means each tile covers roughly **11"x11"** of the physical page

### Lesson Learned
> **Tiling transforms the vision model from useless to highly capable.** The right tile size (~2500px) gives the model enough resolution to read small text while retaining enough context to understand what it's looking at. The 2x3 grid achieves 87% recall — close to the embedded PDF text quality — at reasonable processing time.

### Optimal Tile Strategy for D-Size (34"x22") Drawings
- **DPI**: 200 (good balance of quality vs file size)
- **Grid**: 2 rows × 3 columns with 15-20% overlap
- **Resulting tile**: ~2600 × 2500 pixels (~13" × 11" physical area)
- **Processing**: ~72 seconds total for 6 tiles
- **Expected recall**: ~85-90%

---

## 4. ColQwen2 Visual Embeddings (Partial)

### Setup
- ColQwen2 v1.0 (2B params) loaded via `colpali-engine` Python library
- **Cannot run in LMStudio** — ColBERT-style late-interaction model, not a standard LLM
- MPS (Apple Metal) crashes — must run on CPU
- Model loads in ~5 seconds, image embedding takes ~7s per page on CPU

### Results (Partial)
- Successfully generates 128-dim multi-vector embeddings (759 tokens per page)
- Different aspect ratios produce different token counts (759 vs 755) — needs padding for batch operations
- Image embedding works at 1024px max dimension (resized from 6800px)

### Still To Test
- Query-to-page retrieval accuracy
- Whether ColQwen2 can correctly match text queries like "fire suppression floor plan" to the right pages
- Batch scoring with padded embeddings

---

## 5. Architecture Implications

Based on these experiments, the hybrid approach is confirmed:

### What Works
1. **Vision model for classification/understanding** — fast, accurate page-type routing
2. **Tiled vision extraction for text** — 2x3 grid on D-size pages gives ~87% recall
3. **Embedded PDF text as bonus** — when available, provides free ground truth
4. **ColQwen2 for visual retrieval** — runs locally, generates rich multi-vector embeddings

### Recommended Pipeline
```
PDF → Page Images (200 DPI)
  → Embedded PDF text extraction (free, from PyMuPDF)
  → Vision model: classify page type (single pass, full page)
  → Vision model: tiled text extraction (2x3 grid for D-size pages)
  → Reconcile: merge embedded text + tiled vision text
  → ColQwen2: generate visual embeddings for retrieval index
```

### Open Questions
1. **Magistral-Small vs Gemma-3-27b** — not yet compared on same tasks
2. **OCR engines (Surya, DocTR)** — not yet tested, may outperform vision model on pure text extraction
3. **Optimal DPI** — only tested 200 DPI extensively, 150 may suffice for tiled approach
4. **Schedule/table extraction** — worst performance area, may need specialized prompts or table-aware OCR
5. **Overlap deduplication** — tiled extraction produces duplicates in overlap regions, needs merge logic

---

## 6. Environment Notes

- **Python 3.14** — works fine with PyMuPDF, openai, torch, colpali-engine
- **transformers 5.2.0** required for colpali-engine 0.3.14 compatibility
- **MPS (Apple Metal)** crashes with ColQwen2 — CPU-only for now
- **LMStudio** at `192.168.215.90:1234` — serves vision models via OpenAI-compatible API
