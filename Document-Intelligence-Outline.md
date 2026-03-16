Document Intelligence System for Technical Drawings
Project Overview
Build a multi-layer document intelligence system that can ingest construction drawing sets (fire alarm, electrical, mechanical, plumbing — typically 10-100 pages), extract structured data, build a knowledge graph, and answer natural language questions about device connectivity, code compliance, and quality control.
The system processes drawings that may or may not have embedded OCR text. It must handle large-format (D-size, 24x36") sheets with small annotations, symbol legends, device labels, conduit paths, and cross-sheet references.
Architecture
PDF Upload
    │
    ▼
┌─────────────────────────────────┐
│  Layer 1: Page Processing       │
│  pdf2image → tile slicing →     │
│  Surya OCR + layout analysis    │
│  → structured JSON per page     │
└──────────────┬──────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────────┐  ┌───────────────────┐
│ Layer 2:     │  │ Layer 3:          │
│ Visual Index │  │ Graph Construction│
│ ColFlor →    │  │ Devices, rooms,   │
│ Qdrant       │  │ connections →     │
│              │  │ PostgreSQL / Neo4j│
└──────┬───────┘  └────────┬──────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
┌─────────────────────────────────┐
│  Layer 4: Query & Reasoning     │
│  Route question → retrieve      │
│  pages → graph context →        │
│  vision model → answer          │
└─────────────────────────────────┘
Environment Variables
bash# === LM Studio / Local Inference ===
# LM Studio or vLLM base URL for vision model inference
LMSTUDIO_HOST=http://192.168.1.x:1234

# === Model Selection ===
# Vision model served via LM Studio/vLLM (for structured extraction & reading)
VISION_MODEL=mistralai/Magistral-Small-2506

# Embedding model for text chunks (served via vLLM or local)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

# Visual retrieval model for page-level image embeddings (ColFlor or ColSmol)
VISUAL_RETRIEVAL_MODEL=ahmed-masry/ColFlor

# === External API (optional, for reasoning on non-DoD work) ===
ANTHROPIC_API_KEY=sk-ant-...

# === OCR Configuration ===
# Surya OCR device (cuda, cpu, mps)
SURYA_DEVICE=cuda
# Surya batch size — each batch item ~50MB VRAM. Default 256 = ~12.8GB.
# Reduce for constrained environments.
SURYA_BATCH_SIZE=64

# === Vector Database ===
QDRANT_HOST=http://localhost:6333
QDRANT_COLLECTION=drawing_pages

# === PostgreSQL (structured data + graph) ===
DATABASE_URL=postgresql://user:pass@localhost:5432/docint

# === Processing ===
# Tile grid for large drawings (e.g., 4x4 with 20% overlap)
TILE_GRID_COLS=4
TILE_GRID_ROWS=4
TILE_OVERLAP_PCT=0.20

# Max image dimension for OCR passes (pixels)
MAX_OCR_RESOLUTION=2048

# === Output ===
LOG_LEVEL=info
Tech Stack
ComponentTechnologyNotesLanguagePython 3.11+Main pipelinePDF to Imagepdf2image + Popplerpdftoppm under the hoodOCR PrimarySurya (surya-ocr)Layout + OCR + table recognition, 90+ langsOCR FallbackDocTR (python-doctr[torch])Apache 2.0, rotated text support, hook mechanismVisual RetrievalColFlor (174M) via colpali-enginePage-level image embeddings, Canadian originText EmbeddingsQwen3-Embedding-8B via vLLMFor text chunk retrievalVector DBQdrant (Docker)Multi-vector + MaxSim for late interactionRelational DBPostgreSQL + pgvectorStructured metadata, text embeddings, graph edgesVision/ReaderLM Studio or vLLM (OpenAI-compat API)Magistral-Small or similarReasoningClaude API (optional)For complex multi-step reasoningContainer MgmtDocker / PortainerAll services containerized
Phase 1 — Single-Page Structured Extractor
This is where we start. Build a Python script that takes a single drawing page image and produces structured JSON.
Input
A rasterized drawing page (PNG/JPEG), plus optionally a legend/keynotes image cropped from the same sheet or a separate legend sheet.
Output Schema
json{
  "sheet_id": "FA112",
  "sheet_title": "Fire Alarm & Mass Notification - 1st Floor - Area A",
  "scale": "1/8\" = 1'-0\"",

  "rooms": [
    {
      "id": "room_102",
      "number": "102",
      "name": "Lobby",
      "bbox": [x1, y1, x2, y2]
    }
  ],

  "devices": [
    {
      "id": "dev_001",
      "symbol": "F",
      "type": "Fire Alarm Pull Station",
      "room_id": "room_102",
      "wattage": null,
      "ceiling_mounted": false,
      "bbox": [x1, y1, x2, y2],
      "annotations": ["0.25W"]
    }
  ],

  "connections": [
    {
      "from_device": "dev_001",
      "to_device": "dev_005",
      "conduit_id": null,
      "path_description": "via Main Corridor 105"
    }
  ],

  "text_blocks": {
    "general_notes": ["Provide a supervised shut-off valve..."],
    "keynotes": [
      {"symbol": "#", "text": "Egress lighting control..."},
      {"symbol": "↑", "text": "Elevator shunt"}
    ],
    "legend_entries": [
      {"symbol": "FACU", "description": "Main Fire Alarm Control Unit..."},
      {"symbol": "S", "description": "Smoke Detector - Photoelectric"}
    ]
  },

  "cross_references": [
    {"text": "SEE SHEET E-501", "context": "conduit penetrations through RF shield wall"}
  ]
}
Extraction Strategy

First pass — Legend & Notes: Crop or identify the legend, keynotes, and general notes regions. Extract these first — they are the "decoder ring" for everything else on the sheet. Use Surya for OCR with bounding boxes.
Second pass — Full page vision: Send the entire page image to the vision model (via LMSTUDIO_HOST) with the extracted legend/notes as context in the prompt. Ask for a structured room and device inventory.
Third pass — Tile-based detail: Slice the drawing area (excluding legend/notes) into overlapping tiles per TILE_GRID_* settings. Run Surya OCR on each tile to catch small annotations the vision model missed. Merge tile results using spatial coordinates, deduplicating overlapping detections.
Reconciliation: Cross-reference vision model output with OCR output. Flag discrepancies. Validate all devices against the legend. Flag any symbol that appears in the drawing but not in the legend.

Key Implementation Details

Use OpenAI-compatible API format to talk to LM Studio / vLLM:

python  import openai
  client = openai.OpenAI(base_url=os.environ["LMSTUDIO_HOST"] + "/v1", api_key="not-needed")

For Surya, use the Python API directly:

python  from surya.ocr import run_ocr
  from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
  from surya.model.recognition.model import load_model as load_rec_model
  from surya.model.recognition.processor import load_processor as load_rec_processor

For ColFlor visual embeddings:

python  from colpali_engine.models import ColFlor, ColFlorProcessor

All vision model prompts should request JSON output with a clearly defined schema. Include the legend text in the system prompt so the model can decode symbols.
Tile coordinates must be tracked so bounding boxes can be mapped back to full-page coordinates.

File Structure
docint/
├── .env                          # Environment variables
├── pyproject.toml                # Dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                 # Load env vars, validate config
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingest.py             # PDF → page images
│   │   ├── tiler.py              # Slice pages into overlapping tiles
│   │   ├── ocr.py                # Surya + DocTR OCR wrapper
│   │   ├── vision.py             # Vision model structured extraction
│   │   ├── reconcile.py          # Merge OCR + vision outputs, flag discrepancies
│   │   └── page_processor.py     # Orchestrates passes 1-4 for a single page
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── visual_index.py       # ColFlor embedding + Qdrant storage
│   │   ├── text_index.py         # Text chunk embedding + pgvector
│   │   └── search.py             # Unified search across both indexes
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py            # JSON → graph edges (devices, rooms, connections)
│   │   ├── queries.py            # Connectivity queries, path finding
│   │   └── qc.py                 # Quality control checks (orphaned devices, legend mismatches)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic models for all data structures
│   └── api/
│       ├── __init__.py
│       └── query.py              # Question → route → retrieve → reason → answer
├── scripts/
│   ├── process_drawing.py        # CLI: process a single PDF
│   ├── query.py                  # CLI: ask a question about a processed drawing set
│   └── benchmark.py              # Test extraction accuracy against known drawings
├── tests/
│   ├── test_tiler.py
│   ├── test_ocr.py
│   ├── test_vision.py
│   └── test_reconcile.py
└── docker/
    ├── docker-compose.yml        # Qdrant + PostgreSQL + optional services
    └── Dockerfile                # Main pipeline container
Docker Compose (Qdrant + PostgreSQL)
yamlversion: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: docint
      POSTGRES_USER: docint
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  qdrant_data:
  pg_data:
Quality Control Checks to Implement
These are the high-value automated checks the system should perform after ingesting a drawing set:

Legend consistency: Every symbol used in a drawing should appear in the legend. Every legend entry should appear at least once in the drawings. Flag mismatches.
Cross-reference validation: Extract all "SEE SHEET X-NNN" references. Verify the referenced sheet exists in the uploaded set. Flag orphaned references.
Device connectivity: Using the graph, verify all notification devices have a path back to the FACU/control panel. Flag orphaned devices.
Annotation completeness: Verify devices have required annotations (wattage taps, mounting type C/WP, etc.) per the legend definitions.
Room number consistency: Cross-check room numbers between architectural drawings and discipline-specific drawings (fire alarm, electrical, etc.) if multiple disciplines are uploaded.


Getting Started
bash# 1. Clone and setup
git clone <repo>
cd docint
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Start infrastructure
docker compose -f docker/docker-compose.yml up -d

# 3. Copy .env.example to .env and fill in your values
cp .env.example .env

# 4. Process a test drawing
python scripts/process_drawing.py --input /path/to/FA112.pdf --output ./output/

# 5. Ask a question
python scripts/query.py --drawing-set ./output/ --question "What devices are in Lobby 102?"
Development Notes

Start with Phase 1 only. Get page_processor.py working reliably on 5-10 different sheet types before building Phases 2-5.
Use the uploaded FA112 fire alarm sheet as the primary test case.
The vision model prompt engineering is the highest-leverage work. Iterate on prompts before optimizing code.
Keep the vision model interface abstract (OpenAI-compatible) so we can swap between LM Studio, vLLM, and Claude API without code changes.
All Pydantic models go in schemas.py — this is the contract between pipeline stages.