# Advanced Multimodal RAG System

An advanced, structure-aware, and multimodal Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex research documents containing **multi-column text**, **tables**, and **diagrams/images**.

---

## Architecture Overview

Standard Naive RAG fails on documents like research papers because it flattens complex PDF layouts, garbles tables, and discards visual diagrams. 

This system solves those challenges through:

1. **Computer Vision Layout Analysis**: Uses `unstructured` with `strategy="hi_res"` and Table Transformer models to preserve multi-column reading orders, extract tables as structured HTML, and crop diagrams into base64 image strings.
2. **Structure-Aware Title Chunking**: Chunks text by document sections and headings (`chunk_by_title`), eliminating context fragmentation.
3. **Decoupled Search vs. Generation (Multi-Vector Retrieval)**:
   - **For Retrieval (`page_content`)**: Gemini Flash generates dense, searchable summaries packed with key facts, concepts, reverse questions (HyDE), and visual diagram explanations.
   - **For Synthesis (`metadata['original_content']`)**: Exact raw text, structured HTML tables, and base64 diagram payloads are passed directly to the Vision LLM at query time.
4. **Intermediate Checkpointing & Ingestion Caching**: Parsed elements and intermediate chunks are cached locally (`data/cache/`) to avoid re-running expensive vision models.

```
                      [ Complex PDF ]
                             │
                             ▼ (Layout Vision Models + Table Transformer)
                 [ Layout-Aware Elements ]
                             │
                             ▼ (chunk_by_title)
                    [ Smart Chunks ]
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
    [ AI Search Summary ]       [ Original Ground Truth ]
    - Numbers & facts           - Raw text
    - Reverse questions         - Reconstructed HTML tables
    - Diagram descriptions      - Base64 cropped images
             │                               │
             ▼                               │
    [ ChromaDB Vector Store ]                │
    (Cosine HNSW Space)                      │
             │ (Top-k Retrieval)             │
             └───────────────┬───────────────┘
                             ▼
                 [ Multimodal Prompt ]
                (Text + HTML Tables + Images)
                             │
                             ▼
                    [ Gemini Vision LLM ]
                             │
                             ▼
                   [ Grounded Answer ]
```

---

## Project Structure

```text
Advanced-RAG/
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration (models, thresholds, paths)
│
├── src/
│   ├── ingestion/           # Layout analysis & structure-aware chunking
│   │   ├── parser.py        # PDF partitioning with SHA-256 caching
│   │   └── chunker.py       # Title-aware chunking & modality separator
│   │
│   ├── indexing/            # Multimodal summarization & vector database
│   │   ├── summarizer.py    # AI summary engine & JSON exporter
│   │   └── vector_store.py  # ChromaDB manager (HNSW cosine space)
│   │
│   ├── retrieval/           # Context retrieval & metadata unpacking
│   │   └── retriever.py     # Vector search & ground-truth extractor
│   │
│   └── generation/          # Grounded answer synthesis
│       └── generator.py     # Multimodal prompt builder & Vision LLM caller
│
├── data/
│   ├── docs/                # Input PDF documents
│   ├── cache/               # Intermediate checkpoints & caches
│   └── chroma_db/           # Persistent ChromaDB vector store
│
├── scripts/
│   ├── ingest.py            # CLI: Ingest and index a PDF
│   └── query.py             # CLI: Query the indexed database
│
├── .env.example             # API keys template
├── requirements.txt         # Project dependencies
└── README.md                # System documentation
```

---

## Getting Started with Docker (Recommended)

Using Docker completely eliminates the need to install Windows binaries (Poppler, Tesseract-OCR, libmagic) manually.

### 1. Configure `.env`
Ensure your `.env` contains your API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Build the Docker Image
```bash
docker compose build
```

### 3. Launch the Web Chat Interface
You can launch the full interactive web application in your browser:
```bash
docker compose up
```
Open **`http://localhost:8501`** in your browser!
- Upload new PDFs and trigger ingestion with a click.
- Ask questions and view answers.
- Inspect the **reconstructed HTML tables** and **extracted diagrams** used by the Vision LLM for grounding.

### 4. CLI Ingestion (Alternative)
Place your PDF into `data/docs/` (e.g. `data/docs/attention-is-all-you-need.pdf`), then run:
```bash
docker compose run --rm rag python scripts/ingest.py --file data/docs/attention-is-all-you-need.pdf
```
*(All data, caches, and ChromaDB vector stores persist automatically in your local `data/` folder on your host machine!)*

### 5. CLI Query (Alternative)
```bash
docker compose run --rm rag python scripts/query.py --query "How many attention heads does the Transformer use?" --sources
```

---

## Local Setup (Alternative)

If running directly on host without Docker:

### 3. Configure API Keys
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
In `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## Usage

### Step 1: Place Your PDF
Put your PDF file in `data/docs/` (e.g., `data/docs/attention-is-all-you-need.pdf`).

### Step 2: Ingest the Document
Run the ingestion pipeline:
```bash
python scripts/ingest.py --file data/docs/attention-is-all-you-need.pdf
```
*Note: To force a clean re-partition without using cached elements, pass the `--force` flag.*

### Step 3: Query the System
Ask questions against your indexed documents:
```bash
python scripts/query.py --query "How many attention heads does the Transformer use, and what is the dimension of each head?"
```

To see the retrieved chunks, tables, and images used for the answer, add `--sources`:
```bash
python scripts/query.py --query "What is the BLEU score on English-to-German translation?" --sources
```
