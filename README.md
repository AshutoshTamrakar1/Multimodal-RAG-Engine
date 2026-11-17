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

## Getting Started

### 1. Prerequisites (System Dependencies)
The layout analysis engine requires system-level tools:
- **Poppler** (PDF rendering):
  - *Windows*: Download Poppler for Windows and add `bin/` to your PATH.
  - *Ubuntu/Debian*: `sudo apt-get install -y poppler-utils`
  - *macOS*: `brew install poppler`
- **Tesseract OCR** (Image text extraction):
  - *Windows*: Install via installer and add to PATH.
  - *Ubuntu/Debian*: `sudo apt-get install -y tesseract-ocr`
  - *macOS*: `brew install tesseract`

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/AshutoshRanjan31/Advanced-RAG.git
cd Advanced-RAG

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

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
