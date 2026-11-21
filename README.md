# 🧠 Advanced Multimodal RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange)](https://www.trychroma.com/)
[![Gemini API](https://img.shields.io/badge/Google%20Gemini-Flash--Lite%20%26%20Embeddings-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

An enterprise-grade, structure-aware, and multimodal Retrieval-Augmented Generation (RAG) system engineered to parse, index, and reason over complex documents containing **multi-column text**, **nested tables**, and **technical diagrams/figures**.

---

## 📑 Table of Contents
- [Why Advanced Multimodal RAG?](#-why-advanced-multimodal-rag)
- [System Architecture & Lifecycle](#-system-architecture--lifecycle)
  - [Pipeline A: Ingestion & Indexing (Offline)](#pipeline-a-ingestion--indexing-offline)
  - [Pipeline B: Retrieval & Synthesis (Online)](#pipeline-b-retrieval--synthesis-online)
- [The Core Innovation: Decoupled Multi-Vector Schema](#-the-core-innovation-decoupled-multi-vector-schema)
- [Project Structure](#-project-structure)
- [Getting Started (Docker Quickstart)](#-getting-started-docker-quickstart)
- [Interactive Web UI](#-interactive-web-ui)
- [CLI Tools](#-cli-tools)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## ⚡ Why Advanced Multimodal RAG?

Standard (Naive) RAG pipelines treat PDFs as flat streams of plain characters, which catastrophic failure modes on technical papers, engineering manuals, and financial reports:

| Challenge | Naive RAG Failure | How This System Solves It |
| :--- | :--- | :--- |
| **Multi-Column Text** | Reads straight across horizontal lines, creating scrambled nonsense. | **Computer Vision Layout Analysis (`YOLOX`)** identifies visual column bounding boxes and reads down columns in true sequential order. |
| **Complex Tables** | Row and column associations are flattened into meaningless tokens. | **Table Transformer (TATR)** reconstructs the grid hierarchy into clean, structured **HTML tables** (`<table>...</table>`). |
| **Diagrams & Charts** | Images, circuits, and flowcharts are completely discarded. | Crops figures as **base64 image payloads** and feeds them directly to **Vision LLMs**. |
| **Context Severing** | Splitting every 500 characters splits sentences and separates headers from their explanations. | **Hierarchical Title Chunking (`chunk_by_title`)** keeps sections unified under their governing title and merges micro-chunks. |
| **Search Representation Mismatch** | Embedding models cannot effectively match natural language queries against raw HTML table tags or image pixels. | **Decoupled Search vs. Ground-Truth Generation**: Searches dense AI summaries, but passes uncompressed raw text, HTML tables, and images to the LLM. |

---

## 🏗️ System Architecture & Lifecycle

The system is cleanly separated into two independent lifecycles:

```
================================ PIPELINE A: INGESTION & INDEXING (OFFLINE) ================================

  [ PDF File ] 
       │
       ▼ (Computer Vision Layout + Table Transformer)
  [ parser.py ] ──────────► [ SHA-256 Pickle Cache ] (data/cache/*_elements.pkl)
       │ (264 Layout Elements)
       ▼ (Hierarchical Section Splitting)
  [ chunker.py ] ─────────► [ Modality Dispatcher ] (Text, HTML Tables, Base64 Images)
       │ (Smart Chunks)
       ▼ (Gemini Vision Summaries for Non-Text Chunks)
  [ summarizer.py ] ──────► [ chunks_export.json Checkpoint ]
       │ (LangChain Documents: page_content=Summary | metadata=Ground-Truth)
       ▼ (gemini-embedding-001 • 3,072-dimensional Cosine Metric Space)
  [ vector_store.py ] ────► [ ChromaDB Persistence ] (data/chroma_db/)


================================ PIPELINE B: RETRIEVAL & SYNTHESIS (ONLINE) ================================

  [ User Question ] 
       │
       ▼
  [ retriever.py ] ◄─────── [ ChromaDB ] (Matches question embedding against AI Summaries)
       │ (Top-K Chunks)
       ▼
  [ retriever.py ] ────────► Unpacks metadata['original_content'] (Raw Text + HTML Tables + Images)
       │
       ▼
  [ generator.py ] ────────► Multimodal Prompt Assembly (Inlines HTML Tables + Decoded Base64 Images)
       │
       ▼
  [ Gemini 3.1 Flash Lite ] 
       │
       ▼
  [ Grounded Answer ] ────► Streamlit Web UI / CLI
```

### Pipeline A: Ingestion & Indexing (Offline)
1. **`src/ingestion/parser.py`**: Executes `partition_pdf(strategy="hi_res")`. Bypasses re-computation using an automated SHA-256 file hash cache (`data/cache/`).
2. **`src/ingestion/chunker.py`**:
   - `create_chunks_by_title`: Enforces soft and hard character ceilings (`MAX_CHARACTERS=3000`, `NEW_AFTER_N_CHARS=2400`) while respecting section headings.
   - `separate_content_types`: Categorizes constituents into `{text, tables, images, types}`.
3. **`src/indexing/summarizer.py`**: Uses Gemini to synthesize reverse search questions, keyword expansions, and diagram explanations into `doc.page_content`. Retains ground truth in `metadata['original_content']`.
4. **`src/indexing/vector_store.py`**: Embeds summaries into a persistent **ChromaDB** collection with HNSW Cosine distance.

### Pipeline B: Retrieval & Synthesis (Online)
1. **`src/retrieval/retriever.py`**: Matches the user query against the dense searchable summaries.
2. **`unpack_retrieved_content`**: Deserializes ground-truth text, structured HTML tables, and base64 diagram strings from `metadata`.
3. **`src/generation/generator.py`**: Dynamically constructs a multimodal `HumanMessage` with formatted text, HTML tables, and base64 image data URIs (`data:image/jpeg;base64,...`).
4. **Gemini Vision**: Synthesizes a grounded, verifiable Markdown answer referencing exact tables and figures.

---

## 💡 The Core Innovation: Decoupled Multi-Vector Schema

In Naive RAG:
$$\text{Query} \xrightarrow{\text{cosine similarity}} \text{Embedded Raw Text} \xrightarrow{\text{sent to LLM}} \text{Answer}$$

In this Advanced Architecture:
```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LangChain Document Object                       │
├───────────────────────────────────┬────────────────────────────────────┤
│   doc.page_content (RETRIEVAL)    │  doc.metadata['original_content']  │
│                                   │            (SYNTHESIS)             │
├───────────────────────────────────┼────────────────────────────────────┤
│  Dense AI-Enhanced Summary:       │  Ground-Truth Payload:             │
│  • Key facts & numbers            │  • Exact unsummarized raw text     │
│  • Hypothesized reverse questions │  • Reconstructed HTML tables       │
│  • Search synonyms & concepts     │  • High-res base64 image strings   │
│  • Visual diagram descriptions    │                                    │
└───────────────────────────────────┴────────────────────────────────────┘
```
**Why this matters**:
- You search over what is easiest for vector spaces to index (dense, concept-rich summaries).
- You synthesize over ground truth (uncompressed tables and high-resolution diagrams), preventing summary drift or hallucination.

---

## 📂 Project Structure

```text
Advanced-RAG/
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration (models, thresholds, paths)
│
├── src/
│   ├── __init__.py
│   ├── ingestion/           # STEP 1: Document layout analysis & chunking
│   │   ├── __init__.py
│   │   ├── parser.py        # PDF partitioning (hi_res strategy + SHA-256 caching)
│   │   └── chunker.py       # Title-aware chunking & modality classifier
│   │
│   ├── indexing/            # STEP 2: Multimodal summarization & vector database
│   │   ├── __init__.py
│   │   ├── summarizer.py    # Gemini vision summary engine & JSON exporter
│   │   └── vector_store.py  # ChromaDB manager (HNSW cosine space)
│   │
│   ├── retrieval/           # STEP 3: Context search & metadata unpacking
│   │   ├── __init__.py
│   │   └── retriever.py     # Top-K search & ground-truth content unpacker
│   │
│   └── generation/          # STEP 4: Multimodal answer synthesis
│       ├── __init__.py
│       └── generator.py     # Multimodal prompt builder & Vision LLM caller
│
├── data/
│   ├── docs/                # Input PDF documents
│   ├── cache/               # Intermediate checkpoints & SHA-256 pickle caches
│   └── chroma_db/           # Persistent ChromaDB vector database
│
├── scripts/
│   ├── ingest.py            # CLI: Run end-to-end ingestion on a PDF
│   ├── query.py             # CLI: Query the vector store and inspect sources
│   ├── test_connection.py   # Diagnostic: Verify Gemini LLM & embedding connectivity
│   └── check_models.py      # Diagnostic: Discover accessible models for your API key
│
├── app.py                   # Streamlit Web Chat Interface
├── Dockerfile               # Debian-based container with Poppler, Tesseract & OpenCV
├── docker-compose.yml       # Service orchestration & live-mounted storage
├── .env.example             # Environment variable template
├── requirements.txt         # Pinned Python dependencies
└── README.md                # System documentation
```

---

## 🚀 Getting Started (Docker Quickstart)

Using Docker completely eliminates manual installation of system C-libraries (**Poppler**, **Tesseract-OCR**, and **OpenGL/libGL**).

### 1. Configure Your API Key
Copy `.env.example` to `.env` and insert your Google Gemini API key:
```bash
cp .env.example .env
```
In `.env`:
```env
GOOGLE_API_KEY=AIzaSy...YourKeyHere
```
*(Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)).*

### 2. Build the Docker Image
```bash
docker compose build
```

### 3. Verify Model Connectivity
Ensure your API key and network can reach the Gemini models:
```bash
docker compose run --rm rag python scripts/test_connection.py
```

### 4. Launch the Streamlit Web UI
```bash
docker compose up
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser!

---

## 🖥️ Interactive Web UI

The Streamlit app provides an end-to-end graphical workbench:
- **Sidebar**:
  - **PDF Drag & Drop**: Upload any technical document directly from your browser.
  - **Single-Click Ingestion**: Triggers layout analysis and indexing with real-time status indicators.
  - **Search Controls**: Adjust Top-$K$ retrieval sliders and toggle evidence inspection.
- **Main Chat View**:
  - Full conversational history with assistant streaming and Markdown formatting.
  - **Tabbed Evidence Inspector**: Expand any assistant answer to review the exact ground truth used:
    - `📄 Raw Text`: Narrative text paragraphs.
    - `📊 Tables`: Structured tables rendered with clean HTML borders.
    - `🖼️ Diagrams`: High-resolution figures and schematics extracted from the PDF.
    - `💡 AI Search Summary`: The exact summary indexed in ChromaDB.

---

## 💻 CLI Tools

If you prefer operating from the command line:

### Ingest a Document
```bash
docker compose run --rm rag python scripts/ingest.py --file data/docs/attention-is-all-you-need.pdf
```
*(To bypass the cache and force re-partitioning, append `--force`).*

### Query the Indexed Database
```bash
docker compose run --rm rag python scripts/query.py --query "How many attention heads does the Transformer use, and what is the dimension of each head?" --sources
```

---

## ⚙️ Configuration Reference

All hyperparameters and paths are centralized in [`config/settings.py`](file:///c:/Users/ashut/Personal/GitHub/Advanced-RAG/config/settings.py) and can be overridden via `.env`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `LLM_MODEL` | `gemini-3.1-flash-lite` | Vision-capable LLM used for summarization and final answer synthesis. |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model producing 3,072-dimensional dense vectors. |
| `LLM_TEMPERATURE` | `0.0` | Deterministic decoding to prevent hallucinations in indexing and generation. |
| `MAX_CHARACTERS` | `3000` | Hard character ceiling per chunk (~750 tokens). |
| `NEW_AFTER_N_CHARS` | `2400` | Soft character boundary to split at paragraph ends. |
| `COMBINE_TEXT_UNDER_N_CHARS` | `500` | Merges undersized micro-chunks with neighbors to maintain semantic context. |
| `DEFAULT_TOP_K` | `3` | Number of document chunks retrieved per query. |
| `HNSW_SPACE` | `cosine` | Metric space for ChromaDB vector index. |

---

## 🔧 Troubleshooting & FAQ

### 1. `ImportError: libGL.so.1: cannot open shared object file`
- **Cause**: OpenCV GUI bindings missing in minimal Linux containers.
- **Solution**: Handled automatically in our `Dockerfile` by installing `libgl1`, `libglx-mesa0`, and `libglib2.0-0`.

### 2. `402 RESOURCE_EXHAUSTED / Prepayment credits depleted`
- **Cause**: The API key is attached to a Google Cloud project with an active billing account that has \$0 prepaid balance.
- **Solution**: Go to [Google AI Studio](https://aistudio.google.com/app/apikey), click **"Create API key in new project"** (without attaching billing). This grants the 100% Free Tier (15 requests/min).

### 3. `404 NOT_FOUND: models/... is not found`
- **Cause**: Model name mismatch or legacy model deprecation.
- **Solution**: Run `docker compose run --rm rag python scripts/check_models.py` to list all valid model strings available to your specific key.

### 4. How do I clear the database or re-index?
- Clear vector store: Delete contents inside `data/chroma_db/`.
- Clear cached layouts: Delete `.pkl` files in `data/cache/`.
