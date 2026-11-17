import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion import partition_document, create_chunks_by_title
from src.indexing import summarise_chunks, build_vector_store, export_chunks_to_json
from config.settings import DOCS_DIR, CHROMA_DIR


def run_ingestion(pdf_path: str | Path, force_reload: bool = False):
    """Run the complete end-to-end ingestion and indexing pipeline."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: File not found at {pdf_file}")
        sys.exit(1)

    print("=" * 60)
    print(f"🚀 INGESTION PIPELINE: {pdf_file.name}")
    print("=" * 60)

    # Step 1: Partition with layout detection & caching
    elements = partition_document(pdf_file, force_reload=force_reload)

    # Step 2: Title-aware chunking
    chunks = create_chunks_by_title(elements)

    # Step 3: Multimodal AI summarization (decoupled search representations)
    documents = summarise_chunks(chunks)

    # Step 4: Export JSON checkpoint
    export_chunks_to_json(documents)

    # Step 5: Embed and build persistent Chroma vector store
    vector_store = build_vector_store(documents, persist_directory=CHROMA_DIR)

    print("=" * 60)
    print("🎉 Pipeline completed! Vector store is ready for querying.")
    print("=" * 60)
    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and index a PDF for Advanced Multimodal RAG")
    parser.add_argument(
        "--file",
        type=str,
        default=str(DOCS_DIR / "attention-is-all-you-need.pdf"),
        help="Path to the PDF file to ingest"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-parsing without using cached elements"
    )

    args = parser.parse_args()
    run_ingestion(args.file, force_reload=args.force)
