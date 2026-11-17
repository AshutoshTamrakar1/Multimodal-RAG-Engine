import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.indexing import load_vector_store
from src.retrieval import retrieve_relevant_documents, unpack_retrieved_content
from src.generation import generate_final_answer
from config.settings import DEFAULT_TOP_K, CHROMA_DIR


def run_query(query: str, top_k: int = DEFAULT_TOP_K, show_sources: bool = False):
    """Query the existing indexed database and generate an answer."""
    print("=" * 60)
    print(f"❓ QUERY: {query}")
    print("=" * 60)

    # Step 1: Load pre-indexed vector store
    try:
        vector_store = load_vector_store(persist_directory=CHROMA_DIR)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Hint: Run `python scripts/ingest.py --file <path-to-pdf>` first!")
        sys.exit(1)

    # Step 2: Retrieve top-k documents (matched against AI summaries)
    retrieved_docs = retrieve_relevant_documents(query, vector_store, top_k=top_k)

    # Step 3: Generate multimodal ground-truth answer
    answer = generate_final_answer(query, retrieved_docs)

    print("\n" + "=" * 60)
    print("💡 FINAL ANSWER:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    # Optional: Display source chunks inspection
    if show_sources:
        print("\n📚 RETRIEVED SOURCES:")
        unpacked = unpack_retrieved_content(retrieved_docs)
        for chunk in unpacked:
            print(f"\n[Chunk {chunk['chunk_id']}]")
            print(f"- Search Summary Preview: {chunk['search_summary'][:150]}...")
            print(f"- Tables Found: {len(chunk['tables_html'])}")
            print(f"- Images Found: {len(chunk['images_base64'])}")
            if chunk['raw_text']:
                print(f"- Raw Text Preview: {chunk['raw_text'][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Advanced Multimodal RAG system")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Question to ask the system"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve (default: 3)"
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Show details of retrieved chunks"
    )

    args = parser.parse_args()
    run_query(args.query, top_k=args.top_k, show_sources=args.sources)
