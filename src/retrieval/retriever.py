import json
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import DEFAULT_TOP_K


def retrieve_relevant_documents(
    query: str,
    vector_store: Chroma,
    top_k: int = DEFAULT_TOP_K
) -> List[Document]:
    """
    Query the ChromaDB vector store for the most relevant documents.
    
    The search matches against the AI-enhanced summaries (page_content),
    while preserving the raw ground-truth content in metadata.
    """
    print(f"🔍 Searching vector store for query: '{query}' (top_k={top_k})...")
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    documents = retriever.invoke(query)
    print(f"✅ Retrieved {len(documents)} relevant document chunks.")
    return documents


def unpack_retrieved_content(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Unpack the original ground-truth data from retrieved documents:
    - raw_text: Original text paragraphs
    - tables_html: Structured HTML tables
    - images_base64: Raw base64 image strings
    """
    unpacked_records = []

    for i, doc in enumerate(documents):
        record = {
            "chunk_index": i + 1,
            "chunk_id": doc.metadata.get("chunk_id", i + 1),
            "search_summary": doc.page_content,
            "raw_text": "",
            "tables_html": [],
            "images_base64": []
        }

        raw_metadata = doc.metadata.get("original_content", "{}")
        try:
            original_data = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            record["raw_text"] = original_data.get("raw_text", "")
            record["tables_html"] = original_data.get("tables_html", [])
            record["images_base64"] = original_data.get("images_base64", [])
        except Exception as e:
            print(f"⚠️ Warning: Could not unpack original_content for chunk {i+1}: {e}")
            record["raw_text"] = doc.page_content

        unpacked_records.append(record)

    return unpacked_records
