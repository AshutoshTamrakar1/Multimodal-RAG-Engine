from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    CHROMA_DIR,
    HNSW_SPACE
)


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:
    """Initialize the Google Generative AI embeddings model."""
    return GoogleGenerativeAIEmbeddings(model=model_name)


def build_vector_store(
    documents: List[Document],
    persist_directory: Optional[Path] = CHROMA_DIR,
    collection_name: str = "multimodal_rag",
    embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None
) -> Chroma:
    """
    Embed documents and create a persistent ChromaDB vector store.
    
    Uses Cosine distance space ('hnsw:space': 'cosine') for optimal semantic ranking.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    persist_path = Path(persist_directory) if persist_directory else CHROMA_DIR
    persist_path.mkdir(parents=True, exist_ok=True)

    print(f"🔮 Generating embeddings with {embedding_model.model} and indexing {len(documents)} documents...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(persist_path),
        collection_name=collection_name,
        collection_metadata={"hnsw:space": HNSW_SPACE}
    )
    print(f"✅ Vector store successfully built and saved to: {persist_path}")
    return vectorstore


def load_vector_store(
    persist_directory: Optional[Path] = CHROMA_DIR,
    collection_name: str = "multimodal_rag",
    embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None
) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk without re-embedding documents.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    persist_path = Path(persist_directory) if persist_directory else CHROMA_DIR
    if not persist_path.exists():
        raise FileNotFoundError(f"No ChromaDB directory found at: {persist_path}")

    print(f"📂 Loading existing vector store from: {persist_path}")
    vectorstore = Chroma(
        persist_directory=str(persist_path),
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    print("✅ Vector store loaded successfully.")
    return vectorstore
