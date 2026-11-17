import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import (
    DEFAULT_LLM_MODEL,
    LLM_TEMPERATURE,
    CACHE_DIR
)
from src.ingestion.chunker import separate_content_types


def get_default_llm() -> ChatGoogleGenerativeAI:
    """Initialize default vision-capable Gemini LLM for indexing and summarization."""
    return ChatGoogleGenerativeAI(
        model=DEFAULT_LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )


def create_ai_enhanced_summary(
    text: str,
    tables: List[str],
    images: List[str],
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> str:
    """
    Generate a searchable multimodal summary optimized for vector retrieval.
    
    Includes:
    - Key data points, numbers, and facts
    - Concepts and terminology
    - Reverse question generation (HyDE in reverse)
    - Visual chart and diagram explanations
    - Alternative user search queries
    """
    if llm is None:
        llm = get_default_llm()

    try:
        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:
1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""

        # Construct multimodal payload if images are present
        if images:
            message_content = [{"type": "text", "text": prompt_text}]
            for image_base64 in images:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
            message = HumanMessage(content=message_content)
        else:
            message = HumanMessage(content=prompt_text)

        response = llm.invoke([message])
        response_content = response.content

        # Handle Gemini response format variations (string vs block list)
        if isinstance(response_content, str):
            return response_content
        if isinstance(response_content, list):
            text_blocks = []
            for block in response_content:
                if isinstance(block, str):
                    text_blocks.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_blocks.append(block["text"])
            if text_blocks:
                return "\n".join(text_blocks)

        return str(response_content)

    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Graceful fallback: return text snippet with modality annotations
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(
    chunks: List[Any],
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> List[Document]:
    """
    Process chunks into LangChain Document objects with decoupled retrieval metadata.
    
    - page_content: AI-enhanced searchable summary (what is embedded)
    - metadata['original_content']: Exact raw text, HTML tables, and base64 images (what is passed to generator)
    """
    print("🧠 Processing chunks with AI Summaries...")
    if llm is None:
        llm = get_default_llm()

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        content_data = separate_content_types(chunk)

        has_mixed_content = bool(content_data["tables"] or content_data["images"])
        print(f"   [{current_chunk}/{total_chunks}] Modalities: {content_data['types']} "
              f"(Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])})")

        # Selective AI enrichment: Only call Vision LLM when non-text modalities exist
        if has_mixed_content:
            print(f"     → Generating multimodal AI summary...")
            enhanced_content = create_ai_enhanced_summary(
                text=content_data["text"],
                tables=content_data["tables"],
                images=content_data["images"],
                llm=llm
            )
        else:
            enhanced_content = content_data["text"]

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "chunk_id": current_chunk,
                "modalities": content_data["types"],
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables"],
                    "images_base64": content_data["images"]
                })
            }
        )
        langchain_documents.append(doc)

    print(f"✅ Processed {len(langchain_documents)} documents ready for indexing.")
    return langchain_documents


def export_chunks_to_json(
    documents: List[Document],
    output_path: Optional[str | Path] = None
) -> List[Dict[str, Any]]:
    """Export processed documents to a human-readable JSON checkpoint."""
    if output_path is None:
        output_path = CACHE_DIR / "chunks_export.json"
    else:
        output_path = Path(output_path)

    export_data = []
    for doc in documents:
        raw_meta = doc.metadata.get("original_content", "{}")
        original_content = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta

        chunk_data = {
            "chunk_id": doc.metadata.get("chunk_id"),
            "modalities": doc.metadata.get("modalities", []),
            "enhanced_search_content": doc.page_content,
            "metadata": {
                "original_content": original_content
            }
        }
        export_data.append(chunk_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Exported {len(export_data)} chunks to: {output_path}")
    return export_data
