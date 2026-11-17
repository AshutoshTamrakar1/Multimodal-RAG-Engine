from typing import List, Dict, Any
from unstructured.chunking.title import chunk_by_title

from config.settings import (
    MAX_CHARACTERS,
    NEW_AFTER_N_CHARS,
    COMBINE_TEXT_UNDER_N_CHARS
)


def create_chunks_by_title(
    elements: List[Any],
    max_characters: int = MAX_CHARACTERS,
    new_after_n_chars: int = NEW_AFTER_N_CHARS,
    combine_text_under_n_chars: int = COMBINE_TEXT_UNDER_N_CHARS
) -> List[Any]:
    """
    Create hierarchical, structure-aware chunks respecting Title elements.
    
    Prevents context severing by binding sections to their titles and merges
    tiny micro-chunks to retain semantic density.
    """
    print(f"🔨 Creating smart chunks (max_chars={max_characters}, new_after={new_after_n_chars})...")
    chunks = chunk_by_title(
        elements=elements,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars
    )
    print(f"✅ Created {len(chunks)} smart chunks.")
    return chunks


def separate_content_types(chunk: Any) -> Dict[str, Any]:
    """
    Inspect a CompositeElement chunk and separate its constituent modalities:
    - Raw text
    - Structured HTML tables (with fallback to plain text)
    - Base64-encoded images/diagrams
    - Unique modality tags (e.g., ['text', 'table', 'image'])
    """
    content_data = {
        "text": getattr(chunk, "text", str(chunk)),
        "tables": [],
        "images": [],
        "types": ["text"]
    }

    # Inspect the low-level elements merged into this composite chunk
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Reconstructed table handling
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            # Base64 image payload handling
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    # Deduplicate modality flags
    content_data["types"] = list(set(content_data["types"]))
    return content_data
