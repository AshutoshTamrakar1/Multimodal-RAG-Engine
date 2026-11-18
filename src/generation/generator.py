from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.indexing.summarizer import get_default_llm
from src.retrieval.retriever import unpack_retrieved_content


def generate_final_answer(
    query: str,
    documents: List[Document],
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> str:
    """
    Synthesize a final, grounded answer using multimodal ground-truth context.
    
    Feeds:
    - Original raw text
    - Structured HTML tables
    - Base64 image payloads (as vision parts)
    directly to a Vision-capable LLM.
    """
    if llm is None:
        llm = get_default_llm()

    try:
        unpacked_chunks = unpack_retrieved_content(documents)

        # Build prompt using exact ground-truth content (not summaries!)
        prompt_text = f"""Based on the following retrieved documents, please answer this question:
Question: {query}

CONTENT TO ANALYZE:
"""
        total_images = 0
        total_tables = 0

        for chunk in unpacked_chunks:
            idx = chunk["chunk_index"]
            prompt_text += f"\n--- Document {idx} ---\n"

            if chunk["raw_text"]:
                prompt_text += f"TEXT:\n{chunk['raw_text']}\n\n"

            if chunk["tables_html"]:
                prompt_text += "TABLES:\n"
                for t_idx, table in enumerate(chunk["tables_html"]):
                    total_tables += 1
                    prompt_text += f"Table {t_idx + 1}:\n{table}\n\n"

            total_images += len(chunk["images_base64"])

        prompt_text += """
INSTRUCTIONS:
1. Provide a comprehensive, accurate answer synthesized directly from the text, tables, and images provided above.
2. Structure your answer using clean, well-formatted Markdown:
   - Use clear sections with bold titles or subheadings.
   - Use bullet points for lists, key steps, or components.
   - Format numeric data, metrics, or comparisons into clean Markdown tables.
   - Use LaTeX syntax for mathematical formulas (e.g., $d_k$, $\text{softmax}(...)$).
3. Explicitly reference diagrams (e.g., "As seen in Figure 1...") and tables when citing facts.
4. If the retrieved documents do not contain enough information, clearly state:
   "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Construct multimodal message starting with the formatted prompt
        message_content = [{"type": "text", "text": prompt_text}]

        # Append all extracted base64 images from retrieved chunks as vision items
        for chunk in unpacked_chunks:
            for image_base64 in chunk["images_base64"]:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })

        print(f"🤖 Invoking Vision LLM with {len(unpacked_chunks)} chunks "
              f"({total_tables} tables, {total_images} images)...")

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        # Cleanly unpack text content (handles Gemini list-of-blocks format)
        response_content = response.content
        if isinstance(response_content, str):
            return response_content.strip()
        if isinstance(response_content, list):
            text_blocks = []
            for block in response_content:
                if isinstance(block, str):
                    text_blocks.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_blocks.append(block["text"])
            if text_blocks:
                return "\n\n".join(text_blocks).strip()
                
        return str(response_content).strip()

    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return f"Error generating answer: {str(e)}"
