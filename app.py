import io
import base64
from pathlib import Path

import streamlit as st
from PIL import Image

from config.settings import DOCS_DIR, CHROMA_DIR, DEFAULT_TOP_K
from scripts.ingest import run_ingestion
from src.indexing import load_vector_store
from src.retrieval import retrieve_relevant_documents, unpack_retrieved_content
from src.generation import generate_final_answer

# Page Configuration
st.set_page_config(
    page_title="Advanced Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern, polished styling
st.markdown("""
<style>
    /* Global layout tweaks */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }
    
    /* Header typography */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #555e69;
        margin-bottom: 1.4rem;
        line-height: 1.4;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.35rem;
    }
    .badge-text { background-color: #f1f3f5; color: #495057; border: 1px solid #ced4da; }
    .badge-table { background-color: #e7f5ff; color: #1971c2; border: 1px solid #a5d8ff; }
    .badge-image { background-color: #ebfbee; color: #2f9e44; border: 1px solid #b2f2bb; }
    
    /* Evidence card */
    .evidence-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Table styles in evidence */
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }
    th, td {
        border: 1px solid #dee2e6;
        padding: 6px 10px;
        text-align: left;
    }
    th {
        background-color: #f1f3f5;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_vector_store(CHROMA_DIR)
    except Exception:
        st.session_state.vector_store = None


def decode_base64_image(b64_string: str) -> Image.Image:
    """Safely decode base64 string to a PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


# --- SIDEBAR: Document Management & Parameters ---
with st.sidebar:
    st.title("📂 Document Hub")
    st.caption("Upload PDFs with multi-column text, tables & diagrams.")

    # 1. File Upload
    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        save_path = DOCS_DIR / uploaded_file.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")

        force_reload = st.checkbox("Force re-process (bypass cache)", value=False)
        
        if st.button("⚡ Ingest & Index Document", use_container_width=True, type="primary"):
            with st.spinner(f"Parsing layout & indexing {uploaded_file.name}..."):
                try:
                    vs = run_ingestion(save_path, force_reload=force_reload)
                    st.session_state.vector_store = vs
                    st.session_state.messages = []  # Reset chat on new ingest
                    st.success("✅ Ingestion complete! Ready to chat.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # 2. Existing Documents
    existing_pdfs = list(DOCS_DIR.glob("*.pdf"))
    if existing_pdfs:
        st.markdown("**📁 Indexed Documents:**")
        for pdf in existing_pdfs:
            st.markdown(f"• `{pdf.name}`")

    st.divider()

    # 3. Settings
    st.subheader("⚙️ Search Controls")
    top_k = st.slider("Top Chunks Retrieved (k)", min_value=1, max_value=8, value=DEFAULT_TOP_K)
    show_sources = st.toggle("Show Ground-Truth Evidence", value=True)

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Status Pill
    st.divider()
    if st.session_state.vector_store is not None:
        st.success("🟢 Vector Database Connected", icon="✅")
    else:
        st.warning("🟡 No Vector Database loaded", icon="⚠️")


# --- MAIN HEADER ---
st.markdown('<div class="hero-title">🧠 Advanced Multimodal RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Structure-Aware Layout Parsing • Multimodal Summarization • Ground-Truth Synthesis'
    '</div>',
    unsafe_allow_html=True
)

# Helper function to render evidence block
def render_evidence_block(evidence_list):
    if not evidence_list:
        return
        
    with st.expander("🔍 View Retrieved Evidence (Tables, Diagrams & Raw Text)", expanded=False):
        for chunk in evidence_list:
            c_id = chunk.get("chunk_id", "N/A")
            
            # Badge header
            badges = ['<span class="badge badge-text">Text</span>']
            if chunk.get("tables_html"):
                badges.append(f'<span class="badge badge-table">{len(chunk["tables_html"])} Table(s)</span>')
            if chunk.get("images_base64"):
                badges.append(f'<span class="badge badge-image">{len(chunk["images_base64"])} Image(s)</span>')
            
            st.markdown(f"**Chunk {c_id}** " + " ".join(badges), unsafe_allow_html=True)
            
            # Tabbed view for clean inspection
            tabs = st.tabs(["📄 Raw Text", "📊 Tables", "🖼️ Diagrams", "💡 AI Search Summary"])
            
            # Tab 1: Text
            with tabs[0]:
                raw_text = chunk.get("raw_text", "").strip()
                if raw_text:
                    st.markdown(raw_text)
                else:
                    st.caption("No standalone narrative text in this chunk.")
            
            # Tab 2: Tables
            with tabs[1]:
                tables = chunk.get("tables_html", [])
                if tables:
                    for t_i, tbl in enumerate(tables):
                        st.markdown(f"**Table {t_i+1}**")
                        st.markdown(tbl, unsafe_allow_html=True)
                else:
                    st.caption("No tables in this chunk.")
            
            # Tab 3: Images
            with tabs[2]:
                images = chunk.get("images_base64", [])
                if images:
                    for img_i, img_b64 in enumerate(images):
                        try:
                            img = decode_base64_image(img_b64)
                            st.image(img, caption=f"Extracted Diagram {img_i+1}", use_container_width=True)
                        except Exception as e:
                            st.caption(f"Error rendering diagram {img_i+1}: {e}")
                else:
                    st.caption("No images or diagrams in this chunk.")
            
            # Tab 4: Search Summary
            with tabs[3]:
                summary = chunk.get("search_summary", "").strip()
                if summary:
                    st.info(summary)
                else:
                    st.caption("No summary recorded.")
            
            st.divider()


# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("evidence") and show_sources:
            render_evidence_block(msg["evidence"])

# Welcome prompt when conversation is fresh
if not st.session_state.messages:
    if st.session_state.vector_store is not None:
        st.info(
            "👋 **Ready to assist!** Ask questions about your document.\n\n"
            "💡 *Examples you can try:*\n"
            "- *'What are the two main components of the Transformer architecture?'*\n"
            "- *'Compare the BLEU scores and training costs in Table 2.'*\n"
            "- *'Explain the flow of Multi-Head Attention based on Figure 1.'*"
        )
    else:
        st.info("👈 Upload and Ingest a PDF document in the sidebar to get started!")

# --- CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask a question about the document..."):
    if st.session_state.vector_store is None:
        st.error("⚠️ Please ingest or load a document first via the sidebar!")
    else:
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 2. Assistant Response
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching document & synthesizing grounded answer..."):
                # Retrieve chunks
                retrieved_docs = retrieve_relevant_documents(
                    query=prompt,
                    vector_store=st.session_state.vector_store,
                    top_k=top_k
                )
                unpacked_evidence = unpack_retrieved_content(retrieved_docs)
                
                # Generate answer with multimodal ground truth
                answer = generate_final_answer(prompt, retrieved_docs)

                # Render answer cleanly
                st.markdown(answer)

                # Render evidence block
                if show_sources and unpacked_evidence:
                    render_evidence_block(unpacked_evidence)

        # 3. Save assistant message to state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "evidence": unpacked_evidence
        })
