import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()
from config.settings import DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

print("=" * 60)
print("🧪 TESTING MODEL CONNECTIVITY")
print("=" * 60)
print(f"LLM Model: {DEFAULT_LLM_MODEL}")
print(f"Embedding Model: {DEFAULT_EMBEDDING_MODEL}")

# Test 1: LLM
print("\n1️⃣ Testing LLM...")
try:
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0)
    res = llm.invoke("Reply with the word 'CONNECTED' if you can read this.")
    content = res.content
    if isinstance(content, list):
        content = " ".join([b if isinstance(b, str) else b.get("text", "") for b in content])
    print(f"✅ LLM Response: {content.strip()}")
except Exception as e:
    print(f"❌ LLM Error: {e}")

# Test 2: Embeddings
print("\n2️⃣ Testing Embeddings...")
try:
    emb = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    vec = emb.embed_query("Hello world")
    print(f"✅ Embedding Success! Vector dimension: {len(vec)}")
except Exception as e:
    print(f"❌ Embedding Error: {e}")

print("\n" + "=" * 60)
