import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Ensure runtime directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
# Defaulting to fast, cost-effective vision-capable models
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Chunking Settings (Structure-Aware)
MAX_CHARACTERS = int(os.getenv("MAX_CHARACTERS", "3000"))
NEW_AFTER_N_CHARS = int(os.getenv("NEW_AFTER_N_CHARS", "2400"))
COMBINE_TEXT_UNDER_N_CHARS = int(os.getenv("COMBINE_TEXT_UNDER_N_CHARS", "500"))

# Retrieval Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
HNSW_SPACE = "cosine"
