import hashlib
import pickle
from pathlib import Path
from typing import List, Any, Optional

from unstructured.partition.pdf import partition_pdf
from config.settings import CACHE_DIR


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for cache invalidation."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def partition_document(
    file_path: str | Path,
    use_cache: bool = True,
    cache_dir: Optional[Path] = CACHE_DIR,
    force_reload: bool = False
) -> List[Any]:
    """
    Extract layout-aware elements from a PDF using Unstructured with 'hi_res' strategy.
    
    Features:
    - Computer vision layout detection (multi-column reading order)
    - Reconstructed HTML table structures
    - Base64 image payload extraction
    - Automatic caching to prevent re-parsing heavy PDFs
    """
    pdf_path = Path(file_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Document not found at: {pdf_path}")

    # Check cache first
    cache_file = None
    if use_cache and cache_dir:
        file_hash = _compute_file_hash(pdf_path)
        cache_file = cache_dir / f"{pdf_path.stem}_{file_hash}_elements.pkl"
        
        if cache_file.exists() and not force_reload:
            print(f"📦 Loading cached elements from: {cache_file.name}")
            with open(cache_file, "rb") as f:
                elements = pickle.load(f)
            print(f"✅ Loaded {len(elements)} elements from cache.")
            return elements

    print(f"📄 Partitioning document (hi_res layout + tables + images): {pdf_path.name}")
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    print(f"✅ Extracted {len(elements)} elements.")

    # Save to cache
    if use_cache and cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(elements, f)
        print(f"💾 Cached parsed elements to: {cache_file.name}")

    return elements
