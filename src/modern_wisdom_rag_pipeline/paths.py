import os
from pathlib import Path


def repo_root() -> Path:
    # Use MW_DATA_DIR if set (for Fly.io deployments)
    if data_dir := os.getenv("MW_DATA_DIR"):
        return Path(data_dir).parent
    
    # Otherwise, try to find repo root by looking for data directory
    p = Path.cwd()
    if (p / "data").exists():
        return p
    
    # Try going up parent directories (max 4 levels)
    for parent in p.parents[:4]:
        if (parent / "data").exists():
            return parent
    
    # Fallback: use current directory
    return p


DATA_DIR = Path(os.getenv("MW_DATA_DIR", str(repo_root() / "data")))
DOCS_DIR = repo_root() / "docs" / "decisions"
CONTRACTS_DIR = DATA_DIR / "contracts"
SCHEMAS_DIR = CONTRACTS_DIR / "schemas"
SAMPLES_DIR = CONTRACTS_DIR / "sample_responses"
EVALS_DIR = DATA_DIR / "evals" / "rag"
CHUNKS_DIR = DATA_DIR / "chunks"
EMB_DIR = DATA_DIR / "embeddings"
DUCKDB_PATH = DATA_DIR / "duckdb" / "modern_wisdom.duckdb"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# index/version knobs
LIVE_ALIAS = "mw_chunks_live"
INDEX_VERSION = os.getenv("INDEX_VERSION", LIVE_ALIAS)

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
for d in (SCHEMAS_DIR, SAMPLES_DIR, EVALS_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

