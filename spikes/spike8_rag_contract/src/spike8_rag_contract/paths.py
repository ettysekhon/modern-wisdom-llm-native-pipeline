import os
from pathlib import Path


def repo_root() -> Path:
    p = Path.cwd()
    return p if (p / "data").exists() else p.parents[3]


DATA_DIR = repo_root() / "data"
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

for d in (SCHEMAS_DIR, SAMPLES_DIR, EVALS_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)
