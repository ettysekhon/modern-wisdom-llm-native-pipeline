from pathlib import Path

REPO = Path.cwd()
DATA = REPO / "data"
DOCS = REPO / "docs" / "decisions"

EMB_DIR = DATA / "embeddings"
EVALS_DIR = DATA / "evals" / "retrieval"
QA_CSV_DEFAULT = DATA / "qa" / "labels.csv"

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY: str | None = None

LIVE_ALIAS = "mw_chunks_live"
