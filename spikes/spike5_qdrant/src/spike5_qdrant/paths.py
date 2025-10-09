from pathlib import Path


def repo_root() -> Path:
    # same heuristic as other spikes
    cwd = Path.cwd()
    return cwd if (cwd / "data").exists() else Path(__file__).resolve().parents[3]


DATA_DIR = repo_root() / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
EMB_DIR = DATA_DIR / "embeddings"
SNAP_DIR = DATA_DIR / "qdrant" / "snapshots"

# Qdrant defaults (override via env or CLI if needed)
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None  # set via env for cloud
