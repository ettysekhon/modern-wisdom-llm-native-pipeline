import os
from pathlib import Path


def repo_root() -> Path:
    env = os.getenv("MW_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "data").exists():
            return p
    # walk up from CWD
    p = Path.cwd().resolve()
    for up in [p] + list(p.parents):
        if (up / "data" / "transcripts").exists():
            return up
        if (up / "data").exists() and (up / "pyproject.toml").exists():
            return up
    # walk up from this file
    here = Path(__file__).resolve()
    for up in [here] + list(here.parents):
        if (up.parent / "data").exists():
            return up.parent
    return Path.cwd().resolve()


ROOT = repo_root()
DATA_DIR = ROOT / "data"

# Inputs from Spike 3
CHUNKS_DIR = DATA_DIR / "chunks"

# Outputs for Spike 4
EMB_DIR = DATA_DIR / "embeddings"
EVALS_DIR = DATA_DIR / "evals" / "embed"

# Optional DB
DUCKDB_PATH = DATA_DIR / "duckdb" / "modern_wisdom.duckdb"
