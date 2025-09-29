import os
from pathlib import Path

import tiktoken


def repo_root() -> Path:
    """Find the repo root by walking up until we see /data/transcripts or a marker file."""
    # 1) ENV override (handy for CI or custom layouts)
    env = os.getenv("MW_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "data" / "transcripts").exists() or (p / "data").exists():
            return p

    # 2) Start from CWD and walk up
    p = Path.cwd().resolve()
    for up in [p] + list(p.parents):
        if (up / "data" / "transcripts").exists():
            return up
        if (up / "data").exists() and (up / "pyproject.toml").exists():
            return up  # reasonable fallback

    # 3) Start from this file and walk up
    here = Path(__file__).resolve()
    for up in [here] + list(here.parents):
        if (up / "data" / "transcripts").exists():
            return up
        if (up / "data").exists() and (up / "pyproject.toml").exists():
            return up

    # 4) Last resort: CWD
    return Path.cwd().resolve()


DATA_DIR = repo_root() / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
EVALS_DIR = DATA_DIR / "evals" / "chunking"
DOCS_DIR = repo_root() / "docs" / "decisions"
DUCKDB_PATH = DATA_DIR / "duckdb" / "modern_wisdom.duckdb"

ENC = tiktoken.get_encoding("cl100k_base")
