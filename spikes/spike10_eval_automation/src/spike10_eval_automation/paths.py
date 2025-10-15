from pathlib import Path


def repo_root() -> Path:
    p = Path.cwd()
    return p if (p / "data").exists() else p.parents[3]


REPO = repo_root()
DATA = REPO / "data"
EVALS_DIR = DATA / "evals" / "retrieval"
CONTRACTS_DIR = DATA / "contracts"
REPORTS_DIR = DATA / "evals" / "automation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
