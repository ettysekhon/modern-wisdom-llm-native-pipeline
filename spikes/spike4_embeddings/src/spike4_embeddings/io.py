from pathlib import Path

import duckdb
import pandas as pd

from . import paths
from .schema import validate_embed_df_columns


def load_chunks(method: str, episode_id: str, chunks_dir: Path | None = None) -> pd.DataFrame:
    chunks_dir = chunks_dir or paths.CHUNKS_DIR
    p = chunks_dir / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {p}")
    return pd.read_parquet(p)


def read_existing_embeddings(
    emb_v: str, episode_id: str, emb_dir: Path | None = None
) -> pd.DataFrame | None:
    emb_dir = emb_dir or paths.EMB_DIR
    p = emb_dir / emb_v / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def write_embeddings(
    rows: list[dict], emb_v: str, episode_id: str, emb_dir: Path | None = None
) -> Path:
    if not rows:
        raise RuntimeError("No embedding rows to write.")
    df = pd.DataFrame(rows)
    missing = validate_embed_df_columns(df)
    if missing:
        raise ValueError(f"Embedding schema missing columns: {missing}")

    emb_dir = emb_dir or paths.EMB_DIR
    out_dir = emb_dir / emb_v / f"episode_id={episode_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part-00000.snappy.parquet"
    df.to_parquet(path, index=False)
    return path


def upsert_duckdb_embeddings(parquet_path: Path, db_path: Path | None = None):
    from . import paths

    db_path = db_path or paths.DUCKDB_PATH
    con = duckdb.connect(str(db_path))

    # 1) Ensure table exists with a PK (required for OR REPLACE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT PRIMARY KEY,
            episode_id TEXT,
            method TEXT,
            emb_v TEXT,
            dim INTEGER,
            model_id TEXT,
            provider TEXT,
            created_at BIGINT,
            text_hash TEXT,
            tokens INTEGER,
            vector FLOAT[],
            attempts INTEGER,
            status TEXT
        );
    """)

    # 2) Upsert by name (columns mapped by name; safe if order changes)
    con.execute(
        """
        INSERT OR REPLACE INTO embeddings BY NAME
        SELECT * FROM read_parquet(?);
    """,
        [str(parquet_path)],
    )

    con.close()
