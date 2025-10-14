from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths


def embeddings_parquet_path(emb_v: str, episode_id: str) -> Path:
    return paths.EMB_DIR / emb_v / f"episode_id={episode_id}" / "part-00000.snappy.parquet"


def load_chunks_df(method: str, episode_id: str) -> pd.DataFrame:
    p = paths.CHUNKS_DIR / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {p}")
    return pd.read_parquet(p)


def load_embeddings(emb_v: str, episode_id: str) -> pd.DataFrame:
    p = embeddings_parquet_path(emb_v, episode_id)
    if not p.exists():
        raise FileNotFoundError(f"Missing embeddings parquet: {p}")
    df = pd.read_parquet(p)
    if "vector" not in df.columns or "chunk_id" not in df.columns:
        raise ValueError("Embeddings parquet must include 'vector' and 'chunk_id'")
    return df
