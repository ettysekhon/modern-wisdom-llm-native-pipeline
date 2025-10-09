from pathlib import Path

import pandas as pd

from . import paths


def embeddings_parquet_path(emb_v: str, episode_id: str) -> Path:
    return paths.EMB_DIR / emb_v / f"episode_id={episode_id}" / "part-00000.snappy.parquet"


def chunks_parquet_path(method: str, episode_id: str) -> Path:
    return paths.CHUNKS_DIR / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"


def load_embeddings(emb_v: str, episode_id: str) -> pd.DataFrame:
    p = embeddings_parquet_path(emb_v, episode_id)
    if not p.exists():
        raise FileNotFoundError(f"Missing embeddings parquet: {p}")
    df = pd.read_parquet(p)
    if "vector" not in df.columns or "chunk_id" not in df.columns:
        raise ValueError("Embeddings parquet must include 'vector' and 'chunk_id'")
    return df


def load_chunks(method: str, episode_id: str) -> pd.DataFrame:
    p = chunks_parquet_path(method, episode_id)
    if not p.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {p}")
    return pd.read_parquet(p)


def maybe_load_chunks(method: str, episode_id: str):
    p = chunks_parquet_path(method, episode_id)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def join_payload(emb_df: pd.DataFrame, chunk_df: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join minimal chunk metadata if available; otherwise pass embeddings as-is."""
    if chunk_df is None:
        return emb_df
    # Select a lean set of payload columns from chunks
    cols = {
        "chunk_id",
        "episode_id",
        "method",
        "text",
        "n_tokens",
        "duration_s",
        "guest",
        "episode_title",
        "publish_date",
        "headline",
    }
    present = [c for c in cols if c in chunk_df.columns]
    meta = chunk_df[present].drop_duplicates(subset=["chunk_id"])  # type: ignore[arg-type]
    df = emb_df.merge(meta, on=["chunk_id", "episode_id", "method"], how="left")
    if "episode_title" in df.columns and "title" not in df.columns:
        df.rename(columns={"episode_title": "title"}, inplace=True)
    return df


__all__ = [
    "embeddings_parquet_path",
    "chunks_parquet_path",
    "load_embeddings",
    "load_chunks",
    "maybe_load_chunks",
    "join_payload",
]
