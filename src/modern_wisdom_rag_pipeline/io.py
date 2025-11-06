from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import paths


def embeddings_parquet_path(emb_v: str, episode_id: str) -> Path:
    return paths.EMB_DIR / emb_v / f"episode_id={episode_id}" / "part-00000.snappy.parquet"


def load_chunks_df(method: str, episode_id: str) -> pd.DataFrame:
    """
    Robust chunk loader:
    - Supports multiple part files (hive partitions)
    - Aligns columns by name (union_by_name=1)
    - Dynamically selects a text-like column and exposes it as 'text'
    """
    ep_dir = paths.CHUNKS_DIR / method / f"episode_id={episode_id}"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Missing chunks directory: {ep_dir}")

    pat = (ep_dir / "part-*.parquet").as_posix()
    if not list(ep_dir.glob("part-*.parquet")):
        raise FileNotFoundError(f"Missing chunks parquet: {ep_dir}/part-00000.snappy.parquet")

    con = duckdb.connect(database=":memory:")
    try:
        # Read everything unioned by name; don't guess columns in SQL
        df = con.execute(
            "SELECT * FROM read_parquet(?, hive_partitioning=1, union_by_name=1)",
            [pat],
        ).fetchdf()
    finally:
        con.close()

    # Normalise required columns
    if "chunk_id" not in df.columns:
        # If older runs stored it as string index or similar, synthesise IDs
        df["chunk_id"] = [str(i) for i in range(len(df))]

    # Try to standardise timestamps if present
    if "start_ts" not in df.columns:
        # Sometimes 'start' is used
        if "start" in df.columns:
            df["start_ts"] = df["start"].astype(float, errors="ignore")
        else:
            df["start_ts"] = 0.0
    if "end_ts" not in df.columns:
        if "end" in df.columns:
            df["end_ts"] = df["end"].astype(float, errors="ignore")
        else:
            df["end_ts"] = 0.0

    # Choose a text-like column and expose as 'text'
    text_candidates = [
        "text",
        "content",
        "segment_text",
        "segment",
        "transcript",
        "raw_text",
        "body",
        "snippet",
        "utterance",
        "line",
        "value",
    ]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        # As a last resort, make empty strings
        df["text"] = ""
    else:
        df["text"] = df[text_col].astype(str)

    # Enforce types
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["start_ts"] = pd.to_numeric(df["start_ts"], errors="coerce")
    df["start_ts"] = df["start_ts"].fillna(0.0)
    df["end_ts"] = pd.to_numeric(df["end_ts"], errors="coerce")
    df["end_ts"] = df["end_ts"].fillna(0.0)

    return df


def load_embeddings(emb_v: str, episode_id: str) -> pd.DataFrame:
    p = embeddings_parquet_path(emb_v, episode_id)
    if not p.exists():
        raise FileNotFoundError(f"Missing embeddings parquet: {p}")
    df = pd.read_parquet(p)
    if "vector" not in df.columns or "chunk_id" not in df.columns:
        raise ValueError("Embeddings parquet must include 'vector' and 'chunk_id'")
    return df
