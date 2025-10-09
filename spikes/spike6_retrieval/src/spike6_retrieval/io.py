from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths


def embeddings_parquet_path(emb_v: str, episode_id: str) -> Path:
    return paths.EMB_DIR / emb_v / f"episode_id={episode_id}" / "part-00000.snappy.parquet"


def load_embeddings_df(emb_v: str, episode_id: str) -> pd.DataFrame:
    p = embeddings_parquet_path(emb_v, episode_id)
    if not p.exists():
        raise FileNotFoundError(f"Missing embeddings parquet: {p}")
    df = pd.read_parquet(p)
    # Expect columns: chunk_id, vector, start_ts, end_ts, text, episode_id, model_id, provider, emb_v...
    if "vector" not in df.columns or "chunk_id" not in df.columns:
        raise ValueError("Embeddings parquet must include 'vector' and 'chunk_id'")
    return df


def load_qa_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"question", "episode_id", "answer_start_ts", "answer_end_ts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"QA csv missing columns: {missing}")
    return df


def select_provider_from_embeddings(df: pd.DataFrame) -> tuple[str, str]:
    """Infer provider+model_id from embeddings parquet rows."""
    prov = df["provider"].dropna().unique() if "provider" in df.columns else []
    model = df["model_id"].dropna().unique() if "model_id" in df.columns else []
    provider = str(prov[0]) if len(prov) else "fastembed"
    model_id = str(model[0]) if len(model) else "BAAI/bge-small-en-v1.5"
    return provider, model_id
