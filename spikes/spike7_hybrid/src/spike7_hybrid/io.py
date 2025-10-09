from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths


def load_embeddings_df(emb_v: str, episode_id: str) -> pd.DataFrame:
    p = paths.EMB_DIR / emb_v / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing embeddings parquet: {p}")
    return pd.read_parquet(p)


def load_chunks_df(method: str, episode_id: str) -> pd.DataFrame:
    p = paths.CHUNKS_DIR / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {p}")
    return pd.read_parquet(p)


def load_qa_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"question", "episode_id", "answer_start_ts", "answer_end_ts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"QA csv missing columns: {missing}")
    return df
