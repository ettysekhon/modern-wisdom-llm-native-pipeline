from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_chunk_text_map(chunks_dir: Path, method: str, episode_id: str) -> dict[str, str]:
    p = chunks_dir / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if "chunk_id" not in df.columns or "text" not in df.columns:
        return {}
    return dict(zip(df["chunk_id"].astype(str), df["text"].astype(str), strict=False))
