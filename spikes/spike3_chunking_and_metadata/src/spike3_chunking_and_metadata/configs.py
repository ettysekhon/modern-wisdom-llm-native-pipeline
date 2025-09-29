from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import tomli_w

from .paths import CHUNKS_DIR


def _read_params_from_chunks(episode_id: str, method: str, chunks_dir: Path = CHUNKS_DIR) -> dict:
    p = chunks_dir / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    params = {"method": method}
    # token-based
    if row.get("param_size_tokens") is not None:
        params["size_tokens"] = int(row.get("param_size_tokens") or 0)
    if row.get("param_overlap_tokens") is not None:
        params["overlap_tokens"] = int(row.get("param_overlap_tokens") or 0)
    # time-based
    if row.get("param_window_s") is not None:
        params["window_seconds"] = int(row.get("param_window_s") or 0)
    if row.get("param_overlap_s") is not None:
        params["overlap_seconds"] = int(row.get("param_overlap_s") or 0)
    return params


def write_chunking_toml(
    episode_id: str,
    winner_method: str,
    out_path: Path = Path("configs") / "chunking.toml",
    chunks_dir: Path = CHUNKS_DIR,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = _read_params_from_chunks(episode_id, winner_method, chunks_dir)
    if not params:
        params = {"method": winner_method}  # minimal fallback
    # merge with existing if present
    if out_path.exists():
        with out_path.open("rb") as f:
            existing = tomllib.load(f)
    else:
        existing = {}
    merged = dict(existing)
    merged.update(params)
    with out_path.open("wb") as f:
        tomli_w.dump(merged, f)
    return out_path
