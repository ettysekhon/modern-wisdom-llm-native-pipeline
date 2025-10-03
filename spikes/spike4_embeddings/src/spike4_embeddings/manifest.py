from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import paths


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_embed_manifest(emb_v: str) -> dict[str, Any]:
    """
    Aggregate all episode parquet parts for a given emb_v into a deterministic manifest.
    Looks under: data/embeddings/<emb_v>/episode_id=*/part-*.parquet
    """
    root = paths.EMB_DIR / emb_v
    if not root.exists():
        raise FileNotFoundError(f"Embedding dir not found for emb_v={emb_v}: {root}")

    parts: list[Path] = sorted(root.glob("episode_id=*/part-*.snappy.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts under {root}")

    episodes: list[dict[str, Any]] = []
    dim: int | None = None
    provider: str | None = None
    model_id: str | None = None

    for part in parts:
        df = pd.read_parquet(part)
        if df.empty:
            continue

        if dim is None and bool(df["dim"].notna().any()):
            dim = int(df["dim"].dropna().unique()[0])
        if provider is None and bool(df["provider"].notna().any()):
            provider = str(df["provider"].dropna().unique()[0])
        if model_id is None and bool(df["model_id"].notna().any()):
            model_id = str(df["model_id"].dropna().unique()[0])

        eid = part.parent.name.split("=", 1)[1]
        sha = _sha256_file(part)

        episodes.append(
            {
                "episode_id": eid,
                "parquet": str(part),
                "rows": int(len(df)),
                "sum_tokens": int(df["tokens"].fillna(0).sum()),
                "sha256": sha,
            }
        )

    if dim is None:
        raise ValueError("Could not infer embedding dim from parquet files.")
    if provider is None:
        provider = "unknown"
    if model_id is None:
        model_id = "unknown"

    manifest: dict[str, Any] = {
        "emb_v": emb_v,
        "dim": dim,
        "provider": provider,
        "model_id": model_id,
        "created_at": int(time.time()),
        "episodes": episodes,
        "prep": {"method": "sentence_bound", "size_tokens": 700, "overlap_tokens": 100},
    }
    return manifest


def write_embed_manifest(emb_v: str) -> Path:
    """
    Builds the manifest and writes to data/embeddings/<emb_v>/_manifest.json
    (idempotent: overwrites the file deterministically).
    """
    manifest = build_embed_manifest(emb_v)
    out = paths.EMB_DIR / emb_v / "_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out
