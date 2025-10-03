from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .models import openai_embed
from .models_fastembed import fastembed_dim, fastembed_embed


def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def batch(lst: list[Any], size: int) -> list[list[Any]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def select_provider(provider: str) -> Callable[[list[str], str], list[list[float]]]:
    p = (provider or "").lower()
    if p == "openai":
        return lambda texts, model_id: openai_embed(texts, model_id)
    if p in ("fastembed", "fe"):
        return lambda texts, model_id: fastembed_embed(texts, model_id=model_id, normalize=True)
    raise ValueError(f"Unknown provider: {provider}")


def infer_dim(provider: str, model_id: str) -> int:
    if provider.lower() in ("fastembed", "fe"):
        return fastembed_dim(model_id)
    if provider.lower() == "openai":
        # text-embedding-3-small = 1536, etc. If you have a helper, use it.
        from .models_openai import openai_dim_for_model

        return openai_dim_for_model(model_id)
    raise ValueError(f"Unknown provider: {provider}")


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def filter_idempotent(chunks_df: pd.DataFrame, existing_emb: pd.DataFrame | None) -> pd.DataFrame:
    """Return only chunks not yet embedded for this emb_v/episode."""
    if existing_emb is None or existing_emb.empty:
        return chunks_df
    done_ids = set(existing_emb["chunk_id"].astype(str).tolist())
    mask = ~chunks_df["chunk_id"].astype(str).isin(list(done_ids))
    return chunks_df.loc[mask].reset_index(drop=True)


def embed_chunks_df(
    chunks: pd.DataFrame,
    emb_v: str,
    provider: str,
    model_id: str,
    batch_size: int = 64,
    retries: int = 3,
    sleep_base_ms: int = 100,
) -> list[dict[str, Any]]:
    """
    Embed rows in `chunks` and return list[dict] compatible with the parquet schema.
    - Lazily infers vector dim from the first successful batch (no upfront infer_dim).
    - On batch failure after retries, emits error rows for that batch (status='error') instead of raising.
    """
    fn = select_provider(provider)

    rows_out: list[dict[str, Any]] = []
    texts = chunks["text"].astype(str).tolist()
    chunk_ids = chunks["chunk_id"].tolist()
    methods = chunks["method"].tolist()
    episode_ids = chunks["episode_id"].tolist()
    tokens_col = chunks.get("n_tokens", pd.Series([None] * len(chunks)))
    tokens_col = tokens_col.tolist() if tokens_col is not None else [None] * len(chunks)

    dim: int | None = None  # infer lazily

    i = 0
    while i < len(texts):
        j = min(i + batch_size, len(texts))
        batch = texts[i:j]

        attempts = 0
        ok = False
        vecs: list[list[float]] = []

        while attempts <= retries and not ok:
            try:
                vecs = fn(batch, model_id)
                ok = True
            except Exception as e:
                attempts += 1
                if attempts > retries:
                    # Emit error rows for this batch and continue
                    ts = int(time.time() * 1000)
                    for k in range(len(batch)):
                        rows_out.append(
                            {
                                "chunk_id": chunk_ids[i + k],
                                "episode_id": episode_ids[i + k],
                                "method": methods[i + k],
                                "emb_v": emb_v,
                                "dim": dim if dim is not None else None,
                                "model_id": model_id,
                                "provider": provider,
                                "created_at": ts,
                                "text_hash": _hash_text(batch[k]),
                                "tokens": (lambda x: int(x) if x is not None else None)(
                                    tokens_col[i + k]
                                ),
                                "vector": None,
                                "attempts": attempts,
                                "status": f"error: {type(e).__name__}: {e}",
                            }
                        )
                    ok = True  # exit retry loop for this batch
                else:
                    time.sleep((sleep_base_ms / 1000.0) * (2 ** (attempts - 1)))

        # Successful batch?
        if vecs:
            # infer dim once
            if dim is None and len(vecs) > 0:
                dim = len(vecs[0])

            ts = int(time.time() * 1000)
            for k, v in enumerate(vecs):
                v_arr = np.asarray(v, dtype=np.float32)
                if dim is None:
                    dim = len(v_arr)
                elif len(v_arr) != dim:
                    # guard: align on-the-fly if provider returned different dims (shouldn't happen)
                    dim = len(v_arr)
                rows_out.append(
                    {
                        "chunk_id": chunk_ids[i + k],
                        "episode_id": episode_ids[i + k],
                        "method": methods[i + k],
                        "emb_v": emb_v,
                        "dim": dim,
                        "model_id": model_id,
                        "provider": provider,
                        "created_at": ts,
                        "text_hash": _hash_text(batch[k]),
                        "tokens": (lambda x: int(x) if x is not None else None)(tokens_col[i + k]),
                        "vector": v_arr.tolist(),
                        "attempts": 1,
                        "status": "ok",
                    }
                )

        i = j

    return rows_out
