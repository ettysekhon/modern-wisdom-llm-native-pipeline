from __future__ import annotations

import numpy as np

_FE_MODEL = None
_FE_MODEL_ID = None


def _load_fe(model_id: str, cache_dir: str | None = None):
    global _FE_MODEL, _FE_MODEL_ID
    if _FE_MODEL is None or model_id != _FE_MODEL_ID:
        from fastembed import TextEmbedding

        _FE_MODEL = TextEmbedding(model_id=model_id, cache_dir=cache_dir)
        _FE_MODEL_ID = model_id
    return _FE_MODEL


def fastembed_embed(texts: list[str], model_id: str, normalize: bool = True) -> list[list[float]]:
    """Return float32 vectors; deterministic for same model+input."""
    model = _load_fe(model_id)
    # fastembed returns an iterable of np.ndarray
    vecs = list(model.embed(texts))
    out: list[list[float]] = []
    for v in vecs:
        v = v.astype(np.float32, copy=False)
        if normalize:
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        out.append(v.tolist())
    return out


def fastembed_dim(model_id: str) -> int:
    """Infer dim by embedding a tiny sample once (cached model)."""
    # small, safe call; runs once per process/model_id
    v = fastembed_embed(["dim_probe"], model_id=model_id, normalize=False)[0]
    return len(v)
