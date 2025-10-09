from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def compute_hit_mrr(
    ranked_docs: list[dict[str, Any]],
    answer_start: float,
    answer_end: float,
    ks: list[int],
    tolerance_s: float = 7.0,
) -> dict:
    """ranked_docs = [{'payload': {'start_ts', 'end_ts', ...}, 'score': ...}, ...]"""
    first_hit_rank: int | None = None
    for idx, d in enumerate(ranked_docs, start=1):
        pld = d.get("payload", {}) or {}
        c_start = float(pld.get("start_ts", 0.0))
        c_end = float(pld.get("end_ts", 0.0))
        overlap = time_overlap(c_start, c_end, answer_start - tolerance_s, answer_end + tolerance_s)
        if overlap >= 1.0:
            first_hit_rank = idx
            break

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Hit@{k}"] = 1.0 if (first_hit_rank is not None and first_hit_rank <= k) else 0.0
    metrics["MRR"] = (1.0 / first_hit_rank) if first_hit_rank else 0.0
    return metrics


def p95(latencies_ms: Iterable[float]) -> float:
    arr = np.array(list(latencies_ms), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 95))
