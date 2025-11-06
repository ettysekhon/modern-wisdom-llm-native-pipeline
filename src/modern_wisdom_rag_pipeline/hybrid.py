from __future__ import annotations

from typing import Any


def rrf_fuse(rankings: dict[str, list[str]], k: float = 60.0) -> list[tuple[str, float]]:
    """
    rankings: {"vec": [id1,id2,...], "lex": [...]} → returns [(id, fused_score), ...] sorted desc
    """
    scores: dict[str, float] = {}
    for _, ranked_ids in rankings.items():
        for rank_idx, pid in enumerate(ranked_ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank_idx)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def top_ids_by_score(items: list[Any], n: int) -> list[str]:
    return [d.id for d in sorted(items, key=lambda x: x.score, reverse=True)[:n]]

