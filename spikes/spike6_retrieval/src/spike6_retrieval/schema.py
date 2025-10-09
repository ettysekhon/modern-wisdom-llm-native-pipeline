from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class EvalParams:
    emb_v: str
    collection: str  # could be LIVE alias
    ks: list[int]
    tolerance_s: int
    top_k_search: int  # max K to request from qdrant per query


class QaRow(TypedDict):
    question: str
    episode_id: str
    answer_start_ts: float
    answer_end_ts: float
    optional_keywords: str | None


class ScoredDoc(TypedDict):
    id: str
    score: float
    payload: dict[str, Any]
