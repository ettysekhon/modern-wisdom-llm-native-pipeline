from dataclasses import dataclass, field
from typing import Any, TypedDict


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


@dataclass(frozen=True)
class HybridParams:
    k_vec: int = 20  # vector search size
    k_lex: int = 200  # lexical pool size
    rrf_k: float = 60.0  # RRF parameter
    tolerance_s: int = 7
    ks_report: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass(frozen=True)
class Filters:
    guest: str | None = None
    date_from: str | None = None  # 'YYYY-MM-DD'
    date_to: str | None = None
