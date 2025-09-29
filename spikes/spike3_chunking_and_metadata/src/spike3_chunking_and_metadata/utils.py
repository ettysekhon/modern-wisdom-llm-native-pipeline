import re
import uuid

from .paths import ENC


def count_tokens(text: str) -> int:
    return len(ENC.encode(text or ""))


def time_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def chunk_uuid(episode_id: str, method: str, start_ts: float, end_ts: float, chunk_v: str) -> str:
    ns = uuid.uuid5(uuid.NAMESPACE_URL, f"mw:{episode_id}:{method}:{chunk_v}")
    return str(uuid.uuid5(ns, f"{start_ts:.3f}-{end_ts:.3f}"))


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    return s


def tokenize_for_bm25(texts: list[str]) -> list[list[str]]:
    """Tokenize documents into word lists for BM25 ranking."""
    return [re.findall(r"[a-zA-Z0-9']+", (t or "").lower()) for t in texts]


__all__ = [
    "count_tokens",
    "time_overlap",
    "chunk_uuid",
    "clean_text",
    "tokenize_for_bm25",
]
