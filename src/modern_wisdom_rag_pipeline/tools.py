from __future__ import annotations

import re
import time
from typing import Any, NotRequired, TypedDict

from . import paths
from .bm25 import score_bm25
from .hybrid import rrf_fuse
from .io import load_chunks_df
from .qdrant import client, vector_search
from .retrieval import embed_question_fastembed


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class RagSearchArgs(TypedDict):
    question: str
    episode_id: NotRequired[str]
    top_k: int
    scope: str  # episode | corpus | auto


WORD_RE = re.compile(r"\w+")
YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _tokset(s: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(s or "")}


def _overlap(q: str, t: str) -> int:
    return len(_tokset(q) & _tokset(t))


def _safe_load_chunks_df(ep_id: str):
    """Load chunks df; return empty-frame shaped mapping on failure."""
    try:
        return load_chunks_df("sentence_bound", ep_id)
    except Exception:
        import pandas as pd

        return pd.DataFrame(
            {"chunk_id": [], "start_ts": [], "end_ts": [], "text": [], "publish_date": []}
        )


def _enrich_many(docs, df_cache: dict[str, dict], default_ep: str | None = None) -> list[dict]:
    """
    Enrich Qdrant points with chunk metadata from parquet; builds per-episode row cache on demand.
    Falls back to Qdrant payload if local chunks are unavailable.
    """
    out = []
    for d in docs:
        did = d["id"] if isinstance(d, dict) else getattr(d, "id", "")
        payload = d.get("payload", {}) if isinstance(d, dict) else getattr(d, "payload", {}) or {}
        ep_id = (
            str(payload.get("episode_id", default_ep or ""))
            if payload is not None
            else (default_ep or "")
        )

        if ep_id and ep_id not in df_cache:
            df = _safe_load_chunks_df(ep_id)
            df_cache[ep_id] = {str(row["chunk_id"]): row.to_dict() for _, row in df.iterrows()}

        row = df_cache.get(ep_id, {}).get(str(did), {}) or {}

        # Fall back to Qdrant payload if local chunks are unavailable
        out.append(
            {
                "chunk_id": str(did),
                "score": float(
                    d["score"] if isinstance(d, dict) else getattr(d, "score", 0.0) or 0.0
                ),
                "start_ts": float(row.get("start_ts") or payload.get("start_ts", 0.0)),
                "end_ts": float(row.get("end_ts") or payload.get("end_ts", 0.0)),
                "text": row.get("text") or payload.get("text", ""),
                "episode_id": ep_id,
                "publish_date": row.get("publish_date") or payload.get("publish_date", ""),
            }
        )
    return out


def _topk_unique(recs: list[dict], k_: int) -> list[dict]:
    seen, final = set(), []
    for r in recs:
        cid = r.get("chunk_id")
        if cid in seen:
            continue
        final.append(r)
        seen.add(cid)
        if len(final) >= k_:
            break
    return final


def _ids_sorted_by_vecscore(docs: list[dict]) -> list[str]:
    """IDs sorted by vector score (desc)."""
    return [d["chunk_id"] for d in sorted(docs, key=lambda x: -float(x.get("score", 0.0)))]


def _ids_sorted_by_bm25(question: str, docs: list[dict]) -> list[str]:
    """IDs sorted by BM25 score (desc) over the candidate texts."""
    texts = [d.get("text", "") for d in docs]
    scores = score_bm25(query=question, docs=texts)
    ord_idx = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
    return [docs[i]["chunk_id"] for i in ord_idx]


def _apply_year_boost(
    order_ids: list[str],
    fused_pairs: list[tuple[str, float]],
    id2doc: dict[str, dict],
    years: list[int],
) -> list[str]:
    """
    Apply a tiny year-aware bias: if publish_date year is in query years, nudge forward.
    We re-rank by subtracting a small delta from fused reciprocal rank score (i.e., better).
    """
    if not years:
        return order_ids
    hints = set(years)
    # Build mutable score map from fused pairs (higher is better in typical RRF implementations)
    scores = {cid: float(s) for cid, s in fused_pairs}
    for cid in order_ids:
        doc = id2doc.get(cid, {})
        pd = str(doc.get("publish_date") or "")
        yr = None
        if len(pd) >= 4 and pd[:4].isdigit():
            yr = int(pd[:4])
        if yr and yr in hints:
            # Small positive bump
            scores[cid] = scores.get(cid, 0.0) + 0.25
    # Return IDs sorted by bumped score descending
    return sorted(order_ids, key=lambda x: -scores.get(x, 0.0))


def _apply_topic_boost(order_ids: list[str], id2doc: dict[str, dict]) -> list[str]:
    """
    Ensure we bias slightly toward chunks that actually mention discipline/routines/habits.
    This is a soft tie-breaker applied after RRF (stable for non-ties).
    """
    topic = {"discipline", "disciplined", "routines", "routine", "habit", "habits"}

    def has_topic(cid: str) -> bool:
        t = (id2doc.get(cid, {}).get("text") or "").lower()
        return any(w in t for w in topic)

    # Stable sort with key: topic first (False < True implies reverse logic)
    return sorted(order_ids, key=lambda cid: (not has_topic(cid)))


def _hybrid_rerank(question: str, docs: list[dict], k: int, years: list[int]) -> list[dict]:
    """
    RRF(vec-order, bm25-order) → optional year/topic boosts → top-k unique.
    Uses imported rrf_fuse signature that accepts {"name": [ids...]} and returns [(id, score)].
    """
    if not docs:
        return []

    # Build id->doc map
    by_id = {d["chunk_id"]: d for d in docs}

    # Vector order (by vector score descending)
    vec_ids = _ids_sorted_by_vecscore(docs)

    # BM25 order (by lexical score descending)
    bm_ids = _ids_sorted_by_bm25(question, docs)

    # Fuse rankings
    fused: list[tuple[str, float]] = rrf_fuse({"vec": vec_ids, "lex": bm_ids}, k=60.0)
    fused_ids = [cid for cid, _ in fused]

    # Small year bias
    fused_ids = _apply_year_boost(fused_ids, fused, by_id, years)

    # Small topic bias (discipline/routine/habits)
    fused_ids = _apply_topic_boost(fused_ids, by_id)

    # Take top-k unique and return docs in that order
    final_ids = []
    seen = set()
    for cid in fused_ids:
        if cid in seen:
            continue
        seen.add(cid)
        final_ids.append(cid)
        if len(final_ids) >= k:
            break
    return [by_id[cid] for cid in final_ids]


def rag_search(args: RagSearchArgs) -> dict[str, Any]:
    q = args.get("question", "") or ""
    ep = args.get("episode_id")
    k = int(args.get("top_k", 8))
    scope = args.get("scope", "auto")

    # Normalise episode_id ("" | "all" | "corpus" | None) → None
    if isinstance(ep, str):
        ep_norm = ep.strip().lower()
        if ep_norm in {"", "all", "corpus", "none", "null"}:
            ep = None
    if ep is None and scope != "episode":
        scope = "corpus"

    # If caller insists on episode scope but no episode_id, return empty
    if scope == "episode" and ep is None:
        return {"retrieved": [], "retrieve_ms": 0.0, "fallback_used": False, "scope": "episode"}

    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    # Embed question once
    qv = embed_question_fastembed(q, model_id="BAAI/bge-small-en-v1.5")
    years = [int(y) for y in YEAR_RE.findall(q)]

    # No episode_id: corpus-only path
    if ep is None:
        t0 = time.perf_counter()
        corp_docs, _ = vector_search(
            cli, collection=paths.INDEX_VERSION, q_vector=qv, top_k=k * 6, episode_id=None
        )
        corpus_cache: dict[str, dict] = {}
        enriched = _enrich_many(corp_docs, corpus_cache, default_ep=None)

        # Hybrid rerank on candidate pool
        final = _hybrid_rerank(q, enriched, k, years)

        return {
            "retrieved": final,
            "retrieve_ms": (time.perf_counter() - t0) * 1000.0,
            "fallback_used": False,
            "scope": "corpus",
        }

    # We have an episode_id; try episode first
    t0 = time.perf_counter()
    ep_docs, _ = vector_search(
        cli, collection=paths.INDEX_VERSION, q_vector=qv, top_k=k * 3, episode_id=ep
    )

    ep_df_cache: dict[str, dict] = {}
    ep_df_cache[ep] = {
        str(row["chunk_id"]): row.to_dict() for _, row in _safe_load_chunks_df(ep).iterrows()
    }
    ep_out = _enrich_many(ep_docs, ep_df_cache, default_ep=ep)

    # Hybrid rerank within episode
    ep_res = _hybrid_rerank(q, ep_out, k, years)
    took_ms = (time.perf_counter() - t0) * 1000.0

    # If strict episode scope, return immediately
    if scope == "episode":
        return {
            "retrieved": ep_res,
            "retrieve_ms": took_ms,
            "fallback_used": False,
            "scope": "episode",
        }

    # Decide to broaden in auto mode if overlap is weak, or if scope == corpus
    has_overlap = any(_overlap(q, r.get("text", "")) > 0 for r in ep_res)
    need_broaden = (scope == "corpus") or (scope == "auto" and not has_overlap)

    if not need_broaden:
        return {
            "retrieved": ep_res,
            "retrieve_ms": took_ms,
            "fallback_used": False,
            "scope": "episode",
        }

    # Corpus-wide fallback (then hybrid rerank)
    t1 = time.perf_counter()
    corp_docs, _ = vector_search(
        cli, collection=paths.INDEX_VERSION, q_vector=qv, top_k=k * 6, episode_id=None
    )
    corpus_cache_b: dict[str, dict] = {}
    corp_out = _enrich_many(corp_docs, corpus_cache_b, default_ep=None)

    final = _hybrid_rerank(q, corp_out, k, years)
    total_ms = took_ms + (time.perf_counter() - t1) * 1000.0

    return {"retrieved": final, "retrieve_ms": total_ms, "fallback_used": True, "scope": "corpus"}
