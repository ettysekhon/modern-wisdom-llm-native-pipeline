from __future__ import annotations

import re
import time
from typing import Any, NotRequired, TypedDict

from spike7_hybrid.bm25 import score_bm25
from spike7_hybrid.hybrid import rrf_fuse, top_ids_by_score
from spike8_rag_contract import paths
from spike8_rag_contract.io import load_chunks_df
from spike8_rag_contract.qdrant import client, vector_search
from spike8_rag_contract.retrieval import embed_question_fastembed


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


def _tokset(s: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(s or "")}


def _overlap(q: str, t: str) -> int:
    return len(_tokset(q) & _tokset(t))


def _enrich(docs, chunks_df, episode_id: str) -> list[dict]:
    pld = {str(row["chunk_id"]): row.to_dict() for _, row in chunks_df.iterrows()}
    out = []
    for d in docs:
        did = d["id"] if isinstance(d, dict) else getattr(d, "id", "")
        score = d["score"] if isinstance(d, dict) else getattr(d, "score", 0.0)
        cid = str(did)
        row = pld.get(cid, {}) or {}
        out.append(
            {
                "chunk_id": cid,
                "score": float(score or 0.0),
                "start_ts": float(row.get("start_ts", 0.0)),
                "end_ts": float(row.get("end_ts", 0.0)),
                "text": row.get("text", ""),
                "episode_id": str(row.get("episode_id", episode_id)),
            }
        )
    return out


def _episode_local_hybrid(question: str, episode_id: str, top_k: int) -> tuple[list[dict], float]:
    # Params that worked well in Spike 7
    k_vec, k_lex, rrf_k = 20, 200, 60.0

    # Embed
    qv = embed_question_fastembed(question, model_id="BAAI/bge-small-en-v1.5")

    # Vector
    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)
    t0 = time.perf_counter()
    vec_docs, rt_ms_vec = vector_search(
        cli,
        collection=paths.INDEX_VERSION,
        episode_id=episode_id,
        q_vector=qv,
        top_k=k_vec,
    )
    vec_ids = top_ids_by_score(vec_docs, k_vec)

    # BM25 within episode
    chunks_df = load_chunks_df("sentence_bound", episode_id)
    corpus_texts = chunks_df["text"].astype(str).tolist()
    corpus_ids = chunks_df["chunk_id"].astype(str).tolist()
    bm0 = time.perf_counter()
    lex_scores = score_bm25(question, corpus_texts)
    rt_ms_bm25 = (time.perf_counter() - bm0) * 1000.0
    lex_rank = sorted(zip(corpus_ids, lex_scores, strict=False), key=lambda x: x[1], reverse=True)[
        :k_lex
    ]
    lex_ids = [cid for cid, _ in lex_rank]

    # RRF
    fused = rrf_fuse({"vec": vec_ids, "lex": lex_ids}, k=rrf_k)
    fused_ids = [pid for pid, _ in fused][:top_k]

    # Enrich
    pld = {str(row["chunk_id"]): row for _, row in chunks_df.iterrows()}
    results = []
    for cid in fused_ids:
        row = pld.get(str(cid))
        if row is None:
            row = {}
        score = next((float(s) for (pid, s) in fused if pid == cid), 0.0)
        results.append(
            {
                "chunk_id": str(cid),
                "score": float(score),
                "start_ts": _as_float(row.get("start_ts", 0.0)),
                "end_ts": _as_float(row.get("end_ts", 0.0)),
                "text": row.get("text", ""),
                "episode_id": row.get("episode_id", episode_id),
            }
        )

    retrieve_ms = (time.perf_counter() - t0) * 1000.0 + rt_ms_bm25
    return results, retrieve_ms


def _corpus_episode_locator(question: str, max_eps: int = 5) -> list[str]:
    """
    Simple BM25 over the entire transcript corpus to find episodes that actually
    mention the query tokens. Uses Spike 8 chunk parquet to keep things simple.
    """
    # Load all chunks for current index (one file per episode); if you have a global parquet, use it.
    # Here, we piggyback on Spike 8’s cached per-episode reader by scanning the index file list from DuckDB/paths.
    # If you have a central DuckDB table, switch to a SELECT ... GROUP BY episode_id ORDER BY MAX(score).
    import duckdb
    from spike4_embeddings import paths as p4

    db = duckdb.connect(p4.DUCKDB_PATH.as_posix(), read_only=True)

    df = db.execute("""
        SELECT episode_id, chunk_id, text
        FROM mw_chunks_live
    """).df()

    texts = df["text"].astype(str).tolist()
    scores = score_bm25(question, texts)
    df["__score"] = scores

    top = (
        df[["episode_id", "__score"]]
        .groupby("episode_id", as_index=False)
        .max()
        .sort_values(by="__score", ascending=False)  # type: ignore[arg-type]
        .head(max_eps)
    )
    return [str(x) for x in top["episode_id"].tolist()]


def rag_search(args: RagSearchArgs) -> dict[str, Any]:
    q = args.get("question", "")
    ep = args.get("episode_id")
    k = int(args.get("top_k", 8))
    scope = args.get("scope", "auto")

    # Handle corpus-only search (no episode_id)
    if ep is None:
        cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)
        qv = embed_question_fastembed(q, model_id="BAAI/bge-small-en-v1.5")
        t0 = time.perf_counter()
        corp_docs, rt_ms_corp = vector_search(
            cli, collection=paths.INDEX_VERSION, q_vector=qv, top_k=k * 6, episode_id=None
        )
        # Build cache and enrich results
        corpus_cache: dict[str, dict] = {}
        corpus_out: list[dict] = []
        for d in corp_docs:
            did = d["id"] if isinstance(d, dict) else getattr(d, "id", "")
            payload = (
                d.get("payload", {}) if isinstance(d, dict) else getattr(d, "payload", {}) or {}
            )
            ep_id = str(payload.get("episode_id", ""))
            if ep_id and ep_id not in corpus_cache:
                corpus_cache[ep_id] = {
                    str(row["chunk_id"]): row.to_dict()
                    for _, row in load_chunks_df("sentence_bound", ep_id).iterrows()
                }
            row = corpus_cache.get(ep_id, {}).get(str(did), {}) or {}
            corpus_out.append(
                {
                    "chunk_id": str(did),
                    "score": float(
                        d["score"] if isinstance(d, dict) else getattr(d, "score", 0.0) or 0.0
                    ),
                    "start_ts": float(row.get("start_ts", 0.0)),
                    "end_ts": float(row.get("end_ts", 0.0)),
                    "text": row.get("text", ""),
                    "episode_id": ep_id,
                }
            )
        # Keep top-k unique by score
        seen = set()
        final = []
        for r in corpus_out:
            if r["chunk_id"] in seen:
                continue
            final.append(r)
            seen.add(r["chunk_id"])
            if len(final) >= k:
                break
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "retrieved": final,
            "retrieve_ms": total_ms,
            "fallback_used": False,
            "scope": "corpus",
        }

    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    # 1) embed
    qv = embed_question_fastembed(q, model_id="BAAI/bge-small-en-v1.5")

    # 2) episode scope first (fast path)
    t0 = time.perf_counter()
    ep_docs, rt_ms_ep = vector_search(
        cli, collection=paths.INDEX_VERSION, q_vector=qv, top_k=k * 3, episode_id=ep
    )
    chunks_df = load_chunks_df("sentence_bound", ep)
    ep_res = _enrich(ep_docs, chunks_df, ep)[:k]
    took_ms = (time.perf_counter() - t0) * 1000.0

    if scope == "episode":
        return {
            "retrieved": ep_res,
            "retrieve_ms": took_ms,
            "fallback_used": False,
            "scope": "episode",
        }

    # 3) decide to broaden in "auto" if overlap is weak
    has_overlap = any(_overlap(q, r.get("text", "")) > 0 for r in ep_res)
    need_broaden = (scope == "corpus") or (scope == "auto" and not has_overlap)

    if not need_broaden:
        return {
            "retrieved": ep_res,
            "retrieve_ms": took_ms,
            "fallback_used": False,
            "scope": "episode",
        }

    # 4) corpus-wide pass
    t1 = time.perf_counter()
    corp_docs, rt_ms_corp = vector_search(
        cli, collection=paths.INDEX_VERSION, q_vector=qv, top_k=k * 6, episode_id=None
    )
    # enriching requires the right DF per episode; make a quick cache
    df_cache: dict[str, dict] = {}
    out: list[dict] = []
    for d in corp_docs:
        did = d["id"] if isinstance(d, dict) else getattr(d, "id", "")
        payload = d.get("payload", {}) if isinstance(d, dict) else getattr(d, "payload", {}) or {}
        ep_id = str(payload.get("episode_id", ep))
        if ep_id not in df_cache:
            df_cache[ep_id] = {
                str(row["chunk_id"]): row.to_dict()
                for _, row in load_chunks_df("sentence_bound", ep_id).iterrows()
            }
        row = df_cache[ep_id].get(str(did), {}) or {}
        out.append(
            {
                "chunk_id": str(did),
                "score": float(
                    d["score"] if isinstance(d, dict) else getattr(d, "score", 0.0) or 0.0
                ),
                "start_ts": float(row.get("start_ts", 0.0)),
                "end_ts": float(row.get("end_ts", 0.0)),
                "text": row.get("text", ""),
                "episode_id": ep_id,
            }
        )

    # keep top-k unique by score (already sorted by qdrant)
    seen = set()
    final = []
    for r in out:
        if r["chunk_id"] in seen:
            continue
        final.append(r)
        seen.add(r["chunk_id"])
        if len(final) >= k:
            break

    total_ms = took_ms + (time.perf_counter() - t1) * 1000.0
    return {"retrieved": final, "retrieve_ms": total_ms, "fallback_used": True, "scope": "corpus"}
