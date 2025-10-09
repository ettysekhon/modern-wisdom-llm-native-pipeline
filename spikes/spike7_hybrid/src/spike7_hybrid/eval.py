from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from . import paths
from .bm25 import score_bm25
from .hybrid import rrf_fuse, top_ids_by_score
from .io import (
    load_chunks_df,
    load_qa_csv,
)  # keep load_embeddings_df import (harmless)
from .qdrant import client, vector_search
from .query_embedder import embed_questions  # ← added
from .schema import Filters, HybridParams

console = Console()


def time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def compute_hit_mrr(
    ranked_docs: list[dict[str, Any]],
    answer_start: float,
    answer_end: float,
    ks: list[int],
    tolerance_s: float = 7.0,
) -> dict:
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
        metrics[f"Hit@{k}"] = 1.0 if (first_hit_rank and first_hit_rank <= k) else 0.0
    metrics["MRR"] = (1.0 / first_hit_rank) if first_hit_rank else 0.0
    return metrics


@dataclass
class RunSummary:
    episode_id: str
    emb_v: str
    method: str
    collection: str
    ks: list[int]
    tolerance_s: int
    rrf_k: float
    vec_k: int
    lex_k: int
    filters: dict
    metrics: dict
    p95_latency_ms: float


def evaluate_hybrid(
    episode_id: str,
    emb_v: str,
    method: str,
    qa_csv: Path,
    collection: str,
    params: HybridParams,
    filters: Filters,
) -> dict[str, Any]:
    # --- minimal change: embed questions and then filter to episode ---
    _ = load_qa_csv(qa_csv)  # validate schema (kept)
    qa_all = embed_questions(
        str(qa_csv),
        model_id="BAAI/bge-small-en-v1.5",
        use_bge_query_prefix=getattr(paths, "BGE_QUERY_PREFIX", False),  # optional: see below
    )
    qa_df = qa_all[qa_all["episode_id"] == episode_id].reset_index(drop=True)
    if qa_df.empty:
        raise ValueError(f"No QA rows for episode_id={episode_id}")

    # no longer need to steal a query vector from chunk embeddings
    # emb_df = load_embeddings_df(emb_v, episode_id)  # kept import above; call removed

    chunks_df = load_chunks_df(method, episode_id)

    # Build BM25 corpus from chunk texts (same episode)
    corpus_texts = chunks_df["text"].astype(str).tolist()
    corpus_ids = chunks_df["chunk_id"].astype(str).tolist()
    id_to_payload = {row["chunk_id"]: {**row} for _, row in chunks_df.iterrows()}

    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    latencies_ms: list[float] = []
    agg = {f"Hit@{k}": 0.0 for k in params.ks_report}
    agg["MRR"] = 0.0
    n = len(qa_df)

    for _, row in qa_df.iterrows():
        q = str(row["question"])

        # --- minimal change: use the true question embedding for the query vector ---
        q_vec = list(row["vector"])

        # 1) vector search (Qdrant)
        vec_docs, dt_ms = vector_search(
            cli,
            collection=collection,
            episode_id=episode_id,
            q_vector=q_vec,
            top_k=params.k_vec,
            guest=filters.guest,
            date_from=filters.date_from,
            date_to=filters.date_to,
        )
        latencies_ms.append(dt_ms)
        vec_ids = top_ids_by_score(vec_docs, params.k_vec)

        # 2) lexical (BM25) on the episode corpus
        lex_scores = score_bm25(q, corpus_texts)
        lex_rank = sorted(
            zip(corpus_ids, lex_scores, strict=False), key=lambda x: x[1], reverse=True
        )[: params.k_lex]
        lex_ids = [cid for cid, _ in lex_rank]

        # 3) fuse via RRF
        fused = rrf_fuse({"vec": vec_ids, "lex": lex_ids}, k=params.rrf_k)
        fused_ids = [pid for pid, _ in fused][: max(params.ks_report)]

        # 4) Build ranked_docs with payloads for scoring
        ranked_docs = [{"id": pid, "payload": id_to_payload.get(pid, {})} for pid in fused_ids]

        # 5) metrics
        m = compute_hit_mrr(
            ranked_docs,
            answer_start=float(row["answer_start_ts"]),
            answer_end=float(row["answer_end_ts"]),
            ks=list(params.ks_report),
            tolerance_s=float(params.tolerance_s),
        )
        for k in params.ks_report:
            agg[f"Hit@{k}"] += m[f"Hit@{k}"]
        agg["MRR"] += m["MRR"]

    metrics = {k: (v / n) for k, v in agg.items()}
    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    summary = RunSummary(
        episode_id=episode_id,
        emb_v=emb_v,
        method=method,
        collection=collection,
        ks=list(params.ks_report),
        tolerance_s=params.tolerance_s,
        rrf_k=params.rrf_k,
        vec_k=params.k_vec,
        lex_k=params.k_lex,
        filters=asdict(filters),
        metrics=metrics,
        p95_latency_ms=p95,
    )

    report = {"summary": asdict(summary)}
    return report


def write_report(report: dict[str, Any]) -> tuple[Path, Path]:
    paths.EVALS_DIR.mkdir(parents=True, exist_ok=True)
    paths.DOCS.mkdir(parents=True, exist_ok=True)

    out_json = paths.EVALS_DIR / "retrieval_hybrid.json"
    out_md = paths.DOCS / "0007-retrieval-hybrid.md"

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    lines = [
        f"# Retrieval hybrid — {s['emb_v']} (episode {s['episode_id']})",
        "",
        f"- collection: `{s['collection']}`",
        f"- method: `{s['method']}`",
        f"- ks: {s['ks']} tol_s={s['tolerance_s']} rrf_k={s['rrf_k']} vec_k={s['vec_k']} lex_k={s['lex_k']}",
        f"- filters: {s['filters']}",
        "",
        "## Metrics",
        "```json",
        json.dumps(s["metrics"], indent=2),
        "```",
        f"- p95_latency_ms: {s['p95_latency_ms']:.2f}",
    ]
    out_md.write_text("\n".join(lines))
    console.print(f"[green]Wrote[/green] {out_json} and {out_md}")
    return out_json, out_md
