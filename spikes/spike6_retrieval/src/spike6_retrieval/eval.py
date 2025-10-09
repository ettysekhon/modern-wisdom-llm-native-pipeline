from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from . import paths
from .io import load_embeddings_df, load_qa_csv, select_provider_from_embeddings
from .metrics import compute_hit_mrr, p95
from .qdrant import client, search

console = Console()

# --- query embedder (FastEmbed) ---


def embed_queries_fastembed(texts: list[str], model_id: str) -> list[list[float]]:
    from fastembed import TextEmbedding  # local import so pkg remains light until used

    model = TextEmbedding(model_name=model_id)
    # fastembed returns an iterator of np arrays; convert to lists
    return [list(vec) for vec in model.embed(texts)]


@dataclass
class RunSummary:
    episode_id: str
    emb_v: str
    collection: str
    ks: list[int]
    tolerance_s: int
    top_k_search: int
    provider: str
    model_id: str
    n_queries: int
    metrics: dict[str, float]


def evaluate_episode(
    episode_id: str,
    emb_v: str,
    qa_csv: Path,
    ks: list[int] | None = None,
    tolerance_s: int = 7,
    collection: str | None = None,
) -> dict[str, Any]:
    if ks is None:
        ks = [5, 10, 20]
    # Load QA rows for this episode_id
    qa_df = load_qa_csv(qa_csv)
    qa_df = qa_df[qa_df["episode_id"] == episode_id].reset_index(drop=True)
    if qa_df.empty:
        raise ValueError(f"No QA rows for episode_id={episode_id}")

    # read embeddings parquet to infer provider/model and for sanity
    emb_df = load_embeddings_df(emb_v, episode_id)
    provider, model_id = select_provider_from_embeddings(emb_df)

    # Where to search: live alias by default
    coll = collection or paths.LIVE_ALIAS

    # Embed queries
    q_texts = qa_df["question"].astype(str).tolist()
    if provider.lower() in ("fastembed", "oss", "fe"):
        q_vectors = embed_queries_fastembed(q_texts, model_id)
    else:
        raise NotImplementedError(f"Provider '{provider}' not supported in Spike 6 baseline")

    # Qdrant client
    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    # Evaluate
    latencies: list[float] = []
    total = len(qa_df)
    agg = {f"Hit@{k}": 0.0 for k in ks}
    agg["MRR"] = 0.0

    for idx, (_, row) in enumerate(qa_df.iterrows()):
        docs, dt_ms = search(cli, coll, q_vectors[idx], limit=max(ks))
        latencies.append(dt_ms)

        m = compute_hit_mrr(
            ranked_docs=docs,
            answer_start=float(row["answer_start_ts"]),
            answer_end=float(row["answer_end_ts"]),
            ks=ks,
            tolerance_s=float(tolerance_s),
        )
        for k in ks:
            agg[f"Hit@{k}"] += m[f"Hit@{k}"]
        agg["MRR"] += m["MRR"]

    # averages
    metrics = {k: (v / total) for k, v in agg.items()}
    metrics["p95_latency_ms"] = p95(latencies)
    metrics["n"] = float(total)

    result = RunSummary(
        episode_id=episode_id,
        emb_v=emb_v,
        collection=coll,
        ks=ks,
        tolerance_s=tolerance_s,
        top_k_search=max(ks),
        provider=provider,
        model_id=model_id,
        n_queries=total,
        metrics=metrics,
    )
    return {
        "summary": asdict(result),
        "per_query_latency_ms": latencies,  # optional raw
    }


def write_report(report: dict[str, Any]) -> tuple[Path, Path]:
    paths.EVALS_DIR.mkdir(parents=True, exist_ok=True)
    paths.DOCS.mkdir(parents=True, exist_ok=True)

    out_json = paths.EVALS_DIR / "retrieval_baseline.json"
    out_md = paths.DOCS / "0006-retrieval-baseline.md"

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    lines = [
        f"# Retrieval baseline — {s['emb_v']} (episode {s['episode_id']})",
        "",
        f"- collection: `{s['collection']}`",
        f"- provider/model: `{s['provider']}` / `{s['model_id']}`",
        f"- ks: {s['ks']}, tolerance_s={s['tolerance_s']}, top_k_search={s['top_k_search']}",
        "",
        "## Metrics",
        "```json",
        json.dumps(s["metrics"], indent=2),
        "```",
    ]
    out_md.write_text("\n".join(lines))
    console.print(f"[green]Wrote[/green] {out_json} and {out_md}")
    return out_json, out_md
