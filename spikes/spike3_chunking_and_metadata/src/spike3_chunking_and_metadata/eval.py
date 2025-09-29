import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .paths import CHUNKS_DIR, DOCS_DIR, EVALS_DIR
from .schema import validate_chunk_df_columns
from .utils import time_overlap, tokenize_for_bm25


def evaluate_methods(
    episode_id: str,
    qa_csv: Path,
    methods: list[str],
    k: int = 20,
    tol_s: int = 7,
    prefer_efficient: bool = False,
    chunks_dir: Path = CHUNKS_DIR,
):
    qa = pd.read_csv(qa_csv)
    qa = qa[qa["episode_id"] == episode_id].reset_index(drop=True)
    if qa.empty:
        raise ValueError(f"No QA rows for episode_id={episode_id}")

    results = {}
    for method in methods:
        chunk_path = chunks_dir / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing chunks parquet for {method}: {chunk_path}")
        chunks = pd.read_parquet(chunk_path)
        missing = validate_chunk_df_columns(chunks)
        if missing:
            raise ValueError(f"Chunks file missing columns {missing}: {chunk_path}")

        docs = chunks["text"].tolist()
        centers = (chunks["start_ts"] + chunks["end_ts"]) / 2.0
        bm25 = BM25Okapi(tokenize_for_bm25(docs))

        hit_at = {5: 0, 10: 0, 20: 0}
        rr, time_dists = [], []

        for _, row in qa.iterrows():
            q = str(row["question"])
            if (
                "optional_keywords" in qa.columns
                and isinstance(row.get("optional_keywords"), str)
                and row["optional_keywords"]
            ):
                q += " " + row["optional_keywords"]
            scores = bm25.get_scores(re.findall(r"[a-zA-Z0-9']+", q.lower()))
            order = list(np.argsort(scores)[::-1])[:k]
            first_hit_rank = None
            ans_mid = 0.5 * (row["answer_start_ts"] + row["answer_end_ts"])
            for rank_idx, doc_idx in enumerate(order, start=1):
                c_start, c_end = chunks["start_ts"].iloc[doc_idx], chunks["end_ts"].iloc[doc_idx]
                overlap = time_overlap(
                    (c_start, c_end), (row["answer_start_ts"] - tol_s, row["answer_end_ts"] + tol_s)
                )
                if overlap >= 1.0:
                    first_hit_rank = rank_idx
                    time_dists.append(abs(float(centers.iloc[doc_idx]) - float(ans_mid)))
                    break
            rr.append(1.0 / first_hit_rank if first_hit_rank else 0.0)
            if first_hit_rank:
                if first_hit_rank <= 5:
                    hit_at[5] += 1
                if first_hit_rank <= 10:
                    hit_at[10] += 1
                if first_hit_rank <= 20:
                    hit_at[20] += 1

        n = len(qa)
        results[method] = {
            "Hit@5": hit_at[5] / n,
            "Hit@10": hit_at[10] / n,
            "Hit@20": hit_at[20] / n,
            "MRR": float(np.mean(rr)),
            "AvgTimeDistanceSec": float(np.mean(time_dists)) if time_dists else None,
            "AvgTokens": float(chunks["n_tokens"].mean()),
            "AvgDurationSec": float(chunks["duration_s"].mean()),
        }

    eps = 1e-6

    def _cmp_key(m):
        r = results[m]
        return (
            r["Hit@10"],
            r["MRR"],
            -r["AvgTokens"],  # smaller tokens preferred
            -r["AvgDurationSec"],  # smaller duration preferred
        )

    # If close within epsilon, normalize to exact ties for fairness
    def _roundish(x):
        return round(x / eps)  # avoid float wiggles

    def _cmp_key_stable(m):
        r = results[m]
        return (
            _roundish(r["Hit@10"]),
            _roundish(r["MRR"]),
            -_roundish(r["AvgTokens"]),
            -_roundish(r["AvgDurationSec"]),
        )

    if prefer_efficient:
        winner = max(methods, key=_cmp_key_stable)
    else:
        winner = max(methods, key=lambda m: (results[m]["Hit@10"], results[m]["MRR"]))
    return {
        "episode_id": episode_id,
        "k": k,
        "tolerance_s": tol_s,
        "methods": results,
        "winner": winner,
    }


def write_eval_report(
    report: dict, evals_dir: Path = EVALS_DIR, docs_dir: Path = DOCS_DIR
) -> tuple[Path, Path]:
    """Persist the eval results to JSON and a short decision markdown."""
    evals_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    report_path = evals_dir / "chunk_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = docs_dir / "0003-chunking.md"
    md_lines = [
        f"# Chunking decision — episode `{report.get('episode_id', '')}`",
        f"Winner (Hit@10 → tie MRR): **{report['winner']}**",
        "",
        "```json",
        json.dumps(report["methods"], indent=2),
        "```",
        f"_k={report['k']}, tolerance_s={report['tolerance_s']}_",
    ]
    md_path.write_text("\n".join(md_lines))
    return report_path, md_path
