from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import paths


@dataclass(frozen=True)
class Job:
    kind: str  # "baseline" | "hybrid" | "contract"
    episode_id: str
    emb_v: str
    method: str
    extra: dict[str, Any]


def _uv_run(args: list[str]) -> None:
    # Fail fast; stdout/err stream to console
    proc = subprocess.run(["uv", "run", *args], check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)} (rc={proc.returncode})")


def run_job(job: Job) -> Path:
    if job.kind == "baseline":
        _uv_run(
            [
                "run-spike6",
                "eval",
                "--episode-id",
                job.episode_id,
                "--emb-v",
                job.emb_v,
                "--qa-csv",
                job.extra.get("qa_csv", "data/qa/labels.csv"),
                "--ks",
                job.extra.get("ks", "5,10,20"),
                "--tolerance-s",
                str(job.extra.get("tolerance_s", 7)),
            ]
        )
        return paths.EVALS_DIR / "retrieval_baseline.json"

    if job.kind == "hybrid":
        _uv_run(
            [
                "run-spike7",
                "eval",
                "--episode-id",
                job.episode_id,
                "--emb-v",
                job.emb_v,
                "--method",
                job.method,
                "--qa-csv",
                job.extra.get("qa_csv", "data/qa/labels.csv"),
                "--ks",
                job.extra.get("ks", "5,10,20"),
                "--tolerance-s",
                str(job.extra.get("tolerance_s", 7)),
                "--k-vec",
                str(job.extra.get("k_vec", 20)),
                "--k-lex",
                str(job.extra.get("k_lex", 200)),
                "--rrf-k",
                str(job.extra.get("rrf_k", 60)),
            ]
        )
        return paths.EVALS_DIR / "retrieval_hybrid.json"

    if job.kind == "contract":
        # Run Spike 8 ‘run’ subcommand (answer flow)
        _uv_run(
            [
                "run-spike8",
                "run",
                "--episode-id",
                job.episode_id,
                "--emb-v",
                job.emb_v,
                "--method",
                job.method,
                "--question",
                job.extra.get("question", "Short factual question."),
                "--top-k",
                str(job.extra.get("top_k", 8)),
                "--llm-provider",
                job.extra.get("llm_provider", "mock"),
                "--llm-model-id",
                job.extra.get("llm_model_id", "gpt-4o-mini"),
            ]
        )
        return paths.CONTRACTS_DIR / "sample_responses" / "sample_answer.json"

    raise ValueError(f"Unknown job kind: {job.kind}")


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _pick_latency(summary: dict[str, Any]) -> float | None:
    # tolerate different writers
    for k in ("p95_latency_ms", "p95_ms", "latency_p95_ms", "latency_p95"):
        if k in summary:
            return _as_float(summary.get(k))
    return None


def normalize_result(p: Path) -> dict[str, Any]:
    obj = json.loads(p.read_text() or "{}")
    summary = obj.get("summary", obj)

    if p.name.startswith("retrieval_baseline"):
        return {
            "type": "baseline",
            **(summary.get("metrics") or {}),
            "p95_latency_ms": _pick_latency(summary),
            "path": str(p.resolve()),
        }

    if p.name.startswith("retrieval_hybrid"):
        return {
            "type": "hybrid",
            **(summary.get("metrics") or {}),
            "p95_latency_ms": _pick_latency(summary),
            "path": str(p.resolve()),
        }

    # contract envelope (Spike 8)
    return {
        "type": "contract",
        "has_answer": bool(obj.get("answer", "")),
        "retrieved": int(len(obj.get("retrieved", []) or [])),
        "path": str(p.resolve()),
    }


def run_suite(cfg: dict[str, Any]) -> Path:
    jobs_cfg = cfg.get("jobs", [])
    outputs: list[dict[str, Any]] = []

    for j in jobs_cfg:
        job = Job(
            kind=j["kind"],
            episode_id=j["episode_id"],
            emb_v=j["emb_v"],
            method=j.get("method", "sentence_bound"),
            extra=j.get("extra", {}),
        )
        try:
            out_path = run_job(job)
            outputs.append(normalize_result(out_path))
        except Exception as e:
            # Capture failure but do not abort the suite
            outputs.append(
                {
                    "type": job.kind,
                    "episode_id": job.episode_id,
                    "emb_v": job.emb_v,
                    "method": job.method,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    # ensure dir and write report
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "suite_name": cfg.get("name", "default-suite"),
        "outputs": outputs,
        "thresholds": cfg.get("thresholds", {}),
    }
    out = paths.REPORTS_DIR / "suite_report.json"
    out.write_text(json.dumps(report, indent=2))
    return out
