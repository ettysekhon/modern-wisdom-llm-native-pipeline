from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from .paths import DOCS_DIR


def _count_qa_for_episode(qa_csv: Path | None, episode_id: str) -> int | None:
    if not qa_csv or not qa_csv.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(qa_csv)
        return (
            int((df["episode_id"] == episode_id).sum())
            if "episode_id" in df.columns
            else int(len(df))
        )
    except Exception:
        return None


def build_decision_md(
    report: dict[str, Any],
    owner: str = "",
    qa_csv: Path | None = None,
    context_window: int | None = None,
) -> str:
    """Return a complete Markdown decision note with a *closed* fenced JSON block."""
    date_str = dt.date.today().isoformat()
    if not owner:
        owner = os.popen("git config user.name").read().strip() or "unknown"

    qa_count = _count_qa_for_episode(qa_csv, report.get("episode_id", ""))

    # Compact metrics block (include core fields + size hints)
    metrics = {
        m: {
            "Hit@10": float(v.get("Hit@10", 0) or 0),
            "MRR": float(v.get("MRR", 0) or 0),
            "AvgTimeDistanceSec": float(v.get("AvgTimeDistanceSec", 0) or 0),
            "AvgTokens": float(v.get("AvgTokens", 0) or 0),
            "AvgDurationSec": float(v.get("AvgDurationSec", 0) or 0),
        }
        for m, v in report.get("methods", {}).items()
    }

    winner = report.get("winner", "—")
    chosen_line = (
        "sentence_bound, 700 tokens, 100 overlap"
        if winner == "sentence_bound"
        else f"{winner} (see metrics)"
    )

    ctx_text = f"{context_window} tokens" if context_window else "8k–16k tokens (typical)"
    qa_text = (
        f"{qa_count} questions labeled with start/end timestamps"
        if qa_count is not None
        else "tiny set (5–10) labeled with timestamps"
    )

    # NOTE: Properly close the fenced code block with ```
    md = (
        f"# Decision Record — Chunking & Metadata (Spike 3)\n\n"
        f"Date: {date_str}\n"
        f"Owner: {owner}\n\n"
        f"## Context\n\n"
        f"- Transcript segments: avg duration = —, avg tokens/segment = —\n"
        f"- Episode metadata available in DuckDB: title, guest, publish_date, episode_number, headline, duration\n"
        f"- Q/A set: {qa_text}\n"
        f"- Constraint: downstream LLM context window = {ctx_text}\n\n"
        f"## Options Considered\n\n"
        f"- V0 — Fixed tokens (700 size, 100 overlap)\n"
        f"- V1 — Sentence-bounded (700 size, 100 overlap)\n"
        f"- V2 — Time-windowed (≈180–200s window, 30s overlap)\n\n"
        f"## Evaluation Setup\n\n"
        f"- Scorer: BM25 (rank_bm25)\n"
        f"- Top-K: {report.get('k', 20)}\n"
        f"- Hit definition: retrieved chunk overlaps labeled answer by ≥1s (±{report.get('tolerance_s', 7)}s tolerance)\n"
        f"- Metrics: Hit@5/10/20, MRR, coverage (via overlap), avg time distance\n\n"
        f"## Results\n\n"
        "```json\n"
        f"{json.dumps(metrics, indent=2)}\n"
        "```\n\n"
        f"## Decision\n\n"
        f"Chosen: **{chosen_line}**  \n"
        f"Rationale: Winner selected on Hit@10 (primary), then MRR (tie-breaker). "
        f"If equal, prefer smaller AvgTokens and AvgDuration to minimize context cost.\n\n"
        f"## Consequences\n\n"
        f"- Retrieval will use **{winner}** chunks.\n"
        f"- Metadata enrichment includes guest, publish_date, episode_number, title, headline, duration.\n"
        f"- Index size and retrieval latency expected to remain within budget; shorter chunks reduce prompt cost.\n"
        f"- Re-chunking may be required if ASR quality or the embedding model/context window changes significantly.\n"
    )
    return md


def write_decision_md(
    report: dict[str, Any],
    out_path: Path = DOCS_DIR / "0003-chunking.md",
    owner: str = "",
    qa_csv: Path | None = None,
    context_window: int | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = build_decision_md(report, owner=owner, qa_csv=qa_csv, context_window=context_window)
    out_path.write_text(md, encoding="utf-8")
    return out_path
