from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_batch(rows: list[dict]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    n_total = len(df)
    n_ok = int((df["status"] == "ok").sum()) if n_total else 0
    dim = int(df["dim"].dropna().unique()[0]) if n_ok else None
    avg_attempts = float(df["attempts"].mean()) if n_total else 0.0
    return {
        "rows": n_total,
        "rows_ok": n_ok,
        "dim": dim,
        "avg_attempts": avg_attempts,
    }


def estimate_openai_cost(tokens: int, price_per_1k: float) -> float:
    return (tokens / 1000.0) * price_per_1k
