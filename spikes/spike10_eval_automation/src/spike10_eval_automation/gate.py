from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def gate(report_path: Path) -> int:
    # Fail fast on missing/empty report
    txt = (report_path.read_text() or "").strip()
    if not txt:
        print("[gate] FAIL  report is empty")
        return 1

    try:
        rep = json.loads(txt)
    except Exception as e:
        print(f"[gate] FAIL  cannot parse report: {e}")
        return 1

    th = rep.get("thresholds", {})
    # Global defaults
    min_hit10_global = float(th.get("min_hit_at_10", 0.5))
    max_p95 = float(th.get("max_p95_ms", 250.0))
    require_answer = bool(th.get("require_contract_answer", True))

    # Optional per-type overrides
    min_hit10_baseline = float(th.get("min_hit_at_10_baseline", min_hit10_global))
    min_hit10_hybrid = float(th.get("min_hit_at_10_hybrid", min_hit10_global))

    hit10_ok = True
    p95_ok = True
    contract_ok = True

    for o in rep.get("outputs", []):
        t = o.get("type", "")
        if t in ("baseline", "hybrid"):
            hit10 = _as_float(o.get("Hit@10", 1.0), 1.0)
            p95 = o.get("p95_latency_ms", 0.0)
            p95 = _as_float(p95, 0.0)  # tolerate null/NaN

            # choose threshold by type
            min_hit10 = min_hit10_baseline if t == "baseline" else min_hit10_hybrid

            if hit10 < min_hit10:
                hit10_ok = False
            if p95 > max_p95:
                p95_ok = False

        elif t == "contract":
            if require_answer and not bool(o.get("has_answer", False)):
                contract_ok = False

    ok = hit10_ok and p95_ok and contract_ok
    if not ok:
        print(f"[gate] FAIL  hit10_ok={hit10_ok} p95_ok={p95_ok} contract_ok={contract_ok}")
        return 1
    print(f"[gate] PASS  hit10_ok={hit10_ok} p95_ok={p95_ok} contract_ok={contract_ok}")
    return 0
