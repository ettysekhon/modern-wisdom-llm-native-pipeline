from __future__ import annotations

from typing import Any

from .tracing import start_span


def log_feedback(trace_id: str, rating: float | None = None, comment: str | None = None) -> None:
    # Attach feedback fields; Phoenix will show it as a span in the same trace if you call inside the run
    attrs: dict[str, Any] = {"feedback.trace_id": trace_id}
    if rating is not None:
        attrs["feedback.rating"] = float(rating)
    if comment:
        attrs["feedback.comment"] = str(comment)
    with start_span("feedback", kind="CHAIN", attrs=attrs):
        pass
