from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from typing import Any

# Phoenix tracing is disabled - use no-op tracer
# To enable: set PHOENIX_COLLECTOR_ENDPOINT environment variable
_tracer = None


def get_tracer():
    """Get a no-op tracer (Phoenix disabled)."""
    global _tracer
    if _tracer is not None:
        return _tracer

    # Always use no-op tracer (Phoenix disabled)
    from opentelemetry.trace import get_tracer as _gt

    _tracer = _gt(__name__)
    return _tracer


@contextmanager
def start_span(
    name: str, attrs: Mapping[str, Any] | None = None, kind: str | None = None
) -> Iterator[Any]:
    """
    Small helper to start a span and attach OpenInference-ish attributes.
    kind: "RETRIEVER" | "LLM" | "CHAIN" | "VALIDATOR"
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if kind:
            with suppress(Exception):
                # OpenInference semantic conventions
                span.set_attribute("openinference.span.kind", kind)
        if attrs:
            for k, v in attrs.items():
                with suppress(Exception):
                    span.set_attribute(k, v)
        yield span
