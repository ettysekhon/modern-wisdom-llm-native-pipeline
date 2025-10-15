from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from typing import Any

# Phoenix registers an OTEL tracer provider + exporter automatically
try:  # keep import light during unit tests
    from phoenix.otel import register  # type: ignore
except Exception:  # pragma: no cover
    register = None

_tracer = None


def get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    project_name = os.getenv("PHOENIX_PROJECT_NAME", "mw-rag")
    if register is None:
        # Fallback no-op tracer if phoenix isn't installed
        from opentelemetry.trace import get_tracer as _gt

        _tracer = _gt(__name__)
        return _tracer
    tp = register(
        protocol="http/protobuf",
        project_name=project_name,
        use_batch=True,
    )
    _tracer = tp.get_tracer("mw.rag")
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
