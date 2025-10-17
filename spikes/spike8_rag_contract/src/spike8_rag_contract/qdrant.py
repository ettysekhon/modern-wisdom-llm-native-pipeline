from __future__ import annotations

import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from . import paths
from .tracing import start_span


def client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    from qdrant_client import QdrantClient

    return QdrantClient(url=url or paths.QDRANT_URL, api_key=api_key or paths.QDRANT_API_KEY)


def _episode_filter(episode_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="episode_id", match=MatchValue(value=episode_id))])


def vector_search(
    cli: QdrantClient,
    collection: str,
    q_vector: list[float],
    top_k: int = 20,
    episode_id: str | None = None,
) -> tuple[list[Any], float]:
    with start_span(
        "retrieve.qdrant",
        kind="RETRIEVER",
        attrs={
            "retrieval.top_k": int(top_k),
            "retrieval.index_version": collection,
            "retrieval.episode_id": episode_id,
        },
    ) as span:
        t0 = time.perf_counter()

        flt = None
        if episode_id:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue

            flt = Filter(
                must=[FieldCondition(key="episode_id", match=MatchValue(value=episode_id))]
            )

        res = cli.search(
            collection_name=collection,
            query_vector=q_vector,
            limit=top_k,
            query_filter=flt,  # None → no episode filter (corpus-wide)
            with_payload=True,
            with_vectors=False,
            score_threshold=None,
        )
        rt_ms = (time.perf_counter() - t0) * 1000.0
        # annotate results (scores + ids, trimmed)
        try:
            span.set_attribute("retrieval.latency_ms", float(rt_ms))
            for i, p in enumerate(res[: min(5, len(res))]):
                prefix = f"retrieval.documents.{i}.document"
                span.set_attribute(f"{prefix}.id", str(getattr(p, "id", "")))
                span.set_attribute(f"{prefix}.score", float(getattr(p, "score", 0.0)))
        except Exception:
            pass
        return res, rt_ms
