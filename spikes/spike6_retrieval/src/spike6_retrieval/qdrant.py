from __future__ import annotations

from time import perf_counter
from typing import Any

from qdrant_client import QdrantClient

from . import paths


def client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or paths.QDRANT_URL, api_key=api_key or paths.QDRANT_API_KEY)


def search(
    cli: QdrantClient,
    collection: str,
    query_vector: list[float],
    limit: int,
) -> tuple[list[dict[str, Any]], float]:
    t0 = perf_counter()
    res = cli.search(collection_name=collection, query_vector=query_vector, limit=limit)
    dt_ms = (perf_counter() - t0) * 1000.0
    docs = [{"id": str(p.id), "score": float(p.score), "payload": (p.payload or {})} for p in res]
    return docs, dt_ms
