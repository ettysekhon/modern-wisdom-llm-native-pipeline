from __future__ import annotations

from time import perf_counter

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from . import paths


def client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or paths.QDRANT_URL, api_key=api_key or paths.QDRANT_API_KEY)


def _build_filter(
    episode_id: str, guest: str | None, date_from: str | None, date_to: str | None
) -> qm.Filter:
    must: list[qm.Condition] = [
        qm.FieldCondition(key="episode_id", match=qm.MatchValue(value=episode_id))
    ]
    if guest:
        must.append(qm.FieldCondition(key="guest", match=qm.MatchValue(value=guest)))
    if date_from:
        must.append(qm.FieldCondition(key="publish_date", match=qm.MatchValue(value=date_from)))
    if date_to:
        must.append(qm.FieldCondition(key="publish_date", match=qm.MatchValue(value=date_to)))
    return qm.Filter(must=must)


def vector_search(
    cli: QdrantClient,
    collection: str,
    episode_id: str,
    q_vector: list[float],
    top_k: int,
    guest: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[dict], float]:
    qf = _build_filter(episode_id, guest, date_from, date_to)
    t0 = perf_counter()
    res = cli.search(
        collection_name=collection,
        query_vector=q_vector,
        query_filter=qf,
        limit=top_k,
        with_payload=True,
    )
    dt_ms = (perf_counter() - t0) * 1000.0
    docs = [{"id": str(p.id), "score": float(p.score), "payload": (p.payload or {})} for p in res]
    return docs, dt_ms
