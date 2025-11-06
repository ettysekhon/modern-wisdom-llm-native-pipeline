from __future__ import annotations

from time import perf_counter
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


def build_filter(episode_id: str | None = None) -> Filter | None:
    if not episode_id:
        return None
    return Filter(must=[FieldCondition(key="episode_id", match=MatchValue(value=episode_id))])


def retrieve(
    cli: QdrantClient,
    collection: str,
    query_vector: list[float],
    top_k: int = 8,
    episode_id: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    f = build_filter(episode_id)
    t0 = perf_counter()
    hits = cli.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=f,
        with_payload=True,
    )
    dt_ms = (perf_counter() - t0) * 1000.0
    docs: list[dict[str, Any]] = []
    for h in hits:
        pld = getattr(h, "payload", {}) or {}
        docs.append(
            {
                "chunk_id": str(h.id),
                "start_ts": float(pld.get("start_ts", 0.0)),
                "end_ts": float(pld.get("end_ts", 0.0)),
                "score": float(h.score),
                "text": pld.get("text", ""),
            }
        )
    return docs, dt_ms


def embed_question_fastembed(question: str, model_id: str) -> list[float]:
    model = TextEmbedding(model_name=model_id)
    # fastembed returns generator of (vector, metadata). Use .embed with list
    vecs = list(model.embed([question]))
    return list(vecs[0])
