from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from .schema import LIVE_ALIAS, VectorSpec


def client(url: str, api_key: str | None = None) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(cli: QdrantClient, name: str, spec: VectorSpec) -> None:
    # Prefer collection_exists if available, else fallback to get_collection
    try:
        exists = bool(cli.collection_exists(name))  # qdrant-client >=1.5
    except AttributeError:
        try:
            cli.get_collection(name)
            exists = True
        except Exception:
            exists = False

    if not exists:
        params = models.VectorParams(
            size=spec.size,
            distance=getattr(models.Distance, spec.distance),
        )
        # Try the modern create_collection; fall back to recreate_collection
        try:
            cli.create_collection(collection_name=name, vectors_config=params)
        except AttributeError:
            cli.recreate_collection(collection_name=name, vectors_config=params)


def upsert_points(cli: QdrantClient, collection: str, rows: Iterable[dict[str, Any]]) -> None:
    points: list[PointStruct] = []
    for r in rows:
        vec = r.get("vector")
        if vec is None:
            continue
        pid = r["chunk_id"]
        payload = dict(r)
        payload.pop("vector", None)  # vector is stored separately; payload keeps metadata
        points.append(PointStruct(id=pid, vector=vec, payload=payload))
    if points:
        cli.upsert(collection_name=collection, points=points, wait=True)


def alias_set_live(cli: QdrantClient, collection: str, alias: str = LIVE_ALIAS) -> None:
    # v1.15.x API reference uses update_collection_aliases with Create/Delete ops
    # First delete alias (if exists), then create it → atomic “flip”.
    ops: list[models.AliasOperations] = [
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        ),
    ]
    cli.update_collection_aliases(change_aliases_operations=ops)


def search_vector(
    cli: QdrantClient,
    collection: str,
    vector: list[float],
    top_k: int = 5,
    where: dict[str, Any] | None = None,
):
    qf = None
    if where:
        qf = Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
        )
    return cli.search(collection_name=collection, query_vector=vector, limit=top_k, query_filter=qf)
