# spike5_qdrant/qdrant.py
from __future__ import annotations

from qdrant_client import QdrantClient, models


def alias_set_live(cli: QdrantClient, collection: str, alias: str = "mw_chunks_live") -> None:
    """
    Atomically point the alias to `collection`.
    If alias exists and points elsewhere, delete + create in one operation.
    """
    # Does alias already exist?
    existing = {a.alias_name: a.collection_name for a in cli.get_aliases().aliases}
    ops: list[models.AliasOperations] = []

    if alias in existing:
        # Remove the old mapping first (same alias name)
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))

    # Create the new mapping
    ops.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )

    cli.update_collection_aliases(change_aliases_operations=ops)
