from dataclasses import dataclass

# Alias name we point clients to
LIVE_ALIAS = "mw_chunks_live"


@dataclass
class VectorSpec:
    size: int
    distance: str = "COSINE"  # "COSINE" | "DOT" | "EUCLID"


# Build a collection name per embedding version (blue/green by version)
def collection_name_for_emb_v(emb_v: str) -> str:
    # keep lowercase & safe
    safe = emb_v.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
    return f"mw_chunks_{safe}"
