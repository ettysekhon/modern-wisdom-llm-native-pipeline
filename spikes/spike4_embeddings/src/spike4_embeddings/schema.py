REQUIRED_EMBED_COLS = [
    "chunk_id",
    "episode_id",
    "method",
    "emb_v",
    "dim",
    "model_id",
    "provider",
    "created_at",
    "text_hash",
    "tokens",
    "vector",  # list<float32> (or None on error)
    "attempts",
    "status",  # "ok" or "error:<type>"
]


def validate_embed_df_columns(df):
    missing = [c for c in REQUIRED_EMBED_COLS if c not in df.columns]
    return missing
