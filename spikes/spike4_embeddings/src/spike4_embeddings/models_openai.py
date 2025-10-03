def openai_dim_for_model(model_id: str) -> int:
    """Return embedding dimension for known OpenAI models."""
    mapping = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    if model_id not in mapping:
        raise ValueError(f"Unknown OpenAI embedding model: {model_id}")
    return mapping[model_id]
