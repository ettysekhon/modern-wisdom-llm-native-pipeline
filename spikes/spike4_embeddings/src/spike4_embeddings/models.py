import os


# --- OpenAI provider (optional) ---
def _get_openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY env var is required for OpenAI provider.")
    return OpenAI(api_key=key)


def openai_embed(texts: list[str], model_id: str) -> list[list[float]]:
    client = _get_openai_client()
    # OpenAI embeddings are deterministic given the same model_id & input
    resp = client.embeddings.create(model=model_id, input=texts)
    return [d.embedding for d in resp.data]


# --- OSS placeholder (wire later if needed) ---
def oss_embed(texts: list[str], model_id: str) -> list[list[float]]:
    raise NotImplementedError("OSS embedding provider not implemented yet.")
