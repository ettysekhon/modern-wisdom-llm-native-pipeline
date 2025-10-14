import os


def get_openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY env var is required for OpenAI provider.")
    return OpenAI(api_key=key)
