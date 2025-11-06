from __future__ import annotations

from fastembed import TextEmbedding


def embed_questions_bge(
    questions: list[str],
    model_id: str = "BAAI/bge-small-en-v1.5",
    use_query_prefix: bool = True,
) -> list[list[float]]:
    texts = [f"query: {q}" if use_query_prefix else q for q in questions]
    encoder = TextEmbedding(model_id)
    return [list(vec) for vec in encoder.embed(texts)]
