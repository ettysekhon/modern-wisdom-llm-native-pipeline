import pandas as pd
from fastembed import TextEmbedding


def embed_questions(
    qa_csv: str,
    model_id: str = "BAAI/bge-small-en-v1.5",
    use_bge_query_prefix: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(qa_csv)
    texts = df["question"].astype(str).tolist()

    if use_bge_query_prefix:
        texts = [f"query: {t}" for t in texts]

    encoder = TextEmbedding(model_id)
    vecs = list(encoder.embed(texts))
    df["vector"] = vecs
    return df
