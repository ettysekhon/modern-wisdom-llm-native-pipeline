import pandas as pd
import pytest
from spike4_embeddings.embed import embed_chunks_df, filter_idempotent


@pytest.fixture
def sample_chunks():
    return pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "episode_id": "ep1",
                "method": "fixed",
                "text": "hello world",
                "n_tokens": 3,
            },
            {
                "chunk_id": "c2",
                "episode_id": "ep1",
                "method": "fixed",
                "text": "goodbye world",
                "n_tokens": 3,
            },
        ]
    )


@pytest.fixture
def existing_embeddings():
    return pd.DataFrame(
        [{"chunk_id": "c1", "emb_v": "test", "vector": [0.1, 0.2, 0.3], "status": "ok"}]
    )


# --- Tests ---
def test_filter_idempotent_filters_existing(sample_chunks, existing_embeddings):
    """Ensure already embedded chunk_ids are skipped."""
    result = filter_idempotent(sample_chunks, existing_embeddings)
    assert "c1" not in list(result["chunk_id"])
    assert "c2" in list(result["chunk_id"])


def test_filter_idempotent_no_existing(sample_chunks):
    """If no existing embeddings, all chunks should be returned."""
    result = filter_idempotent(sample_chunks, None)
    assert set(result["chunk_id"]) == {"c1", "c2"}


def test_embed_chunks_df_success(monkeypatch, sample_chunks):
    """Embedding should produce correct schema when provider works."""

    def fake_provider(texts, model_id):
        # return vectors same length as text
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr("spike4_embeddings.embed.select_provider", lambda _: fake_provider)

    rows = embed_chunks_df(
        sample_chunks,
        emb_v="test_v1",
        provider="fake",
        model_id="dummy",
        batch_size=1,
    )

    assert all(r["status"] == "ok" for r in rows)
    assert all(isinstance(r["vector"], list) for r in rows)
    assert rows[0]["emb_v"] == "test_v1"


def test_embed_chunks_df_failure(monkeypatch, sample_chunks):
    """When provider throws, rows should be marked as error."""

    def bad_provider(texts, model_id):
        raise RuntimeError("fail")

    monkeypatch.setattr("spike4_embeddings.embed.select_provider", lambda _: bad_provider)

    rows = embed_chunks_df(
        sample_chunks,
        emb_v="test_v1",
        provider="bad",
        model_id="dummy",
        batch_size=1,
        retries=0,  # force immediate failure
    )

    assert all(r["status"].startswith("error:") for r in rows)
    assert rows[0]["vector"] is None
