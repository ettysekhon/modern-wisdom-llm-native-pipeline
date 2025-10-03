import pytest


def test_openai_dim_known_models():
    from spike4_embeddings.models_openai import openai_dim_for_model

    assert openai_dim_for_model("text-embedding-3-small") == 1536
    assert openai_dim_for_model("text-embedding-3-large") == 3072


def test_openai_dim_unknown_model():
    from spike4_embeddings.models_openai import openai_dim_for_model

    with pytest.raises(ValueError):
        openai_dim_for_model("unknown-model")
