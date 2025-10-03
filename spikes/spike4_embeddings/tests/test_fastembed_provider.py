import importlib.util

import pytest


def _has_fastembed():
    return importlib.util.find_spec("fastembed") is not None


@pytest.mark.skipif(not _has_fastembed(), reason="fastembed not installed")
def test_fastembed_vectors_shape():
    from spike4_embeddings.models_fastembed import fastembed_dim, fastembed_embed

    model = "BAAI/bge-small-en-v1.5"
    vecs = fastembed_embed(["hello world", "goodbye"], model_id=model)
    assert len(vecs) == 2
    d = fastembed_dim(model)
    assert d == len(vecs[0]) and d > 0
