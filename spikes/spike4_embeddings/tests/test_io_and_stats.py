from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from spike4_embeddings import cli as cli_mod
from spike4_embeddings.io import (
    load_chunks,
    read_existing_embeddings,
    write_embeddings,
)


# --- helpers ---------------------------------------------------------------
def _write_chunks_parquet(base: Path, method: str, episode_id: str) -> Path:
    df = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "episode_id": episode_id,
                "method": method,
                "text": "hello world",
                "n_tokens": 3,
                "duration_s": 1.5,
            },
            {
                "chunk_id": "c2",
                "episode_id": episode_id,
                "method": method,
                "text": "goodbye world",
                "n_tokens": 3,
                "duration_s": 2.0,
            },
        ]
    )
    out = base / method / f"episode_id={episode_id}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "part-00000.snappy.parquet"
    df.to_parquet(p, index=False)
    return p


def _embedding_rows(episode_id: str):
    return [
        {
            "chunk_id": "c1",
            "episode_id": episode_id,
            "method": "sentence_bound",
            "emb_v": "testv1",
            "dim": 3,
            "model_id": "dummy",
            "provider": "openai",
            "created_at": 0,
            "text_hash": "x",
            "tokens": 3,
            "vector": [0.1, 0.2, 0.3],
            "attempts": 1,
            "status": "ok",
        }
    ]


# --- tests ----------------------------------------------------------------
def test_io_roundtrip_and_cmd_stats(monkeypatch, tmp_path, capsys):
    """Write chunks, write embeddings, read back, and run cmd_stats."""
    # Patch global dirs to tmp
    monkeypatch.setattr("spike4_embeddings.paths.CHUNKS_DIR", tmp_path / "chunks", raising=False)
    monkeypatch.setattr("spike4_embeddings.paths.EMB_DIR", tmp_path / "embeddings", raising=False)
    # Also patch the copies imported inside io/cli modules
    monkeypatch.setattr("spike4_embeddings.io.CHUNKS_DIR", tmp_path / "chunks", raising=False)
    monkeypatch.setattr("spike4_embeddings.io.EMB_DIR", tmp_path / "embeddings", raising=False)

    method = "sentence_bound"
    episode_id = "ep1"
    emb_v = "testv1"

    # Write a chunks parquet
    chunks_path = _write_chunks_parquet(tmp_path / "chunks", method, episode_id)
    assert chunks_path.exists()

    # Load chunks
    chunks = load_chunks(method, episode_id, chunks_dir=tmp_path / "chunks")
    assert len(chunks) == 2
    assert set(chunks.columns) >= {"chunk_id", "episode_id", "text", "n_tokens"}

    # Write embeddings parquet (using required schema)
    out_path = write_embeddings(
        _embedding_rows(episode_id), emb_v, episode_id, tmp_path / "embeddings"
    )
    assert out_path.exists()

    # Read existing embeddings
    emb_df = read_existing_embeddings(emb_v, episode_id, tmp_path / "embeddings")
    assert emb_df is not None
    assert len(emb_df) == 1
    assert emb_df["status"].iloc[0] == "ok"

    # Run cmd_stats and check it prints something sensible
    args = SimpleNamespace(cmd="stats", emb_v=emb_v)
    cli_mod.cmd_stats(args)
    out = capsys.readouterr().out
    assert "Embeddings Stats" in out
    assert emb_v in out
