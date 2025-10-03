from types import SimpleNamespace

import pandas as pd
from spike4_embeddings import cli as cli_mod


def test_cmd_check(monkeypatch, tmp_path, capsys):
    # Patch dirs
    monkeypatch.setattr("spike4_embeddings.paths.CHUNKS_DIR", tmp_path / "chunks", raising=False)
    monkeypatch.setattr("spike4_embeddings.io.CHUNKS_DIR", tmp_path / "chunks", raising=False)

    method = "sentence_bound"
    episode_id = "ep1"

    out = tmp_path / "chunks" / method / f"episode_id={episode_id}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "episode_id": episode_id,
                "method": method,
                "text": "a",
                "n_tokens": 5,
                "duration_s": 10.0,
            },
            {
                "chunk_id": "c2",
                "episode_id": episode_id,
                "method": method,
                "text": "b",
                "n_tokens": 5,
                "duration_s": 20.0,
            },
        ]
    )
    (out / "part-00000.snappy.parquet").write_bytes(
        b""
    )  # ensure file exists even if to_parquet fails
    df.to_parquet(out / "part-00000.snappy.parquet", index=False)

    # Build args and call
    args = SimpleNamespace(cmd="check", episode_id=episode_id, method=method)
    cli_mod.cmd_check(args)
    out_text = capsys.readouterr().out

    assert "Spike 4 Check" in out_text
    assert "Chunks" in out_text
