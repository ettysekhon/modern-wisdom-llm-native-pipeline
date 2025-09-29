import pandas as pd
from conftest import write_valid_chunk_parquet
from spike3_chunking_and_metadata.eval import evaluate_methods


def test_evaluate_methods(tmp_path):
    method = "fixed"
    write_valid_chunk_parquet(tmp_path, episode_id="ep1", method=method)

    qa_csv = tmp_path / "qa.csv"
    pd.DataFrame(
        [{"question": "Hello?", "episode_id": "ep1", "answer_start_ts": 0.0, "answer_end_ts": 2.0}]
    ).to_csv(qa_csv, index=False)

    report = evaluate_methods(
        episode_id="ep1",
        qa_csv=qa_csv,
        methods=[method],
        chunks_dir=tmp_path,
    )
    assert report["winner"] == method
