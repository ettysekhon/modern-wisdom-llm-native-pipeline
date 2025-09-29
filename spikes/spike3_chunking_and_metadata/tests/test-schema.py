import pandas as pd
from spike3_chunking_and_metadata.schema import REQUIRED_CHUNK_COLS, validate_chunk_df_columns


def test_required_cols_present():
    df = pd.DataFrame([dict.fromkeys(REQUIRED_CHUNK_COLS)])
    assert validate_chunk_df_columns(df) == []


def test_missing_cols_detected():
    df = pd.DataFrame([{"episode_id": "ep1", "text": "x"}])
    missing = validate_chunk_df_columns(df)
    assert "chunk_id" in missing and "n_tokens" in missing
