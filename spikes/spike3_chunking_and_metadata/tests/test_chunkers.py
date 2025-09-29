from spike3_chunking_and_metadata.chunkers import (
    PrepParams,
    make_rows_fixed,
    make_rows_sentence_bound,
    make_rows_time_window,
)


def test_make_rows_fixed(sample_df):
    params = PrepParams(method="fixed", size_tokens=20)
    rows = make_rows_fixed(sample_df, params)
    assert rows
    assert all("chunk_id" in r for r in rows)


def test_make_rows_sentence_bound(sample_df):
    params = PrepParams(method="sentence_bound", size_tokens=20)
    rows = make_rows_sentence_bound(sample_df, params)
    assert rows


def test_make_rows_time_window(sample_df):
    params = PrepParams(method="time_window", window_seconds=5, overlap_seconds=1)
    rows = make_rows_time_window(sample_df, params)
    assert rows
