from pathlib import Path

import pandas as pd
import pytest
from spike3_chunking_and_metadata.schema import REQUIRED_CHUNK_COLS


def make_chunk_row(
    *,
    episode_id: str = "ep1",
    method: str = "fixed",
    start_ts: float = 0.0,
    end_ts: float = 2.0,
    text: str = "Hello world",
    chunk_id: str = "test-chunk-1",
) -> dict:
    """Return a dict with ALL REQUIRED_CHUNK_COLS populated (using sensible defaults)."""
    base = {
        "chunk_id": chunk_id,
        "episode_id": episode_id,
        "method": method,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": end_ts - start_ts,
        "text": text,
        "n_tokens": 3,
        "n_chars": len(text),
        "n_sentences": 1,  # OK if None for some methods
        "segment_count": 1,  # OK if None for some methods
        "param_size_tokens": 20 if method != "time_window" else None,
        "param_overlap_tokens": 5 if method != "time_window" else None,
        "param_window_s": 190 if method == "time_window" else None,
        "param_overlap_s": 30 if method == "time_window" else None,
        "avg_asr_confidence": 0.9,
        "contains_hesitation": False,
        "chunk_v": "c1",
        "asr_model": "dummy",
        "asr_v": None,
        "prep_script_version": "s3.0",
    }
    # Ensure all required cols exist (and no extras)
    filled = {c: base.get(c) for c in REQUIRED_CHUNK_COLS}
    return filled


@pytest.fixture
def sample_df():
    # Minimal transcript-like sample used by other tests
    return pd.DataFrame(
        [
            {
                "episode_id": "ep1",
                "segment_idx": 0,
                "start_ts": 0.0,
                "end_ts": 2.0,
                "text": "Hello world",
                "confidence": 0.9,
            },
            {
                "episode_id": "ep1",
                "segment_idx": 1,
                "start_ts": 2.0,
                "end_ts": 4.0,
                "text": "This is a test.",
                "confidence": 0.95,
            },
            {
                "episode_id": "ep1",
                "segment_idx": 2,
                "start_ts": 4.0,
                "end_ts": 6.0,
                "text": "Goodbye!",
                "confidence": 0.92,
            },
        ]
    )


def write_valid_chunk_parquet(
    dirpath: Path, episode_id: str = "ep1", method: str = "fixed"
) -> Path:
    """Create a valid chunks parquet under <dir>/<method>/episode_id=<id>/part-00000.snappy.parquet."""
    out_dir = dirpath / method / f"episode_id={episode_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    row = make_chunk_row(episode_id=episode_id, method=method, chunk_id="chunk-1")
    df = pd.DataFrame([row], columns=REQUIRED_CHUNK_COLS)
    out_path = out_dir / "part-00000.snappy.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
