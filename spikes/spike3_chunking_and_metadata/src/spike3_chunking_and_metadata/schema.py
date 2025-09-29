# All chunk parquets must include these columns
REQUIRED_CHUNK_COLS = [
    "chunk_id",
    "episode_id",
    "method",
    "start_ts",
    "end_ts",
    "duration_s",
    "text",
    "n_tokens",
    "n_chars",
    "n_sentences",  # may be None for some methods
    "segment_count",  # may be None for some methods
    # provenance / params
    "param_size_tokens",
    "param_overlap_tokens",
    "param_window_s",
    "param_overlap_s",
    # quality / version
    "avg_asr_confidence",
    "contains_hesitation",
    "chunk_v",
    "asr_model",
    "asr_v",
    "prep_script_version",
]


def validate_chunk_df_columns(df) -> list[str]:
    missing = [c for c in REQUIRED_CHUNK_COLS if c not in df.columns]
    return missing
