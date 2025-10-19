from pathlib import Path

import numpy as np
import pandas as pd

from .paths import TRANSCRIPTS_DIR
from .utils import clean_text


def load_episode_parquet(episode_id: str, transcripts_dir: Path = TRANSCRIPTS_DIR) -> pd.DataFrame:
    """Load one episode's transcript and normalize schema.

    Accepts files that don't include `episode_id` and injects it.
    """
    ep_dir = transcripts_dir / f"episode_id={episode_id}"
    parts = sorted(ep_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No transcript parquet found for episode_id={episode_id} in {ep_dir}"
        )

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    # Inject episode_id if missing
    if "episode_id" not in df.columns:
        df["episode_id"] = episode_id

    # Sort for stability
    if "segment_idx" in df.columns:
        df = df.sort_values(["segment_idx", "start_ts"]).reset_index(drop=True)
    else:
        df = df.sort_values(["start_ts"]).reset_index(drop=True)

    # Clean text & ensure columns exist
    df["text"] = df["text"].map(clean_text)
    if "confidence" not in df.columns:
        df["confidence"] = np.nan

    # Coerce types
    df["start_ts"] = df["start_ts"].astype(float)
    df["end_ts"] = df["end_ts"].astype(float)

    return df


REQUIRED_TRANSCRIPT_COLS = ["episode_id", "start_ts", "end_ts", "text"]


def validate_transcript_df(df: pd.DataFrame) -> list[str]:
    errs = []
    missing = [c for c in REQUIRED_TRANSCRIPT_COLS if c not in df.columns]
    if missing:
        errs.append(f"Missing columns: {missing}")
    if not np.all(df["end_ts"] >= df["start_ts"]):
        errs.append("Found segments with end_ts < start_ts")
    if bool(df["text"].isna().any()):
        errs.append("Found NA text")
    return errs
