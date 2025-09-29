import re
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tiktoken

from .schema import REQUIRED_CHUNK_COLS

ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text or ""))


def chunk_uuid(episode_id: str, method: str, start_ts: float, end_ts: float, chunk_v: str) -> str:
    ns = uuid.uuid5(uuid.NAMESPACE_URL, f"mw:{episode_id}:{method}:{chunk_v}")
    return str(uuid.uuid5(ns, f"{start_ts:.3f}-{end_ts:.3f}"))


@dataclass
class PrepParams:
    method: str
    size_tokens: int = 700
    overlap_tokens: int = 100
    window_seconds: int = 90
    overlap_seconds: int = 20
    chunk_v: str = "c1"
    prep_script_version: str = "s3.0"


def _fill_missing_fields(row: dict) -> dict:
    """Ensure all REQUIRED_CHUNK_COLS exist (use None/defaults)."""
    filled = dict.fromkeys(REQUIRED_CHUNK_COLS)
    filled.update(row)
    return filled


def make_rows_fixed(df: pd.DataFrame, params: PrepParams) -> list[dict]:
    rows = []
    buf_text, buf_start, buf_end, buf_conf, seg_count = "", None, None, [], 0

    def flush():
        nonlocal buf_text, buf_start, buf_end, buf_conf, seg_count
        if not buf_text:
            return
        row = {
            "episode_id": df["episode_id"].iloc[0],
            "start_ts": float(buf_start),
            "end_ts": float(buf_end),
            "duration_s": float(buf_end - buf_start),
            "text": buf_text,
            "n_tokens": count_tokens(buf_text),
            "n_chars": len(buf_text),
            "n_sentences": None,
            "segment_count": seg_count,
            "method": "fixed",
            "param_size_tokens": params.size_tokens,
            "param_overlap_tokens": params.overlap_tokens,
            "param_window_s": None,
            "param_overlap_s": None,
            "avg_asr_confidence": float(np.nanmean(buf_conf)) if buf_conf else np.nan,
            "contains_hesitation": bool(re.search(r"\b(uh|um|er)\b", buf_text, re.I)),
            "chunk_v": params.chunk_v,
            "asr_model": df.get("asr_model", pd.Series([None])).iloc[0],
            "asr_v": None,
            "prep_script_version": params.prep_script_version,
        }
        row["chunk_id"] = chunk_uuid(
            row["episode_id"], row["method"], row["start_ts"], row["end_ts"], params.chunk_v
        )
        rows.append(_fill_missing_fields(row))
        buf_text = ""
        buf_start = buf_end = None
        buf_conf, seg_count = [], 0

    for _, r in df.iterrows():
        seg = (r["text"] or "").strip()
        if not seg:
            continue
        tok = count_tokens(seg)
        if buf_text == "":
            buf_text, buf_start, buf_end, buf_conf, seg_count = (
                seg,
                r["start_ts"],
                r["end_ts"],
                [r.get("confidence", np.nan)],
                1,
            )
        else:
            if count_tokens(buf_text) + tok > params.size_tokens:
                flush()
                buf_text, buf_start, buf_end, buf_conf, seg_count = (
                    seg,
                    r["start_ts"],
                    r["end_ts"],
                    [r.get("confidence", np.nan)],
                    1,
                )
            else:
                buf_text = f"{buf_text} {seg}".strip()
                buf_end = r["end_ts"]
                seg_count += 1
                buf_conf.append(r.get("confidence", np.nan))
    flush()
    return rows


# --- sentence splitting (blingfire with safe fallback) ---
try:
    from blingfire import text_to_sentences as _bf_text_to_sentences  # type: ignore

    def _sent_split(txt: str) -> list[str]:
        return [s.strip() for s in _bf_text_to_sentences(txt or "").split("\n") if s.strip()]

except Exception:
    # very simple regex fallback (good enough for Spike 3)
    def _sent_split(txt: str) -> list[str]:
        if not txt:
            return []
        # split on ., !, ? while keeping abbreviations minimally intact
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", txt.strip())
        return [p.strip() for p in parts if p.strip()]


def sentence_split_segments(df: pd.DataFrame) -> list[dict]:
    """
    Split each segment into sentences, inheriting the segment's start/end/confidence.
    (Spike-3 approximation: sentence times == segment times.)
    """
    sents: list[dict] = []
    has_conf = "confidence" in df.columns
    for _, r in df.iterrows():
        txt = (r["text"] or "").strip()
        if not txt:
            continue
        for s in _sent_split(txt):
            sents.append(
                {
                    "text": s,
                    "start_ts": float(r["start_ts"]),
                    "end_ts": float(r["end_ts"]),
                    "confidence": float(r["confidence"]) if has_conf else np.nan,
                }
            )
    return sents


def make_rows_sentence_bound(df: pd.DataFrame, params: PrepParams) -> list[dict]:
    sents = sentence_split_segments(df)
    rows: list[dict] = []
    i = 0
    while i < len(sents):
        window: list[dict] = []
        total = 0
        j = i
        # pack sentences up to size_tokens (respect boundaries)
        while j < len(sents):
            t = count_tokens(sents[j]["text"])
            if total + t > params.size_tokens and window:
                break
            window.append(sents[j])
            total += t
            j += 1

        # fallback if one sentence is huge
        if not window:
            window = [sents[j]]
            j += 1
            total = count_tokens(window[0]["text"])

        text = " ".join(x["text"] for x in window).strip()
        start_ts = float(window[0]["start_ts"])
        end_ts = float(window[-1]["end_ts"])
        avg_conf = float(np.nanmean([w.get("confidence", np.nan) for w in window]))

        row = {
            "episode_id": df["episode_id"].iloc[0],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_s": end_ts - start_ts,
            "text": text,
            "n_tokens": count_tokens(text),
            "n_chars": len(text),
            "n_sentences": len(window),
            "segment_count": None,
            "method": "sentence_bound",
            "param_size_tokens": params.size_tokens,
            "param_overlap_tokens": params.overlap_tokens,
            "param_window_s": None,
            "param_overlap_s": None,
            "avg_asr_confidence": avg_conf,
            "contains_hesitation": bool(re.search(r"\b(uh|um|er)\b", text, re.I)),
            "chunk_v": params.chunk_v,
            "asr_model": df.get("asr_model", pd.Series([None])).iloc[0],
            "asr_v": None,
            "prep_script_version": params.prep_script_version,
        }
        row["chunk_id"] = chunk_uuid(
            row["episode_id"], row["method"], start_ts, end_ts, params.chunk_v
        )
        rows.append(_fill_missing_fields(row))

        # implement ~overlap_tokens by reusing tail sentences
        if params.overlap_tokens > 0:
            overlap_tokens = 0
            k = len(window) - 1
            tail = 0
            while k >= 0 and overlap_tokens < params.overlap_tokens:
                overlap_tokens += count_tokens(window[k]["text"])
                tail += 1
                k -= 1
            i = max(i + 1, j - tail)
        else:
            i = j
    return rows


def make_rows_time_window(df: pd.DataFrame, params: PrepParams) -> list[dict]:
    rows: list[dict] = []
    if df.empty:
        return rows

    t0 = float(df["start_ts"].min())
    t1 = float(df["end_ts"].max())
    step = max(1, params.window_seconds - params.overlap_seconds)
    cur = t0

    mids = (df["start_ts"] + df["end_ts"]) / 2.0

    while cur < t1:
        win_start = cur
        win_end = min(t1, cur + params.window_seconds)
        mask = (mids >= win_start) & (mids < win_end)
        sub = df.loc[mask]

        if not sub.empty:
            text = " ".join(sub["text"].astype(str).tolist()).strip()
            start_ts = float(sub["start_ts"].min())
            end_ts = float(sub["end_ts"].max())
            avg_conf = (
                float(np.nanmean(sub["confidence"])) if "confidence" in sub.columns else np.nan
            )

            row = {
                "episode_id": df["episode_id"].iloc[0],
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_s": end_ts - start_ts,
                "text": text,
                "n_tokens": count_tokens(text),
                "n_chars": len(text),
                "n_sentences": None,
                "segment_count": int(len(sub)),
                "method": "time_window",
                "param_size_tokens": None,
                "param_overlap_tokens": None,
                "param_window_s": params.window_seconds,
                "param_overlap_s": params.overlap_seconds,
                "avg_asr_confidence": avg_conf,
                "contains_hesitation": bool(re.search(r"\b(uh|um|er)\b", text, re.I)),
                "chunk_v": params.chunk_v,
                "asr_model": df.get("asr_model", pd.Series([None])).iloc[0],
                "asr_v": None,
                "prep_script_version": params.prep_script_version,
            }
            row["chunk_id"] = chunk_uuid(
                row["episode_id"], row["method"], start_ts, end_ts, params.chunk_v
            )
            rows.append(_fill_missing_fields(row))

        cur += step

    return rows


__all__ = [
    "make_rows_fixed",
    "make_rows_sentence_bound",
    "make_rows_time_window",
    "sentence_split_segments",
]
