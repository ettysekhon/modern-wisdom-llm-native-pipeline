from __future__ import annotations

import argparse
import json
import math
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

# Optional, but we depend on them
import tiktoken
from blingfire import text_to_sentences
from rank_bm25 import BM25Okapi
from rich import box, print
from rich.console import Console
from rich.table import Table

console = Console()

# ---------- paths & helpers ----------


def repo_root() -> Path:
    # works both when run as module or as script
    return Path.cwd() if (Path.cwd() / "data").exists() else Path(__file__).resolve().parents[3]


DATA_DIR = repo_root() / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
EVALS_DIR = DATA_DIR / "evals" / "chunking"
DOCS_DIR = repo_root() / "docs" / "decisions"
DUCKDB_PATH = DATA_DIR / "duckdb" / "modern_wisdom.duckdb"

# Tokenizer
ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text or ""))


def time_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def chunk_uuid(episode_id: str, method: str, start_ts: float, end_ts: float, chunk_v: str) -> str:
    ns = uuid.uuid5(uuid.NAMESPACE_URL, f"mw:{episode_id}:{method}:{chunk_v}")
    return str(uuid.uuid5(ns, f"{start_ts:.3f}-{end_ts:.3f}"))


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    return s


# ---------- chunkers ----------


@dataclass
class PrepParams:
    method: str
    size_tokens: int = 700
    overlap_tokens: int = 100
    window_seconds: int = 90
    overlap_seconds: int = 20
    chunk_v: str = "c1"
    prep_script_version: str = "s3.0"


def load_episode_parquet(episode_id: str, transcripts_dir: Path = TRANSCRIPTS_DIR) -> pd.DataFrame:
    # Expect layout: data/transcripts/episode_id=<id>/part-*.parquet
    parts = sorted((transcripts_dir / f"episode_id={episode_id}").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No transcript parquet found for episode_id={episode_id}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # normalize
    required = ["episode_id", "start_ts", "end_ts", "text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df["start_ts"] = df["start_ts"].astype(float)
    df["end_ts"] = df["end_ts"].astype(float)
    if "segment_idx" in df.columns:
        df = df.sort_values(["segment_idx", "start_ts"]).reset_index(drop=True)
    else:
        df = df.sort_values(["start_ts"]).reset_index(drop=True)
    df["text"] = df["text"].map(clean_text)
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    return df


def make_rows_fixed(df: pd.DataFrame, params: PrepParams) -> list[dict[str, Any]]:
    rows = []
    buf_text, buf_start, buf_end, buf_conf, seg_count = "", None, None, [], 0
    tail_text_tokens: list[int] = []  # token counts per sentence-ish piece we append

    def flush():
        nonlocal buf_text, buf_start, buf_end, buf_conf, seg_count, tail_text_tokens
        if not buf_text:
            return
        n_tokens = count_tokens(buf_text)
        row = {
            "episode_id": df["episode_id"].iloc[0],
            "start_ts": float(buf_start),
            "end_ts": float(buf_end),
            "text": buf_text,
            "n_tokens": n_tokens,
            "n_chars": len(buf_text),
            "n_sentences": None,
            "duration_s": float(buf_end - buf_start),
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
        rows.append(row)
        # prepare overlap: keep last ~overlap_tokens worth of text
        if params.overlap_tokens > 0:
            # simple heuristic: keep tail 1/3 of the text; refine later if needed
            tail_keep = max(1, math.ceil(0.33 * len(buf_text)))
            buf_text = buf_text[-tail_keep:]
            # timestamps: we don't strictly know; approximate by pulling end minus proportional duration
            dur = row["duration_s"]
            approx = max(row["start_ts"], row["end_ts"] - dur * 0.33)
            buf_start = approx
            buf_end = row["end_ts"]
            buf_conf = [row["avg_asr_confidence"]]
            seg_count = 1
        else:
            buf_text, buf_start, buf_end, buf_conf, seg_count = "", None, None, [], 0
        tail_text_tokens = []

    for _, r in df.iterrows():
        seg = r["text"]
        if not seg:
            continue
        seg = seg.strip()
        tok = count_tokens(seg)
        if buf_text == "":
            buf_text = seg
            buf_start = r["start_ts"]
            buf_end = r["end_ts"]
            seg_count = 1
            buf_conf = [r.get("confidence", np.nan)]
        else:
            if count_tokens(buf_text) + tok > params.size_tokens:
                flush()
                if buf_text == "":  # after flush with no overlap
                    buf_text = seg
                    buf_start = r["start_ts"]
                    buf_end = r["end_ts"]
                    seg_count = 1
                    buf_conf = [r.get("confidence", np.nan)]
                    continue
            buf_text = (buf_text + " " + seg).strip()
            buf_end = r["end_ts"]
            seg_count += 1
            buf_conf.append(r.get("confidence", np.nan))
    flush()
    return rows


def sentence_split_segments(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Split each segment text into sentences, inheriting the segment start/end.
    (Good enough for Spike 3; precise alignment can come later.)"""
    sents = []
    for _, r in df.iterrows():
        txt = r["text"]
        if not txt:
            continue
        for sent in [s.strip() for s in text_to_sentences(txt).split("\n") if s.strip()]:
            sents.append(
                {
                    "text": sent,
                    "start_ts": float(r["start_ts"]),
                    "end_ts": float(r["end_ts"]),
                    "confidence": r.get("confidence", np.nan),
                }
            )
    return sents


def make_rows_sentence_bound(df: pd.DataFrame, params: PrepParams) -> list[dict[str, Any]]:
    sents = sentence_split_segments(df)
    rows = []
    i = 0
    while i < len(sents):
        cur: list[dict[str, Any]] = []
        total = 0
        j = i
        while j < len(sents) and total + count_tokens(sents[j]["text"]) <= params.size_tokens:
            cur.append(sents[j])
            total += count_tokens(sents[j]["text"])
            j += 1
        if not cur:
            # extremely long sentence, force cut
            cur = [sents[j]]
            j += 1
            total = count_tokens(cur[0]["text"])
        text = " ".join(x["text"] for x in cur)
        start = cur[0]["start_ts"]
        end = cur[-1]["end_ts"]
        avg_conf = float(np.nanmean([x.get("confidence", np.nan) for x in cur]))
        row = {
            "episode_id": df["episode_id"].iloc[0],
            "start_ts": start,
            "end_ts": end,
            "text": text,
            "n_tokens": count_tokens(text),
            "n_chars": len(text),
            "n_sentences": len(cur),
            "duration_s": end - start,
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
        row["chunk_id"] = chunk_uuid(row["episode_id"], row["method"], start, end, params.chunk_v)
        rows.append(row)

        # overlap: reuse tail sentences until ~overlap_tokens
        overlap_tokens = 0
        tail = []
        k = len(cur) - 1
        while k >= 0 and overlap_tokens < params.overlap_tokens:
            tail.append(cur[k])
            overlap_tokens += count_tokens(cur[k]["text"])
            k -= 1
        # next window starts at index j - len(tail)
        i = max(i + 1, j - len(tail)) if params.overlap_tokens > 0 else j
    return rows


def make_rows_time_window(df: pd.DataFrame, params: PrepParams) -> list[dict[str, Any]]:
    rows = []
    if df.empty:
        return rows
    t0 = float(df["start_ts"].min())
    t1 = float(df["end_ts"].max())
    step = params.window_seconds - params.overlap_seconds
    w = params.window_seconds
    cur = t0
    while cur < t1:
        win_start = cur
        win_end = min(t1, cur + w)
        # include segments whose midpoint falls inside window
        mids = (df["start_ts"] + df["end_ts"]) / 2.0
        mask = (mids >= win_start) & (mids < win_end)
        sub = df.loc[mask]
        if sub.empty:
            cur += step
            continue
        text = " ".join(sub["text"].tolist()).strip()
        n_tok = count_tokens(text)
        # If too small, try extend window (up to +30s)
        if n_tok < 500 and win_end + 30 <= t1:
            win_end = min(t1, win_end + 30)
            mask = (mids >= win_start) & (mids < win_end)
            sub = df.loc[mask]
            text = " ".join(sub["text"].tolist()).strip()
            n_tok = count_tokens(text)
        start_ts = float(sub["start_ts"].min())
        end_ts = float(sub["end_ts"].max())
        row = {
            "episode_id": df["episode_id"].iloc[0],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "text": text,
            "n_tokens": n_tok,
            "n_chars": len(text),
            "n_sentences": None,
            "duration_s": end_ts - start_ts,
            "segment_count": int(len(sub)),
            "method": "time_window",
            "param_size_tokens": params.size_tokens,
            "param_overlap_tokens": None,
            "param_window_s": params.window_seconds,
            "param_overlap_s": params.overlap_seconds,
            "avg_asr_confidence": float(np.nanmean(sub["confidence"].tolist())),
            "contains_hesitation": bool(re.search(r"\b(uh|um|er)\b", text, re.I)),
            "chunk_v": params.chunk_v,
            "asr_model": df.get("asr_model", pd.Series([None])).iloc[0],
            "asr_v": None,
            "prep_script_version": params.prep_script_version,
        }
        row["chunk_id"] = chunk_uuid(
            row["episode_id"], row["method"], start_ts, end_ts, params.chunk_v
        )
        rows.append(row)
        cur += step
    return rows


# ---------- persist ----------


def write_chunks_parquet(
    rows: list[dict[str, Any]], method: str, episode_id: str, out_dir: Path = CHUNKS_DIR
) -> Path:
    if not rows:
        raise RuntimeError(f"No chunks produced for method={method}")
    df = pd.DataFrame(rows)
    dest = out_dir / method / f"episode_id={episode_id}"
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "part-00000.snappy.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def upsert_duckdb(parquet_path: Path, db_path: Path = DUCKDB_PATH):
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            episode_id TEXT,
            method TEXT,
            start_ts DOUBLE, end_ts DOUBLE, duration_s DOUBLE,
            text TEXT,
            n_tokens INTEGER, n_chars INTEGER, n_sentences INTEGER, segment_count INTEGER,
            param_size_tokens INTEGER, param_overlap_tokens INTEGER,
            param_window_s INTEGER, param_overlap_s INTEGER,
            avg_asr_confidence DOUBLE, contains_hesitation BOOLEAN,
            episode_title TEXT, publish_date TIMESTAMP, guest TEXT,
            chunk_v TEXT, asr_model TEXT, asr_v TEXT, prep_script_version TEXT
        );
    """)
    con.execute(f"""
        INSERT OR REPLACE INTO chunks
        SELECT * FROM read_parquet('{parquet_path.as_posix()}');
    """)
    con.close()


# ---------- eval ----------


def tokenize_for_bm25(texts: list[str]) -> list[list[str]]:
    return [re.findall(r"[a-zA-Z0-9']+", (t or "").lower()) for t in texts]


def bm25_rank(query: str, docs: list[str]) -> list[int]:
    tokenized_docs = tokenize_for_bm25(docs)
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(re.findall(r"[a-zA-Z0-9']+", query.lower()))
    return list(np.argsort(scores)[::-1])


def load_chunks_for_method(episode_id: str, method: str) -> pd.DataFrame:
    p = CHUNKS_DIR / method / f"episode_id={episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing chunks parquet for method {method}: {p}")
    return pd.read_parquet(p)


def evaluate_methods(
    episode_id: str, qa_csv: Path, methods: list[str], k: int = 20, tol_s: int = 7
) -> dict[str, Any]:
    qa = pd.read_csv(qa_csv)
    qa = qa[qa["episode_id"] == episode_id].reset_index(drop=True)
    if qa.empty:
        raise ValueError(f"No QA rows for episode_id={episode_id}")

    results: dict[str, Any] = {}
    for method in methods:
        chunks = load_chunks_for_method(episode_id, method)
        docs = chunks["text"].tolist()
        centers = (chunks["start_ts"] + chunks["end_ts"]) / 2.0

        hit_at = {5: 0, 10: 0, 20: 0}
        rr: list[float] = []
        time_dists: list[float] = []

        for _, row in qa.iterrows():
            q = str(row["question"])
            if (
                "optional_keywords" in qa.columns
                and isinstance(row.get("optional_keywords"), str)
                and row["optional_keywords"]
            ):
                q = q + " " + row["optional_keywords"]
            order = bm25_rank(q, docs)
            topk = order[:k]
            first_hit_rank = None
            ans_mid = 0.5 * (row["answer_start_ts"] + row["answer_end_ts"])
            for rank_idx, doc_idx in enumerate(topk, start=1):
                c_start = chunks["start_ts"].iloc[doc_idx]
                c_end = chunks["end_ts"].iloc[doc_idx]
                overlap = time_overlap(
                    (c_start, c_end), (row["answer_start_ts"] - tol_s, row["answer_end_ts"] + tol_s)
                )
                if overlap >= 1.0:
                    first_hit_rank = rank_idx
                    dist = abs(float(centers.iloc[doc_idx]) - float(ans_mid))
                    time_dists.append(dist)
                    break
            # metrics
            if first_hit_rank is not None:
                if first_hit_rank <= 5:
                    hit_at[5] += 1
                if first_hit_rank <= 10:
                    hit_at[10] += 1
                if first_hit_rank <= 20:
                    hit_at[20] += 1
                rr.append(1.0 / first_hit_rank)
            else:
                rr.append(0.0)

        n = len(qa)
        results[method] = {
            "Hit@5": hit_at[5] / n,
            "Hit@10": hit_at[10] / n,
            "Hit@20": hit_at[20] / n,
            "MRR": float(np.mean(rr)),
            "AvgTimeDistanceSec": float(np.mean(time_dists)) if time_dists else None,
            "AvgTokens": float(chunks["n_tokens"].mean()),
            "AvgDurationSec": float(chunks["duration_s"].mean()),
        }

    # choose winner
    winner = max(methods, key=lambda m: (results[m]["Hit@10"], results[m]["MRR"]))
    return {
        "episode_id": episode_id,
        "k": k,
        "tolerance_s": tol_s,
        "methods": results,
        "winner": winner,
    }


# --- validators ---
REQUIRED_TRANSCRIPT_COLS = ["episode_id", "start_ts", "end_ts", "text"]


def validate_transcript_df(df: pd.DataFrame) -> list[str]:
    errs = []
    missing = [c for c in REQUIRED_TRANSCRIPT_COLS if c not in df.columns]
    if missing:
        errs.append(f"Missing columns: {missing}")
    if not np.all(df["end_ts"] >= df["start_ts"]):
        errs.append("Found segments with end_ts < start_ts")
    if df["text"].isna().any():
        errs.append("Found NA text")
    if (df["end_ts"].diff().fillna(0) < -1e-6).any() and "segment_idx" not in df.columns:
        errs.append("Timestamps not monotonic; consider sorting by start_ts")
    tiny = (df["text"].str.len() < 3).sum()
    if tiny > 0:
        errs.append(f"{tiny} segments have <3 chars (will be ignored/merged later)")
    return errs


def validate_chunks_df(chunks: pd.DataFrame, method: str) -> list[str]:
    errs = []
    if chunks.empty:
        errs.append(f"No chunks produced for {method}")
    if (chunks["n_tokens"] <= 0).any():
        errs.append("n_tokens <= 0")
    if (chunks["end_ts"] <= chunks["start_ts"]).any():
        errs.append("chunk end_ts <= start_ts")
    if chunks["chunk_id"].duplicated().any():
        errs.append("duplicate chunk_id values")
    if (chunks["duration_s"] <= 0).any():
        errs.append("non-positive duration")
    # soft cap sanity
    if chunks["n_tokens"].mean() > 900:
        errs.append("avg n_tokens too large (>900)")
    return errs


# ---------- CLI ----------


def cmd_chunk(args: argparse.Namespace):
    episode_id = args.episode_id
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    params = PrepParams(
        method="",  # per method
        size_tokens=args.size_tokens,
        overlap_tokens=args.overlap_tokens,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        chunk_v=args.chunk_v,
    )
    df = load_episode_parquet(episode_id)
    errs = validate_transcript_df(df)
    if errs:
        console.print("[red]Transcript preflight issues:[/red]")
        for e in errs:
            console.print(f" - {e}")
        if not args.force:
            sys.exit(2)

    produced: list[tuple[str, Path]] = []
    for m in methods:
        params.method = m
        if m == "fixed":
            rows = make_rows_fixed(df, params)
        elif m == "sentence_bound":
            rows = make_rows_sentence_bound(df, params)
        elif m == "time_window":
            rows = make_rows_time_window(df, params)
        else:
            raise ValueError(f"Unknown method: {m}")
        out_path = write_chunks_parquet(rows, m, episode_id)
        if args.duckdb:
            upsert_duckdb(out_path)
        produced.append((m, out_path))
        out_path = write_chunks_parquet(rows, m, episode_id) if not args.dry_run else None
        df_out = pd.DataFrame(rows)
        issues = validate_chunks_df(df_out, m)
        if issues:
            console.print(f"[red]Chunk validation failed for {m}[/red]")
            for e in issues:
                console.print(f" - {e}")
            if not args.force:
                sys.exit(3)
        if args.duckdb and not args.dry_run:
            upsert_duckdb(out_path)
        produced.append((m, out_path or Path("<dry-run>")))

    table = Table(title="Chunk outputs", box=box.SIMPLE)
    table.add_column("Method")
    table.add_column("Parquet path")
    table.add_column("Rows")
    for m, p in produced:
        n = pd.read_parquet(p).shape[0]
        table.add_row(m, p.as_posix(), str(n))
    console.print(table)


def cmd_eval(args: argparse.Namespace):
    episode_id = args.episode_id
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    report = evaluate_methods(
        episode_id, Path(args.qa_csv), methods, k=args.k, tol_s=args.tolerance_s
    )
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVALS_DIR / "chunk_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    # tiny MD decision note
    md = DOCS_DIR / "0003-chunking.md"
    lines = [
        f"# Chunking decision — {datetime.utcnow().isoformat()}Z",
        f"- Episode: `{episode_id}`",
        f"- Methods evaluated: {', '.join(methods)}",
        f"- Winner (Hit@10 → tie-break MRR): **{report['winner']}**",
        "## Metrics",
        "```json",
        json.dumps(report["methods"], indent=2),
        "```",
        f"_k={report['k']}, tolerance_s={report['tolerance_s']}_",
        "",
        "Carry these params forward to Spike 4.",
    ]
    md.write_text("\n".join(lines))
    print(f"[green]Wrote[/green] {report_path.as_posix()} and {md.as_posix()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spike3", description="Spike 3 — Chunking & metadata")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_chunk = sub.add_parser("chunk", help="Create chunks for one episode")
    p_chunk.add_argument("--episode-id", required=True)
    p_chunk.add_argument(
        "--methods",
        default="fixed,sentence_bound,time_window",
        help="Comma list: fixed,sentence_bound,time_window",
    )
    p_chunk.add_argument("--size-tokens", type=int, default=700)
    p_chunk.add_argument("--overlap-tokens", type=int, default=100)
    p_chunk.add_argument("--window-seconds", type=int, default=90)
    p_chunk.add_argument("--overlap-seconds", type=int, default=20)
    p_chunk.add_argument("--chunk-v", default="c1")
    p_chunk.add_argument("--duckdb", action="store_true", help="Upsert chunks into DuckDB")
    p_chunk.add_argument(
        "--dry-run", action="store_true", help="Build in-memory, validate, but don't write"
    )
    p_chunk.add_argument(
        "--force", action="store_true", help="Ignore validation errors and continue"
    )
    p_chunk.set_defaults(func=cmd_chunk)

    p_eval = sub.add_parser("eval", help="BM25-based lexical evaluation")
    p_eval.add_argument("--episode-id", required=True)
    p_eval.add_argument("--methods", default="fixed,sentence_bound,time_window")
    p_eval.add_argument(
        "--qa-csv",
        required=True,
        help="CSV with question,episode_id,answer_start_ts,answer_end_ts[,optional_keywords]",
    )
    p_eval.add_argument("--k", type=int, default=20)
    p_eval.add_argument("--tolerance-s", type=int, default=7)
    p_eval.set_defaults(func=cmd_eval)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
