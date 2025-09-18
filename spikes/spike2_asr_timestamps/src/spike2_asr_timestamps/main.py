from __future__ import annotations

import csv
import os
import time
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import requests

# ---------- optional local backend (faster-whisper) ----------
try:
    from faster_whisper import WhisperModel  # only used if ASR_BACKEND=local
except Exception:  # pragma: no cover
    WhisperModel = None  # type: ignore

# ---------- config ----------
DATASET = "mw"  # same schema from Spike 1
DB_PATH = Path(os.environ.get("DUCK_PATH", "./data/duckdb/modern_wisdom.duckdb")).resolve()
AUDIO_DIR = Path("./data/audio").resolve()
OUT_DIR = Path("./data/transcripts").resolve()
INDEX_CSV = Path("./data/transcripts/index.csv").resolve()

ASR_DIARIZATION = int(os.environ.get("ASR_DIARIZATION", "1"))

# backend selection
BACKEND = os.environ.get("ASR_BACKEND", "assemblyai").lower()  # "assemblyai" | "local"

# local (faster-whisper) knobs
MODEL = os.environ.get("ASR_MODEL", "medium")
BEAM = int(os.environ.get("ASR_BEAM", "5"))
BATCH_N = int(os.environ.get("ASR_BATCH", "8"))

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- utilities ----------
def _episode_paths(episode_id: str):
    ep_dir = OUT_DIR / f"episode_id={episode_id}"
    final_pq = ep_dir / "part-00000.snappy.parquet"
    tmp_pq = ep_dir / "part-00000.snappy.parquet.tmp"
    err_txt = OUT_DIR / f"episode_id={episode_id}.error.txt"
    return ep_dir, final_pq, tmp_pq, err_txt


def _segment_words(words, target_window_sec=15.0):
    rows, buf, start_ts = [], [], None
    cur_end = 0.0
    for w in words or []:
        s = (w.start or 0) / 1000.0
        e = (w.end or 0) / 1000.0
        t = w.text or ""
        if start_ts is None:
            start_ts = s
        buf.append(t)
        cur_end = e
        # split on sentence-ish punctuation or window size
        if t.endswith((".", "!", "?", "…")) or (cur_end - start_ts) >= target_window_sec:
            rows.append((" ".join(buf).strip(), start_ts, cur_end))
            buf, start_ts = [], None
    if buf:
        rows.append((" ".join(buf).strip(), start_ts or cur_end, cur_end))
    return rows


def autotune_batch_size(model, wav_path, candidates=(12, 10, 8, 6, 4, 2)) -> int:
    """Try a few batch sizes on a short slice; keep the fastest. Local backend only."""
    best_bs, best_rate = None, 0.0
    for bs in candidates:
        with suppress(Exception):
            t0 = time.time()
            # Short trial: limit compute via chunk_length
            list(
                model.transcribe(
                    str(wav_path),
                    beam_size=5,
                    vad_filter=True,
                    chunk_length=30,
                    # fast-path kwargs (some versions may not accept them)
                    **({"batch_size": bs} if bs else {}),
                    **({"num_workers": 4}),
                )[0]
            )
            dt = max(time.time() - t0, 1e-6)
            rate = 30.0 / dt  # "sec of audio per wall-sec" for trial
            if rate > best_rate:
                best_rate, best_bs = rate, bs
    return best_bs or 4


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _index_header():
    return [
        "episode_id",
        "status",
        "segments",
        "audio_seconds",
        "seconds_taken",
        "avg_seconds_per_ep",
        "started_at",
        "ended_at",
        "asr_model",
        "error",
    ]


def _append_index_row(row: dict):
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = INDEX_CSV.exists()
    with INDEX_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_index_header())
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _write_parquet(episode_id: str, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    ep_dir, final_pq, tmp_pq, _ = _episode_paths(episode_id)
    ep_dir.mkdir(parents=True, exist_ok=True)
    # write to .tmp first, then rename atomically
    df.to_parquet(tmp_pq, compression="snappy")
    os.replace(tmp_pq, final_pq)


# ---------- episode iterator ----------
def _count_candidates(limit: int | None) -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = f"""
      SELECT id
      FROM {DATASET}.episodes
      WHERE audio_url IS NOT NULL
      ORDER BY publish_date
      {"" if limit is None else f"LIMIT {int(limit)}"}
    """
    total = con.execute(f"SELECT COUNT(*) FROM ({q}) t").fetchone()[0]
    con.close()
    return total


def iter_episodes(limit: int | None = None) -> Iterable[dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = f"""
      SELECT id as episode_id, audio_url
      FROM {DATASET}.episodes
      WHERE audio_url IS NOT NULL
      ORDER BY publish_date
      {"" if limit is None else f"LIMIT {int(limit)}"}
    """
    for ep in con.execute(q).fetchall():
        yield {"episode_id": ep[0], "audio_url": ep[1]}
    con.close()


# ---------- downloads ----------
def _download(url: str, dst: Path, tries: int = 3, backoff: float = 1.5) -> Path:
    for i in range(tries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            return dst
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(backoff**i)


# ---------- ASSEMBLYAI backend ----------
def transcribe_assemblyai(audio_url: str) -> tuple[list[dict], str]:
    """
    AssemblyAI SDK >= 0.43.x
    Env:
      - ASSEMBLYAI_API_KEY (required)
      - ASR_DIARIZATION={0|1,true,false,...} to include speaker labels
    Returns (rows, "assemblyai").
    """
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set")

    import assemblyai as aai

    aai.settings.api_key = api_key

    cfg = aai.TranscriptionConfig(
        punctuate=True,
        speaker_labels=True,  # enable diarization if requested
        speaker_options={
            "min_speakers": 1,
            "max_speakers": 10,
        },
    )
    tx = aai.Transcriber().transcribe(audio_url, config=cfg)

    # Per latest SDK: check error attr (or status==error)
    if getattr(tx, "error", None):
        raise RuntimeError(tx.error)

    rows: list[dict] = []
    if getattr(tx, "utterances", None):  # diarization path (preferred; has timestamps)
        for i, u in enumerate(tx.utterances):
            row = {
                "segment_idx": i,
                "start_ts": float(u.start) / 1000.0,
                "end_ts": float(u.end) / 1000.0,
                "text": (u.text or "").strip(),
                "asr_model": "assemblyai",
                "confidence": float(getattr(u, "confidence", 0.0) or 0.0),
            }
            if ASR_DIARIZATION and getattr(u, "speaker", None) is not None:
                row["speaker"] = u.speaker
            rows.append(row)
    else:
        # Word-based segmentation fallback (keeps timestamps even w/o utterances)
        chunks = _segment_words(getattr(tx, "words", []), target_window_sec=15.0)
        if chunks:
            for i, (text, s, e) in enumerate(chunks):
                rows.append(
                    {
                        "segment_idx": i,
                        "start_ts": s,
                        "end_ts": e,
                        "text": text,
                        "asr_model": "assemblyai",
                        "confidence": float(getattr(tx, "confidence", 0.0) or 0.0),
                    }
                )
        else:
            # Final fallback: single full-text segment
            duration = float(getattr(tx, "audio_duration", 0.0) or 0.0)
            rows.append(
                {
                    "segment_idx": 0,
                    "start_ts": 0.0,
                    "end_ts": duration,
                    "text": (tx.text or "").strip(),
                    "asr_model": "assemblyai",
                    "confidence": float(getattr(tx, "confidence", 0.0) or 0.0),
                }
            )

    return rows, "assemblyai"


# ---------- LOCAL backend (faster-whisper) ----------
def _transcribe_file_local(
    model: WhisperModel,
    wav_path: Path,
    *,
    beam: int,
    batch_size: int | None,
    num_workers: int | None,
):
    """Transcribe a single file with faster-whisper, returning rows with timestamps."""
    # Try newer API; fallback if needed
    try:
        segments, info = model.transcribe(
            str(wav_path),
            beam_size=beam,
            vad_filter=True,
            chunk_length=30,
            **({"batch_size": batch_size} if batch_size else {}),
            **({"num_workers": num_workers} if num_workers else {}),
            condition_on_previous_text=False,
        )
    except TypeError:
        segments, info = model.transcribe(
            str(wav_path),
            beam_size=beam,
            vad_filter=True,
            chunk_length=30,
            condition_on_previous_text=False,
        )

    rows = []
    for idx, s in enumerate(segments):
        rows.append(
            {
                "segment_idx": idx,
                "start_ts": float(s.start),
                "end_ts": float(s.end),
                "text": s.text.strip(),
                "asr_model": f"faster-whisper/{MODEL}",
                "confidence": float(getattr(s, "avg_logprob", 0.0) or 0.0),
            }
        )
    return rows


def transcribe_local(audio_url: str, episode_id: str) -> tuple[list[dict], str]:
    if WhisperModel is None:
        raise RuntimeError("Local backend unavailable: faster-whisper not installed")

    model = WhisperModel(MODEL, device="auto", compute_type="int8")

    # ensure local file exists
    local_audio = AUDIO_DIR / f"{episode_id}.mp3"
    if not local_audio.exists():
        _download(audio_url, local_audio)

    # autotune once per run using this first file
    tuned_batch = autotune_batch_size(model, local_audio, candidates=(12, 10, 8, 6, 4, 2))
    print(f"[autotune] using batch_size={tuned_batch}")

    rows = _transcribe_file_local(
        model,
        local_audio,
        beam=BEAM,
        batch_size=tuned_batch,
        num_workers=4,
    )
    return rows, f"faster-whisper/{MODEL}"


# ---------- main ----------
def main():
    # Resolve limit (0 or unset = all episodes)
    limit_env = int(os.environ.get("ASR_LIMIT", "0"))
    limit = limit_env or None

    # Retry toggle for episodes that previously errored
    retry_errors = os.environ.get("ASR_RETRY_ERRORS", "").lower() in {"1", "true", "yes", "on"}

    # Count totals
    total_all = _count_candidates(None)  # all available in DB
    total = _count_candidates(limit)  # planned for this run

    processed = ok = fail = 0
    cumulative_sec = 0.0
    start_all = time.time()

    mode = "all episodes" if limit is None else f"first {limit} episodes"
    print(f"ASR start | backend={BACKEND} | mode={mode} | available={total_all} | planned={total}")

    for ep in iter_episodes(limit=limit):
        episode_id = ep["episode_id"]
        audio_url = ep["audio_url"]

        ep_dir, final_pq, tmp_pq, err_txt = _episode_paths(episode_id)

        # Strict skip logic:
        if final_pq.exists():
            processed += 1
            avg = (cumulative_sec / ok) if ok else 0.0
            eta = max(total - processed, 0) * avg
            print(
                f"[{processed}/{total}] skip {episode_id} (done) | avg {avg:,.1f}s/ep | ETA {_fmt_hms(eta)}"
            )
            continue

        # Clean up partial temp files from a prior crash
        if ep_dir.exists() and tmp_pq.exists():
            with suppress(Exception):
                tmp_pq.unlink()

        # If there was a prior error and no retry requested, skip
        if err_txt.exists() and not retry_errors:
            processed += 1
            avg = (cumulative_sec / ok) if ok else 0.0
            eta = max(total - processed, 0) * avg
            print(
                f"[{processed}/{total}] skip {episode_id} (error; set ASR_RETRY_ERRORS=1 to retry) | avg {avg:,.1f}s/ep | ETA {_fmt_hms(eta)}"
            )
            continue
        elif err_txt.exists() and retry_errors:
            with suppress(Exception):
                err_txt.unlink()

        ep_started = time.time()
        status = "ok"
        err_msg = ""
        seg_count = 0
        audio_seconds = None
        model_name = BACKEND

        try:
            if BACKEND == "assemblyai":
                rows, model_name = transcribe_assemblyai(audio_url)
            elif BACKEND == "local":
                rows, model_name = transcribe_local(audio_url, episode_id)
            else:
                raise ValueError(f"Unsupported ASR_BACKEND: {BACKEND}")

            seg_count = len(rows)
            if seg_count:
                audio_seconds = max((r["end_ts"] for r in rows), default=None)

            _write_parquet(episode_id, rows)
            ok += 1

        except Exception as e:
            status = "error"
            err_msg = str(e)
            err_txt.write_text(err_msg)
            fail += 1

        took = time.time() - ep_started
        processed += 1
        if status == "ok":
            cumulative_sec += took
        avg = (cumulative_sec / ok) if ok else 0.0
        eta = max(total - processed, 0) * avg

        print(
            f"[{processed}/{total}] {episode_id} | "
            f"{'OK' if status == 'ok' else 'ERR'} | "
            f"took {took:,.1f}s | avg {avg:,.1f}s/ep | ETA {_fmt_hms(eta)}"
            + (f" | segs {seg_count}" if seg_count else "")
        )

        _append_index_row(
            {
                "episode_id": episode_id,
                "status": status,
                "segments": seg_count,
                "audio_seconds": f"{audio_seconds:.1f}" if audio_seconds is not None else "",
                "seconds_taken": f"{took:.1f}",
                "avg_seconds_per_ep": f"{avg:.1f}" if ok else "",
                "started_at": datetime.fromtimestamp(ep_started, tz=UTC).isoformat(),
                "ended_at": datetime.fromtimestamp(time.time(), tz=UTC).isoformat(),
                "asr_model": model_name,
                "error": err_msg,
            }
        )

    total_time = time.time() - start_all
    avg_ok = (cumulative_sec / ok) if ok else 0.0
    print(
        f"ASR done | backend={BACKEND} | mode={mode} | "
        f"available={total_all} | planned={total} | processed={processed} | ok={ok} | fail={fail} | "
        f"elapsed={_fmt_hms(total_time)} | avg_ok={avg_ok:.1f}s/ep"
    )


if __name__ == "__main__":
    main()
