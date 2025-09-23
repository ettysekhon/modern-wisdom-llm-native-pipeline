import os
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path

import dlt
import duckdb
from dlt.destinations import duckdb as duckdb_dest

DUCK_PATH = os.environ.get("DUCK_PATH", "./data/duckdb/modern_wisdom.duckdb")

_TAG_RE = re.compile(r"<[^>]+>")
_TS_RE = re.compile(
    r"(?:(?:^|\b|\(|\[))(?P<time>\d{1,2}:\d{2}(?::\d{2})?)(?:\)|\])?\s*(?:[-–—:]\s*)?(?P<label>[^\n\r]+?)(?=(?:\s*(?:\(|\[)?\d{1,2}:\d{2}(?::\d{2})?|\n|$))",
    re.IGNORECASE,
)


def _strip_html(s: str | None) -> str:
    if not s:
        return ""
    s = unescape(s)
    s = _TAG_RE.sub("", s).replace("\xa0", " ")
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()


def _normalize_hms(t: str) -> str:
    parts = t.split(":")
    if len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    elif len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        return t
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_timestamps(text: str) -> list[dict]:
    out: list[dict] = []
    for m in _TS_RE.finditer(text or ""):
        out.append(
            {"time": _normalize_hms(m.group("time")), "label": m.group("label").strip(" -–—\t")}
        )
    seen: set[tuple[str, str]] = set()
    dedup: list[dict] = []
    for t in out:
        key = (t["time"], t["label"])
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    return dedup


def _to_iso_utc(entry) -> str | None:
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if t:
        return datetime(*t[:6], tzinfo=UTC).isoformat()
    s = getattr(entry, "published", "") or getattr(entry, "updated", "")
    if not s:
        return None
    try:
        return parsedate_to_datetime(s).astimezone(UTC).isoformat()
    except Exception:
        return None


def _ensure_duck_dir(path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def _get_prev_stats(db_file: str, dataset: str):
    if not Path(db_file).exists():
        return 0, None
    con = duckdb.connect(db_file, read_only=False)
    try:
        prev_total = con.execute(f"SELECT COUNT(*) FROM {dataset}.episodes").fetchone()[0]
    except Exception:
        prev_total = 0
    last_run_started_at = None
    try:
        last_row = con.execute(
            f"""
            SELECT started_at
            FROM {dataset}.ingest_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
        if last_row and last_row[0]:
            last_run_started_at = str(last_row[0])
    except Exception:
        pass
    con.close()
    return prev_total, last_run_started_at


def _post_run_stats(
    db_file: str, dataset: str, last_run_started_at_iso: str | None, prev_total: int
):
    con = duckdb.connect(db_file, read_only=False)
    total_after = con.execute(f"SELECT COUNT(*) FROM {dataset}.episodes").fetchone()[0]
    rows_ingested = max(total_after - prev_total, 0)
    rows_updated = 0
    if last_run_started_at_iso:
        changed_since = con.execute(
            f"""
            SELECT COUNT(*) FROM {dataset}.episodes
            WHERE updated_at IS NOT NULL AND updated_at > ?
            """,
            [last_run_started_at_iso],
        ).fetchone()[0]
        rows_updated = max(changed_since - rows_ingested, 0)
    con.close()
    return rows_ingested, rows_updated, total_after


@dlt.resource(name="episodes", primary_key="id", write_disposition="merge")
def rss_episodes(cursor=None):
    if cursor is None:
        cursor = dlt.sources.incremental("updated_at", initial_value="2000-01-01T00:00:00Z")
    import hashlib

    import feedparser
    from dateutil import parser as dtparser

    feed = feedparser.parse(os.environ.get("MW_RSS_URL", "https://feeds.megaphone.fm/modernwisdom"))
    for entry in feed.entries:
        guid = entry.get("id") or entry.get("guid") or entry.get("link")
        title = (entry.get("title") or "").strip()
        link = entry.get("link")
        updated_iso = _to_iso_utc(entry)
        updated_at = updated_iso
        if updated_at is None:
            updated_raw = entry.get("updated") or entry.get("published")
            updated_at = (
                dtparser.parse(updated_raw).astimezone(UTC).isoformat() if updated_raw else None
            )
        publish_raw = entry.get("published") or entry.get("updated")
        publish_date = dtparser.parse(publish_raw).date().isoformat() if publish_raw else None
        audio_url = None
        for enc in entry.get("enclosures") or entry.get("links") or []:
            if (enc.get("type") or "").startswith("audio"):
                audio_url = enc.get("href")
                break
        itunes_dur = entry.get("itunes_duration") or entry.get("itunes:duration")
        duration = itunes_dur if isinstance(itunes_dur, str) else None
        html_chunks = []
        if "content" in entry:
            html_chunks += [part.get("value", "") for part in (entry.content or [])]
        if getattr(entry, "summary", None):
            html_chunks.append(entry.summary)
        if getattr(entry, "subtitle", None):
            html_chunks.append(entry.subtitle)
        description = _strip_html("\n".join([h for h in html_chunks if h]))
        norm_title = re.sub(r"[–—]", "-", title)
        parts = [p.strip() for p in re.split(r"\s*-\s*", norm_title) if p.strip()]
        episode_number = None
        guest = None
        headline = None
        if parts:
            ep_raw = parts[0]
            if ep_raw.startswith("#"):
                ep_digits = ep_raw.lstrip("#")
                episode_number = int(ep_digits) if ep_digits.isdigit() else None
            if len(parts) >= 3:
                guest = parts[1]
                headline = " - ".join(parts[2:])
            elif len(parts) == 2:
                guest = None
                headline = parts[1]
        if not headline:
            first_sentence = (description.split(". ")[0] if description else "").strip()
            headline = first_sentence[:160] or None
        concat = "|".join(
            [
                guid or "",
                title or "",
                description[:5000],
                audio_url or "",
                headline or "",
                guest or "",
                str(episode_number or ""),
                updated_at or "",
            ]
        )
        row_hash = hashlib.sha256(concat.encode("utf-8")).hexdigest()
        yield {
            "id": guid,
            "title": title,
            "guest": guest,
            "episode_number": episode_number,
            "headline": headline,
            "description": description,
            "publish_date": publish_date,
            "audio_url": audio_url,
            "duration": duration,
            "source_url": link,
            "hash": row_hash,
            "updated_at": updated_at,
        }


@dlt.resource(
    name="episode_timestamps", primary_key=("episode_id", "idx"), write_disposition="merge"
)
def rss_episode_timestamps():
    import feedparser

    feed = feedparser.parse(os.environ.get("MW_RSS_URL", "https://feeds.megaphone.fm/modernwisdom"))
    for entry in feed.entries:
        episode_id = entry.get("id") or entry.get("guid") or entry.get("link")
        html_chunks = []
        if "content" in entry:
            html_chunks += [part.get("value", "") for part in (entry.content or [])]
        if getattr(entry, "summary", None):
            html_chunks.append(entry.summary)
        if getattr(entry, "subtitle", None):
            html_chunks.append(entry.subtitle)
        description = _strip_html("\n".join([h for h in html_chunks if h]))
        stamps = _extract_timestamps(description)
        for idx, ts in enumerate(stamps):
            yield {"episode_id": episode_id, "idx": idx, "time": ts["time"], "label": ts["label"]}


@dlt.resource(name="ingest_runs", primary_key="run_id", write_disposition="append")
def ingest_run_record(rows_ingested: int, rows_updated: int, status: str):
    now = datetime.now(UTC).isoformat()
    yield {
        "run_id": f"{int(time.time())}",
        "started_at": now,
        "ended_at": now,
        "rows_ingested": rows_ingested,
        "rows_updated": rows_updated,
        "status": status,
    }


def main():
    db_file = _ensure_duck_dir(DUCK_PATH)
    dataset = "mw"
    prev_total, last_run_started_at = _get_prev_stats(db_file, dataset)
    pipeline = dlt.pipeline(
        pipeline_name="modern_wisdom_pipeline",
        destination=duckdb_dest(credentials=db_file),
        dataset_name=dataset,
    )
    pipeline.run(rss_episodes)
    pipeline.run(rss_episode_timestamps)
    rows_ingested, rows_updated, total_after = _post_run_stats(
        db_file, dataset, last_run_started_at, prev_total
    )
    pipeline.run(
        ingest_run_record(rows_ingested=rows_ingested, rows_updated=rows_updated, status="ok")
    )
    print(f"Loaded (new): {rows_ingested}, Updated (est): {rows_updated}")
    print(f"Total episodes: {total_after}")
    print(f"DuckDB at {db_file}")
