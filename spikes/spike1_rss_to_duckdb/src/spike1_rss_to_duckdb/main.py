import os
import time
from datetime import UTC, datetime
from pathlib import Path

import dlt
import duckdb
from dlt.destinations import duckdb as duckdb_dest

DUCK_PATH = os.environ.get("DUCK_PATH", "./data/duckdb/modern_wisdom.duckdb")


def _ensure_duck_dir(path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def _get_prev_stats(db_file: str, dataset: str):
    """Return (prev_total, last_run_started_at_utc_iso or None)."""
    if not Path(db_file).exists():
        return 0, None
    con = duckdb.connect(db_file, read_only=False)
    try:
        # episodes may not exist on first run
        prev_total = con.execute(f"""
            SELECT COUNT(*) FROM {dataset}.episodes
        """).fetchone()[0]
    except Exception:
        prev_total = 0

    last_run_started_at = None
    try:
        last_row = con.execute(f"""
            SELECT started_at
            FROM {dataset}.ingest_runs
            ORDER BY started_at DESC
            LIMIT 1
        """).fetchone()
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
        # episodes that changed since the previous run
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


@dlt.resource(
    name="episodes",
    primary_key="id",
    write_disposition="merge",
)
def rss_episodes(
    cursor=None,
):
    if cursor is None:
        cursor = dlt.sources.incremental("updated_at", initial_value="2000-01-01T00:00:00Z")

    import hashlib

    import feedparser
    from dateutil import parser as dtparser

    feed = feedparser.parse(os.environ.get("MW_RSS_URL", "https://feeds.megaphone.fm/modernwisdom"))

    for entry in feed.entries:
        guid = entry.get("id") or entry.get("guid") or entry.get("link")
        title = entry.get("title") or ""
        link = entry.get("link")

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

        concat = "|".join(
            [
                guid or "",
                title or "",
                (entry.get("summary") or entry.get("subtitle") or "")[:5000],
                audio_url or "",
                updated_at or "",
            ]
        )
        row_hash = hashlib.sha256(concat.encode("utf-8")).hexdigest()

        yield {
            "id": guid,
            "title": title,
            "guest": title.split(" - ")[-1].strip() if " - " in title else None,
            "publish_date": publish_date,
            "audio_url": audio_url,
            "duration": duration,
            "source_url": link,
            "hash": row_hash,
            "updated_at": updated_at,
        }


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

    rows_ingested, rows_updated, total_after = _post_run_stats(
        db_file, dataset, last_run_started_at, prev_total
    )

    pipeline.run(
        ingest_run_record(rows_ingested=rows_ingested, rows_updated=rows_updated, status="ok")
    )

    print(f"Loaded (new): {rows_ingested}, Updated (est): {rows_updated}")
    print(f"Total episodes: {total_after}")
    print(f"DuckDB at {db_file}")
