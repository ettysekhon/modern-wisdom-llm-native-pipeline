from pathlib import Path

import duckdb


def load_episode_meta(episode_id: str, db_path: Path) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT id as episode_id, title, guest, publish_date, episode_number, duration, headline, description
            FROM mw.episodes
            WHERE id = ?
            LIMIT 1
        """,
            [episode_id],
        ).df()
    finally:
        con.close()
    return {} if df.empty else df.iloc[0].to_dict()


def enrich_chunk_rows(rows: list[dict], episode_meta: dict) -> list[dict]:
    if not episode_meta:
        return rows
    out = []
    for r in rows:
        r = dict(r)
        r.setdefault("episode_title", episode_meta.get("title"))
        r.setdefault("guest", episode_meta.get("guest"))
        r.setdefault("publish_date", episode_meta.get("publish_date"))
        r.setdefault("episode_number", episode_meta.get("episode_number"))
        r.setdefault("duration", episode_meta.get("duration"))
        r.setdefault("headline", episode_meta.get("headline"))
        r.setdefault("description", episode_meta.get("description"))
        out.append(r)
    return out
