from __future__ import annotations

import contextlib
import re
from typing import Any, cast

import duckdb
from jsonschema import validate

from . import paths
from .specs import (
    CLIP_LINKER_IN,
    CLIP_LINKER_OUT,
    EPISODE_LOCATOR_IN,
    EPISODE_LOCATOR_OUT,
    RAG_SEARCH_IN,
    RAG_SEARCH_OUT,
    SQL_DUCKDB_IN,
    SQL_DUCKDB_OUT,
    TIMELINE_BUILDER_IN,
    TIMELINE_BUILDER_OUT,
)
from .tools import rag_search as _rag_search  # your step-2 rag_search
from .tracing import start_span


def _check(inp: dict[str, Any], schema: dict[str, Any]) -> None:
    validate(instance=inp, schema=schema)


def _ok(out: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    validate(instance=out, schema=schema)
    return out


def _err(msg: str) -> dict[str, Any]:
    return {"error": msg}


# ---- rag_search (wrapper) ----
def rag_search(inp: dict[str, Any]) -> dict[str, Any]:
    # Normalise planner typos: accept "query" alias then validate
    norm = dict(inp)
    if "query" in norm and "question" not in norm:
        norm["question"] = str(norm.pop("query"))

    _check(norm, RAG_SEARCH_IN)

    # Normalise empty, whitespace, or special "all" values
    ep_raw = str(norm.get("episode_id", "") or "").strip().lower()
    if ep_raw in {"", "all", "corpus", "none", "null"}:
        norm["episode_id"] = None

    # If episode_id is empty or missing, treat as corpus search (unless explicitly "episode")
    if norm.get("episode_id") is None and norm.get("scope") != "episode":
        norm["scope"] = "corpus"

    with start_span(
        "tool.rag_search",
        kind="RETRIEVER",
        attrs={"scope": norm["scope"], "top_k": int(norm["top_k"])},
    ):
        # Primary attempt
        out = _rag_search(
            cast(
                Any,
                {
                    "question": norm["question"],
                    "episode_id": norm.get("episode_id"),
                    "top_k": int(norm["top_k"]),
                    "scope": norm["scope"],
                },
            )
        )

        # Ensure shape and annotate scope/fallback
        out = {
            **out,
            "fallback_used": bool(out.get("fallback_used", False)),
            "scope": out.get("scope", norm["scope"]),
        }

        # Escalate to corpus if nothing retrieved and scope != corpus
        if not out.get("retrieved") and norm["scope"] != "corpus":
            out2 = _rag_search(
                cast(
                    Any,
                    {
                        "question": norm["question"],
                        "episode_id": None,
                        "top_k": int(norm["top_k"]),
                        "scope": "corpus",
                    },
                )
            )
            out = {
                **out2,
                "fallback_used": True,
                "scope": "corpus",
            }

        return _ok(out, RAG_SEARCH_OUT)


YEAR_RE = re.compile(r"\b(20\d{2})\b")


def episode_locator(inp: dict[str, Any]) -> dict[str, Any]:
    _check(inp, EPISODE_LOCATOR_IN)
    q = inp["query"].strip()
    limit = int(inp["limit"])
    years = [int(y) for y in YEAR_RE.findall(q)]  # e.g., [2021, 2024]

    with start_span(
        "tool.episode_locator", kind="RETRIEVER", attrs={"query": q, "limit": limit, "years": years}
    ):
        con = duckdb.connect(paths.DUCKDB_PATH.as_posix(), read_only=True)
        try:
            # Pick episodes table
            candidate_schemas = ["mw", "mw_staging", "main"]
            table_name = None
            for sch in candidate_schemas:
                row = con.execute(
                    "select count(*) from information_schema.tables "
                    "where table_schema=? and table_name='episodes'",
                    [sch],
                ).fetchone()
                if row and row[0]:
                    table_name = f"{sch}.episodes"
                    break
            if not table_name:
                return _err("No episodes table found in schemas: mw, mw_staging, main")

            # Discover available columns
            cols = {
                r[0]
                for r in con.execute(
                    "select column_name from information_schema.columns "
                    "where table_schema=? and table_name='episodes'",
                    [table_name.split(".", 1)[0]],
                ).fetchall()
            }

            # Metadata search (ILIKE) with optional year bias
            search_candidates = ["title", "guest", "headline", "description", "summary"]
            search_cols = [c for c in search_candidates if c in cols]

            where_parts: list[str] = []
            params: list[Any] = []

            if search_cols:
                likes = " OR ".join([f"{c} ILIKE ?" for c in search_cols])
                where_parts.append(f"({likes})")
                params.extend([f"%{q}%"] * len(search_cols))

            date_col = "publish_date" if "publish_date" in cols else None
            if date_col and years:
                year_list = ", ".join(str(y) for y in sorted(set(years)))
                where_parts.append(
                    f"EXTRACT(year FROM try_cast({date_col} AS DATE)) IN ({year_list})"
                )

            where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

            select_parts = ["id AS episode_id"]
            for c in ("title", "guest", "headline", "description"):
                select_parts.append(c if c in cols else f"CAST('' AS TEXT) AS {c}")

            order_sql = ""
            if date_col:
                if years:
                    # Prioritise proximity to the first mentioned year, then recency
                    pivot_year = years[0]
                    order_sql = (
                        f"ORDER BY ABS(EXTRACT(year FROM try_cast({date_col} AS DATE)) - {pivot_year}) ASC, "
                        f"{date_col} DESC"
                    )
                else:
                    order_sql = f"ORDER BY {date_col} DESC"

            sql_meta = f"""
                SELECT {", ".join(select_parts)}
                FROM {table_name}
                {where_sql}
                {order_sql}
                LIMIT ?
            """
            rows_meta = con.execute(sql_meta, [*params, limit]).fetchall()
            if rows_meta:
                desc = [d[0] for d in (con.description or [])]
                idx = {n: i for i, n in enumerate(desc)}

                def g(r, n, d=""):
                    i = idx.get(n)
                    return d if i is None or r[i] is None else r[i]

                episodes = [
                    {
                        "episode_id": str(g(r, "episode_id", "")),
                        "title": str(g(r, "title", "")),
                        "guest": str(g(r, "guest", "")),
                        "headline": str(g(r, "headline", "")),
                        "description": str(g(r, "description", "")),
                    }
                    for r in rows_meta
                ]
                return _ok({"episodes": episodes}, EPISODE_LOCATOR_OUT)

            # Content fallback (partitioned chunks)
            # Expected: data/chunks/sentence_bound/episode_id=<UUID>/part-*.parquet
            chunks_glob = (
                paths.CHUNKS_DIR / "sentence_bound" / "episode_id=*" / "part-*.parquet"
            ).as_posix()

            # Detect the text column in chunk parquet
            cols_df = con.execute(
                "SELECT * FROM read_parquet(?, hive_partitioning=1, union_by_name=1) LIMIT 0",
                [chunks_glob],
            ).fetchdf()
            avail_cols = set(cols_df.columns)
            text_candidates = [
                "text",
                "content",
                "segment_text",
                "segment",
                "transcript",
                "raw_text",
                "body",
                "snippet",
                "utterance",
                "line",
                "value",
            ]
            text_col = next((c for c in text_candidates if c in avail_cols), None)

            # Tokenisation: OR over non-numeric informative tokens
            raw_tokens = re.findall(r"[A-Za-z0-9']+", q.lower())
            stop = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "to",
                "of",
                "on",
                "in",
                "vs",
                "view",
                "views",
                "about",
            }
            tokens = [t for t in raw_tokens if t.isalpha() and len(t) >= 3 and t not in stop]
            if not tokens:
                tokens = ["discipline"]

            ep_ids: list[str] = []
            if text_col:
                like_pred = " OR ".join([f"lower({text_col}) LIKE ?"] * len(tokens))
                like_args = [f"%{t}%" for t in tokens]

                sql_hits = f"""
                    WITH hits AS (
                        SELECT COALESCE(episode_id, '') AS episode_id, COUNT(*) AS hits
                        FROM read_parquet('{chunks_glob}', hive_partitioning=1, union_by_name=1)
                        WHERE {like_pred}
                        GROUP BY 1
                    )
                    SELECT h.episode_id, h.hits
                    FROM hits h
                    ORDER BY h.hits DESC
                    LIMIT ?
                """
                hit_rows = con.execute(sql_hits, [*like_args, limit * 3]).fetchall()
                ep_ids = [str(r[0]) for r in hit_rows if r and r[0]]

            episodes: list[dict[str, Any]] = []
            if ep_ids:
                ph = ", ".join(["?"] * len(ep_ids))
                order_bias = ""
                if date_col and years:
                    pivot_year = years[0]
                    order_bias = (
                        f"ORDER BY ABS(EXTRACT(year FROM try_cast({date_col} AS DATE)) - {pivot_year}) ASC, "
                        f"{date_col} DESC"
                    )
                elif date_col:
                    order_bias = f"ORDER BY {date_col} DESC"

                sql_join = f"""
                    SELECT
                        e.id AS episode_id,
                        {"e.title" if "title" in cols else "CAST('' AS TEXT) AS title"},
                        {"e.guest" if "guest" in cols else "CAST('' AS TEXT) AS guest"},
                        {"e.headline" if "headline" in cols else "CAST('' AS TEXT) AS headline"},
                        {"e.description" if "description" in cols else "CAST('' AS TEXT) AS description"}
                    FROM {table_name} e
                    WHERE e.id IN ({ph})
                    {order_bias}
                    LIMIT ?
                """
                rows_join = con.execute(sql_join, [*ep_ids, limit]).fetchall()
                if rows_join:
                    desc = [d[0] for d in (con.description or [])]
                    idx = {n: i for i, n in enumerate(desc)}

                    def gj(r, n, d=""):
                        i = idx.get(n)
                        return d if i is None or r[i] is None else r[i]

                    episodes = [
                        {
                            "episode_id": str(gj(r, "episode_id", "")),
                            "title": str(gj(r, "title", "")),
                            "guest": str(gj(r, "guest", "")),
                            "headline": str(gj(r, "headline", "")),
                            "description": str(gj(r, "description", "")),
                        }
                        for r in rows_join
                    ]

            if episodes:
                return _ok({"episodes": episodes[:limit]}, EPISODE_LOCATOR_OUT)

            # Last resort: latest episodes
            if "publish_date" in cols:
                sql_latest = f"""
                    SELECT id AS episode_id,
                           {"title" if "title" in cols else "CAST('' AS TEXT) AS title"},
                           {"guest" if "guest" in cols else "CAST('' AS TEXT) AS guest"},
                           {"headline" if "headline" in cols else "CAST('' AS TEXT) AS headline"},
                           {"description" if "description" in cols else "CAST('' AS TEXT) AS description"}
                    FROM {table_name}
                    ORDER BY {date_col if date_col else "id"} DESC
                    LIMIT ?
                """
                rows_latest = con.execute(sql_latest, [limit]).fetchall()
                if rows_latest:
                    desc = [d[0] for d in (con.description or [])]
                    idx = {n: i for i, n in enumerate(desc)}

                    def gl(r, n, d=""):
                        i = idx.get(n)
                        return d if i is None or r[i] is None else r[i]

                    episodes = [
                        {
                            "episode_id": str(gl(r, "episode_id", "")),
                            "title": str(gl(r, "title", "")),
                            "guest": str(gl(r, "guest", "")),
                            "headline": str(gl(r, "headline", "")),
                            "description": str(gl(r, "description", "")),
                        }
                        for r in rows_latest
                    ]
                    return _ok({"episodes": episodes}, EPISODE_LOCATOR_OUT)

            return _ok({"episodes": []}, EPISODE_LOCATOR_OUT)

        except Exception as e:
            return _err(str(e))
        finally:
            with contextlib.suppress(Exception):
                con.close()


# ---- timeline_builder (cluster and summarise short range text) ----
def timeline_builder(inp: dict[str, Any]) -> dict[str, Any]:
    _check(inp, TIMELINE_BUILDER_IN)
    segs = sorted(inp["segments"], key=lambda s: (s["episode_id"], s["start_ts"]))
    timeline = []
    if not segs:
        return _ok({"timeline": []}, TIMELINE_BUILDER_OUT)
    with start_span("tool.timeline_builder", kind="CHAIN", attrs={"segments": len(segs)}):
        # Greedy merge contiguous/nearby segments per episode
        cur = {
            "episode_id": segs[0]["episode_id"],
            "t0": segs[0]["start_ts"],
            "t1": segs[0]["end_ts"],
            "buf": [segs[0]["text"]],
        }
        for s in segs[1:]:
            if s["episode_id"] == cur["episode_id"] and s["start_ts"] <= cur["t1"] + 15.0:
                cur["t1"] = max(cur["t1"], s["end_ts"])
                cur["buf"].append(s["text"])
            else:
                summary = " ".join(" ".join(cur["buf"]).split()[:60]).strip()
                timeline.append(
                    {
                        "episode_id": cur["episode_id"],
                        "t0": cur["t0"],
                        "t1": cur["t1"],
                        "summary": summary,
                    }
                )
                cur = {
                    "episode_id": s["episode_id"],
                    "t0": s["start_ts"],
                    "t1": s["end_ts"],
                    "buf": [s["text"]],
                }
        summary = " ".join(" ".join(cur["buf"]).split()[:60]).strip()
        timeline.append(
            {"episode_id": cur["episode_id"], "t0": cur["t0"], "t1": cur["t1"], "summary": summary}
        )
        return _ok({"timeline": timeline}, TIMELINE_BUILDER_OUT)


# ---- sql_duckdb (strict read-only, ACL) ----
def sql_duckdb(inp: dict[str, Any]) -> dict[str, Any]:
    _check(inp, SQL_DUCKDB_IN)
    sql = inp["sql"].strip().rstrip(";")
    # Deny writes and DDL
    forbidden = ("drop ", "create ", "alter ", "insert ", "update ", "delete ", "copy ", "pragma ")
    low = sql.lower() + " "
    if any(tok in low for tok in forbidden) or "information_schema" in low or "pg_catalog" in low:
        raise PermissionError("sql_duckdb: write/ddl/system queries are not allowed.")
    with start_span("tool.sql_duckdb", kind="TOOL", attrs={"sql_preview": sql[:200]}):
        con = duckdb.connect(paths.DUCKDB_PATH.as_posix(), read_only=True)
        rows = con.execute(sql).fetchmany(1000)
        return _ok({"rows": rows}, SQL_DUCKDB_OUT)


# ---- clip_linker (fetch audio URL from RSS feed data) ----
def clip_linker(inp: dict[str, Any]) -> dict[str, Any]:
    _check(inp, CLIP_LINKER_IN)
    ep = inp["episode_id"]
    ts = float(inp["timestamp"])
    with start_span("tool.clip_linker", kind="TOOL", attrs={"episode_id": ep, "ts": ts}):
        # Query DuckDB for the audio URL from RSS feed
        con = duckdb.connect(paths.DUCKDB_PATH.as_posix(), read_only=True)
        try:
            # Try to find episodes table in mw or mw_staging schema
            candidate_schemas = ["mw", "mw_staging", "main"]
            table_name = None
            for sch in candidate_schemas:
                row = con.execute(
                    "select count(*) from information_schema.tables "
                    "where table_schema=? and table_name='episodes'",
                    [sch],
                ).fetchone()
                if row and row[0]:
                    table_name = f"{sch}.episodes"
                    break

            if not table_name:
                # No episodes table found
                return _ok(
                    {"url": "", "timestamp": int(ts), "error": "Episodes table not found"},
                    CLIP_LINKER_OUT,
                )

            # Check available columns
            cols = {
                r[0]
                for r in con.execute(
                    "select column_name from information_schema.columns "
                    "where table_schema=? and table_name='episodes'",
                    [table_name.split(".", 1)[0]],
                ).fetchall()
            }

            # Query for audio_url (and other metadata)
            audio_col = "audio_url" if "audio_url" in cols else None
            title_col = "title" if "title" in cols else None
            guest_col = "guest" if "guest" in cols else None

            if audio_col:
                select_parts = [f"{audio_col} AS audio_url"]
                if title_col:
                    select_parts.append(f"{title_col} AS title")
                if guest_col:
                    select_parts.append(f"{guest_col} AS guest")

                sql = f"""
                    SELECT {", ".join(select_parts)}
                    FROM {table_name}
                    WHERE id = ?
                    LIMIT 1
                """
                row = con.execute(sql, [ep]).fetchone()

                if row and row[0]:
                    result = {
                        "url": str(row[0]),
                        "timestamp": int(ts),
                    }
                    if title_col and len(row) > 1 and row[1]:
                        result["title"] = str(row[1])
                    if guest_col and len(row) > 2 and row[2]:
                        result["guest"] = str(row[2])
                    return _ok(result, CLIP_LINKER_OUT)

            # Fallback: no audio_url found
            return _ok(
                {"url": "", "timestamp": int(ts), "error": "Audio URL not available"},
                CLIP_LINKER_OUT,
            )

        except Exception as e:
            # Fallback on error
            return _ok(
                {"url": "", "timestamp": int(ts), "error": f"Database error: {str(e)}"},
                CLIP_LINKER_OUT,
            )
        finally:
            with contextlib.suppress(Exception):
                con.close()
