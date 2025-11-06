from __future__ import annotations

import contextlib
import logging
import os
import random
from collections import defaultdict
from typing import cast

import chainlit as cl
import duckdb

from modern_wisdom_rag_pipeline import paths
from modern_wisdom_rag_pipeline.agent import run_deep_agent

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Optional: configure defaults via environment variables
DEFAULT_PROVIDER = os.getenv("MW_LLM_PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MW_LLM_MODEL", "gpt-4o-mini")
DEFAULT_SCOPE = os.getenv("MW_SCOPE", "corpus")
DEFAULT_TOP_K = int(os.getenv("MW_TOP_K", "8"))
DEFAULT_EPISODE_ID = os.getenv("MW_EPISODE_ID", "")

HISTORY_TURNS = 4  # Keep the last 4 message pairs (8 turns)

THINKING_MESSAGES = [
    "Pondering deeply",
    "Consulting my neural networks",
    "Summoning wisdom",
    "Diving into knowledge",
    "Processing thoughts",
    "Gathering insights",
    "Searching the archives",
    "Exploring episodes",
    "Connecting the dots",
]


def _format_timestamp(seconds: float) -> str:
    """Format seconds to human-friendly time format (HH:MM:SS or MM:SS)."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def _format_timestamp_range(start: float, end: float) -> str:
    """Format timestamp range in human-friendly format."""
    start_str = _format_timestamp(start)
    end_str = _format_timestamp(end)
    return f"{start_str}–{end_str}"


@cl.set_starters
async def get_starters(user=None, locale=None):
    return [
        cl.Starter(
            label="Optimize my morning routine",
            message="What does the science say about structuring a morning routine for peak performance? I want to know about caffeine timing, exercise, and mental clarity.",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Lose a few pounds safely",
            message="I need to lose a few pounds. What does the science say about sustainable weight loss—diet, exercise, or both? Any insights from health experts on the podcast?",
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Coffee timing for energy",
            message="When should I drink my first coffee after waking up to avoid an afternoon crash and optimize energy levels?",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="High performer habits",
            message="What daily habits and routines do elite performers have in common? How can I apply them to my own life?",
            icon="/public/learn.svg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])


# Episode metadata cache (in-memory, per-process)
_episode_metadata_cache: dict[str, dict[str, str]] = {}
_episode_table_info: dict[str, tuple[str | None, set[str]]] = {}  # schema -> (table_name, columns)


def _get_episode_table_info() -> tuple[str | None, set[str]]:
    """Get episodes table name and columns (cached)."""
    global _episode_table_info

    cache_key = "episodes"
    if cache_key in _episode_table_info:
        return _episode_table_info[cache_key]

    try:
        con = duckdb.connect(paths.DUCKDB_PATH.as_posix(), read_only=True)
        try:
            # Find episodes table
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
                _episode_table_info[cache_key] = (None, set())
                return None, set()

            # Check available columns
            cols = {
                r[0]
                for r in con.execute(
                    "select column_name from information_schema.columns "
                    "where table_schema=? and table_name='episodes'",
                    [table_name.split(".", 1)[0]],
                ).fetchall()
            }

            _episode_table_info[cache_key] = (table_name, cols)
            return table_name, cols
        finally:
            with contextlib.suppress(Exception):
                con.close()
    except Exception:
        _episode_table_info[cache_key] = (None, set())
        return None, set()


def _get_episode_metadata_batch(episode_ids: list[str]) -> dict[str, dict[str, str]]:
    """Batch fetch episode metadata for multiple episodes (with caching)."""
    global _episode_metadata_cache
    logger = logging.getLogger(__name__)

    # Separate cached and uncached episodes
    result = {}
    uncached_ids = []

    for ep_id in episode_ids:
        if ep_id in _episode_metadata_cache:
            result[ep_id] = _episode_metadata_cache[ep_id]
        else:
            uncached_ids.append(ep_id)

    if not uncached_ids:
        return result

    try:
        table_name, cols = _get_episode_table_info()
        if not table_name or not cols:
            # Return empty dicts for uncached episodes
            for ep_id in uncached_ids:
                result[ep_id] = {}
                _episode_metadata_cache[ep_id] = {}
            return result

        # Build query
        select_parts = []
        if "audio_url" in cols:
            select_parts.append("audio_url")
        if "title" in cols:
            select_parts.append("title")
        if "guest" in cols:
            select_parts.append("guest")
        if "publish_date" in cols:
            select_parts.append("publish_date")

        if not select_parts:
            for ep_id in uncached_ids:
                result[ep_id] = {}
                _episode_metadata_cache[ep_id] = {}
            return result

        # Batch query with IN clause
        placeholders = ",".join(["?"] * len(uncached_ids))
        sql = f"""
            SELECT id, {", ".join(select_parts)}
            FROM {table_name}
            WHERE id IN ({placeholders})
        """

        con = duckdb.connect(paths.DUCKDB_PATH.as_posix(), read_only=True)
        try:
            rows = con.execute(sql, uncached_ids).fetchall()

            # Process results
            for row in rows:
                ep_id = str(row[0])
                meta = {}
                idx = 1
                if "audio_url" in cols:
                    meta["audio_url"] = str(row[idx]) if row[idx] else ""
                    idx += 1
                if "title" in cols:
                    meta["title"] = str(row[idx]) if row[idx] else ""
                    idx += 1
                if "guest" in cols:
                    meta["guest"] = str(row[idx]) if row[idx] else ""
                    idx += 1
                if "publish_date" in cols:
                    meta["publish_date"] = str(row[idx]) if row[idx] else ""

                result[ep_id] = meta
                _episode_metadata_cache[ep_id] = meta

            # Cache empty results for episodes not found
            for ep_id in uncached_ids:
                if ep_id not in result:
                    result[ep_id] = {}
                    _episode_metadata_cache[ep_id] = {}

        finally:
            with contextlib.suppress(Exception):
                con.close()
    except Exception as e:
        logger.error(f"Error batch fetching metadata: {e}")
        # Return empty dicts for failed episodes
        for ep_id in uncached_ids:
            if ep_id not in result:
                result[ep_id] = {}

    return result


def _get_episode_metadata(episode_id: str) -> dict[str, str]:
    """Fetch episode metadata (single episode, uses batch internally)."""
    batch_result = _get_episode_metadata_batch([episode_id])
    return batch_result.get(episode_id, {})


def _trim_history(hist: list[dict], keep_pairs: int = HISTORY_TURNS) -> list[dict]:
    # Keep the last N user/assistant pairs (2N turns)
    turns = []
    for h in hist:
        if h.get("role") in ("user", "assistant"):
            turns.append(h)
    return turns[-(2 * keep_pairs) :]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    q = (message.content or "").strip()
    if not q:
        await cl.Message(content="Please type a question.").send()
        return

    history = cast(list[dict], cl.user_session.get("history") or [])
    history = _trim_history(history)

    thinking_msg = random.choice(THINKING_MESSAGES)

    # Create streaming message for the answer
    msg = cl.Message(content="")

    try:
        async with cl.Step(name=thinking_msg, type="tool") as step:
            out = run_deep_agent(
                question=q,
                episode_id=DEFAULT_EPISODE_ID,
                top_k=DEFAULT_TOP_K,
                scope=DEFAULT_SCOPE,
                provider=DEFAULT_PROVIDER,
                model_id=DEFAULT_MODEL_ID,
                step_cap=6,
                per_step_timeout_s=8.0,
                chat_history=history,
            )
            step.output = "Analysis complete"

        result = out.get("result") or {}
        answer = (result.get("answer") or "").strip()
        evidence = result.get("evidence") or []

        # Update and trim history for the next follow-up
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        cl.user_session.set("history", _trim_history(history))

        # Group evidence by episode_id to handle duplicates
        episode_clips = defaultdict(list)
        for ev in evidence:
            ep = str(ev.get("episode_id", ""))
            if ep:
                episode_clips[ep].append(
                    {
                        "start_ts": float(ev.get("start_ts", 0.0)),
                        "end_ts": float(ev.get("end_ts", 0.0)),
                    }
                )

        # Limit to top 5 episodes
        episode_items = list(episode_clips.items())[:5]

        # Batch fetch all episode metadata at once (much faster than individual queries)
        episode_ids = [ep for ep, _ in episode_items]
        metadata_batch = _get_episode_metadata_batch(episode_ids)

        # Build evidence display with audio players
        evidence_elements = []
        evidence_lines = []

        if episode_items:
            for i, (ep, clips) in enumerate(episode_items, 1):
                # Get metadata from batch result
                meta = metadata_batch.get(ep, {})
                audio_url = meta.get("audio_url", "")
                title = meta.get("title", "")
                guest = meta.get("guest", "")
                publish_date = meta.get("publish_date", "")

                # Build display text
                if title and guest:
                    display_text = f"**{title}** with {guest}"
                elif title:
                    display_text = f"**{title}**"
                else:
                    # Show shortened episode ID when no title is available
                    short_ep = ep[:8] if len(ep) > 8 else ep
                    display_text = f"**Episode** `{short_ep}...`"

                # Format publish date
                date_text = f" ({publish_date})" if publish_date else ""

                # Format all timestamps for this episode
                timestamp_ranges = [
                    _format_timestamp_range(clip["start_ts"], clip["end_ts"]) for clip in clips
                ]

                # Join multiple timestamps with ", " or " and " for the last one
                if len(timestamp_ranges) == 1:
                    timestamps_text = f"Relevant clip at {timestamp_ranges[0]}"
                elif len(timestamp_ranges) == 2:
                    timestamps_text = (
                        f"Relevant clips at {timestamp_ranges[0]} and {timestamp_ranges[1]}"
                    )
                else:
                    # More than 2: "clip1, clip2, and clip3"
                    timestamps_text = f"Relevant clips at {', '.join(timestamp_ranges[:-1])}, and {timestamp_ranges[-1]}"

                if audio_url:
                    # Add audio player element (one per episode)
                    if title:
                        audio_name = f"{i}. {title}"
                    else:
                        short_ep = ep[:8] if len(ep) > 8 else ep
                        audio_name = f"{i}. Episode {short_ep}..."
                    evidence_elements.append(
                        cl.Audio(
                            name=audio_name,
                            url=audio_url,
                            display="inline",
                        )
                    )
                    evidence_lines.append(f"{i}. {display_text}{date_text} — {timestamps_text}")
                else:
                    evidence_lines.append(
                        f"{i}. {display_text}{date_text} — {timestamps_text} (Audio not available)"
                    )

            evidence_md = "\n".join(evidence_lines)
        else:
            evidence_md = "_No matching clips found._"

        # Stream the answer with evidence (word-by-word for better performance)
        await msg.send()

        # Stream answer section
        answer_section = f"**Answer**\n\n{answer}\n\n"
        await msg.stream_token(answer_section)

        # Small delay to make streaming visible but not too slow
        import asyncio

        await asyncio.sleep(0.05)

        # Stream evidence section
        evidence_section = f"**Evidence**\n\n{evidence_md}"
        await msg.stream_token(evidence_section)

        # Finalize the message
        await msg.update()

        # Send audio elements if available (separate message for compatibility)
        if evidence_elements:
            await cl.Message(
                content="🎧 **Episode Audio Players**",
                elements=evidence_elements,
            ).send()

        # Attach debug artifacts if they exist (disabled in production by default)
        # Set MW_DEBUG=true to enable debug artifacts
        if os.getenv("MW_DEBUG", "false").lower() == "true":
            debug_elements = []
            plan_path = paths.DATA_DIR / "agents" / "plans" / "latest_plan.json"
            result_path = paths.DATA_DIR / "evals" / "agents" / "latest_result.json"

            if plan_path.exists():
                debug_elements.append(
                    cl.File(name="latest_plan.json", path=str(plan_path.resolve()))
                )
            if result_path.exists():
                debug_elements.append(
                    cl.File(name="latest_result.json", path=str(result_path.resolve()))
                )

            if debug_elements:
                await cl.Message(
                    content="Debug artifacts attached.",
                    elements=debug_elements,
                ).send()

    except Exception as e:
        import traceback

        error_msg = f"**Error**: {e}\n\n```\n{traceback.format_exc()}\n```"
        await msg.send()
        await msg.stream_token(error_msg)
        await msg.update()
