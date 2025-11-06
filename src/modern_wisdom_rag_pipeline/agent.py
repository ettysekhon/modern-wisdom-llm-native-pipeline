from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from jsonschema import ValidationError, validate

from . import paths
from .generator import generate_answer
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
from .tracing import start_span
from .tools_constrained import (
    clip_linker,
    episode_locator,
    rag_search,
    sql_duckdb,
    timeline_builder,
)

# ---------------------------
# Tool registry (names must match planner prompt exactly)
# ---------------------------
TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "rag_search": rag_search,
    "episode_locator": episode_locator,
    "timeline_builder": timeline_builder,
    "sql_duckdb": sql_duckdb,
    "clip_linker": clip_linker,
}

# ---------------------------
# Task output schema
# ---------------------------
TASK_OUT_SCHEMA = {
    "type": "object",
    "required": ["answer", "evidence"],
    "properties": {
        "answer": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["episode_id", "start_ts", "end_ts"],
                "properties": {
                    "episode_id": {"type": "string"},
                    "start_ts": {"type": "number"},
                    "end_ts": {"type": "number"},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": False,
}

# ---------------------------
# System prompt used by planner
# ---------------------------
SYS_PROMPT = (
    "You are a constrained agent. Decide the next step as JSON ONLY.\n"
    "Allowed tools: rag_search, episode_locator, timeline_builder, sql_duckdb, clip_linker.\n"
    "ARGUMENT SHAPES (must match exactly):\n"
    "- rag_search.args = {\"question\": string, \"episode_id\": string, \"top_k\": int, \"scope\": \"episode\"|\"corpus\"|\"auto\"}\n"
    "- episode_locator.args = {\"query\": string, \"limit\": int}\n"
    "- timeline_builder.args = {\"segments\": [{\"episode_id\": string, \"start_ts\": number, \"end_ts\": number, \"text\": string}]}\n"
    "- sql_duckdb.args = {\"sql\": string}\n"
    "- clip_linker.args = {\"episode_id\": string, \"timestamp\": number}\n"
    "Return ONLY one of:\n"
    "1) {\"tool\": \"<name>\", \"args\": {...}}\n"
    "2) {\"final\": {\"answer\": string, \"evidence\": [{\"episode_id\": string, \"start_ts\": number, \"end_ts\": number}]}}\n"
    "Never invent fields; follow the shapes exactly."
    "Never paste raw transcript text as the final answer. Always summarise in your own words."
    "The \"final.answer\" must be 2–8 sentences; supporting clips go only in \"final.evidence\"."
)


# ---------------------------
# Planner call + robust logging
# ---------------------------
def _planner_decide(
    question: str,
    scratch: list[dict[str, Any]],
    model_id: str,
    provider: str,
    history_text: str = "",
) -> dict[str, Any]:
    """
    Calls the LLM planner and logs raw output + parse failures.
    Returns either a parsed plan dict, or a sentinel dict that triggers fallback.
    """
    plans_dir = paths.DATA_DIR / "agents" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    llm_log = plans_dir / "latest_llm.jsonl"
    err_log = plans_dir / "latest_errors.log"

    # Keep state concise in prompt
    ctx = json.dumps(
        {"question": question, "history": history_text, "scratch": scratch[-6:]}, ensure_ascii=False
    )[:7000]
    user_msg = f"State:\n{ctx}\n\nDecide next step."

    out = generate_answer(
        question=user_msg,
        context=SYS_PROMPT,
        citations=[],
        provider=provider,
        model_id=model_id,
    ).strip()

    # Always log raw output for observability
    try:
        with llm_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "raw": out}) + "\n")
    except Exception:
        pass

    # Parse as JSON; on failure, return sentinel to enable fallback
    try:
        return json.loads(out)
    except Exception as e:
        try:
            with err_log.open("a", encoding="utf-8") as f:
                f.write(f"[{time.time()}] JSON parse error: {e}\n{out}\n\n")
        except Exception:
            pass
        return {"__parse_error__": True, "raw": out}


# ---------------------------
# Deterministic fallback pipeline (always returns something demoable)
# ---------------------------
def fallback_pipeline(question: str, top_k: int = 8) -> dict[str, Any]:
    # RAG search corpus-wide (robust default)
    rag = rag_search(
        {
            "question": question,
            "episode_id": None,
            "top_k": int(top_k),
            "scope": "corpus",
        }
    )
    retrieved = rag.get("retrieved", [])[:top_k]

    # Build a compact timeline summary
    segs = [
        {
            "episode_id": r.get("episode_id", ""),
            "start_ts": float(r.get("start_ts", 0.0)),
            "end_ts": float(r.get("end_ts", 0.0)),
            "text": r.get("text", ""),
        }
        for r in retrieved
        if r.get("episode_id")
    ]
    tl = timeline_builder({"segments": segs}) if segs else {"timeline": []}

    # Link top clips
    links = []
    for r in retrieved[:3]:
        try:
            link = clip_linker(
                {
                    "episode_id": r.get("episode_id", ""),
                    "timestamp": float(r.get("start_ts", 0.0)),
                }
            )
            if "url" in link:
                links.append({"episode_id": r.get("episode_id", ""), "url": link["url"]})
        except Exception:
            continue

    # Compose final result
    answer = "\n".join([t["summary"] for t in tl.get("timeline", [])]) if tl.get("timeline") else ""
    evidence = [
        {
            "episode_id": r.get("episode_id", ""),
            "start_ts": r.get("start_ts", 0.0),
            "end_ts": r.get("end_ts", 0.0),
        }
        for r in retrieved
        if r.get("episode_id")
    ]

    return {
        "answer": answer or "Retrieved and summarized top clips.",
        "evidence": evidence,
        "links": links,
    }


def _format_history(hist: list[dict[str, str]] | None, max_chars: int = 1200) -> str:
    """
    Convert chat history to a compact string.
    Trim to avoid exceeding the context window.
    """
    if not hist:
        return ""
    lines = []
    for h in hist[-8:]:  # Last 8 turns maximum
        role = h.get("role", "user").capitalize()
        content = (h.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "..." + text[-max_chars:]
    return text


# ---------------------------
# Deep agent runner
# ---------------------------
def run_deep_agent(
    question: str,
    episode_id: str,
    top_k: int,
    scope: str,
    provider: str,
    model_id: str,
    step_cap: int = 6,
    per_step_timeout_s: float = 8.0,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    scratch: list[dict[str, Any]] = []
    history_text = _format_history(chat_history)

    with start_span(
        "agent.run",
        kind="CHAIN",
        attrs={"step_cap": step_cap, "timeout_per_step_s": per_step_timeout_s, "scope": scope},
    ):
        for step in range(step_cap):
            with start_span("agent.plan", kind="LLM", attrs={"step": step}):
                plan = _planner_decide(
                    question=question,
                    scratch=scratch,
                    model_id=model_id,
                    provider=provider,
                    history_text=history_text,
                )

            # Planner produced a final answer
            if isinstance(plan, dict) and "final" in plan:
                result = plan["final"]
                try:
                    validate(result, TASK_OUT_SCHEMA)
                except ValidationError:
                    result = {"answer": str(result), "evidence": []}
                return {"result": result, "scratch": scratch, "steps": step + 1}

            # Planner failed to produce valid JSON; use fallback
            if plan.get("__parse_error__"):
                fb = fallback_pipeline(question=question, top_k=top_k)
                try:
                    validate(fb, TASK_OUT_SCHEMA)
                except ValidationError:
                    fb = {"answer": fb.get("answer", ""), "evidence": fb.get("evidence", [])}
                return {"result": fb, "scratch": scratch, "steps": step + 1}

            # Execute tool call
            tool_name = str(plan.get("tool", "")).strip()
            args = plan.get("args") or {}
            if tool_name not in TOOLS:
                # Unknown tool; use fallback
                fb = fallback_pipeline(question=question, top_k=top_k)
                try:
                    validate(fb, TASK_OUT_SCHEMA)
                except ValidationError:
                    fb = {"answer": fb.get("answer", ""), "evidence": fb.get("evidence", [])}
                return {"result": fb, "scratch": scratch, "steps": step + 1}

            # Normalise rag_search args with sensible defaults
            if tool_name == "rag_search":
                args.setdefault("episode_id", episode_id if episode_id else None)
                args.setdefault("top_k", top_k)
                args.setdefault("scope", scope)
                if "query" in args and "question" not in args:
                    args["question"] = str(args.pop("query"))

            # Execute tool with timing
            t0 = time.perf_counter()
            with start_span(f"agent.tool.{tool_name}", kind="TOOL", attrs={"step": step}):
                try:
                    out = TOOLS[tool_name](args)
                except Exception as e:
                    out = {"error": str(e)}
            dt_ms = (time.perf_counter() - t0) * 1000.0

            scratch.append({"tool": tool_name, "args": args, "output": out, "latency_ms": dt_ms})

        # Step cap reached; use fallback and finalise
        fb = fallback_pipeline(question=question, top_k=top_k)
        try:
            validate(fb, TASK_OUT_SCHEMA)
        except ValidationError:
            fb = {"answer": fb.get("answer", ""), "evidence": fb.get("evidence", [])}
        return {"result": fb, "scratch": scratch, "steps": step_cap}

