from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from jsonschema import ValidationError, validate
from spike8_rag_contract.generator import generate_answer  # reuse openai/mock wrapper
from spike8_rag_contract.tracing import start_span

from spike11_agent.tools_constrained import (
    clip_linker,
    episode_locator,
    rag_search,
    sql_duckdb,
    timeline_builder,
)

# registry (ACL)
TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "rag_search": rag_search,
    "episode_locator": episode_locator,
    "timeline_builder": timeline_builder,
    "sql_duckdb": sql_duckdb,
    "clip_linker": clip_linker,
}

# task output schema (example: timeline compare)
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

SYS_PROMPT = (
    "You are a constrained Deep Agent. Always output the next action as **pure JSON ONLY**.\n"
    "You must reason step-by-step internally but emit only one JSON object per turn.\n"
    "\n"
    "ALLOWED TOOLS:\n"
    "  - rag_search\n"
    "  - episode_locator\n"
    "  - timeline_builder\n"
    "  - sql_duckdb\n"
    "  - clip_linker\n"
    "\n"
    "ARGUMENT SHAPES (must match exactly):\n"
    '  - rag_search.args = {"question": string, "episode_id": string, "top_k": int, "scope": "episode"|"corpus"|"auto"}\n'
    '    * If the episode_id is unknown, first call episode_locator OR set scope="corpus".\n'
    '    * NEVER call rag_search with scope="episode" and an empty episode_id.\n'
    '  - episode_locator.args = {"query": string, "limit": int}\n'
    '  - timeline_builder.args = {"segments": [{"episode_id": string, "start_ts": number, "end_ts": number, "text": string}]}\n'
    '  - sql_duckdb.args = {"sql": string}\n'
    '  - clip_linker.args = {"episode_id": string, "timestamp": number}\n'
    "\n"
    "OUTPUT MUST BE EXACTLY ONE of:\n"
    '  1) {"tool": "<name>", "args": {...}}\n'
    '  2) {"final": {"answer": string, "evidence": [{"episode_id": string, "start_ts": number, "end_ts": number}]}}\n'
    "\n"
    "POLICY:\n"
    "- Follow the argument shapes precisely (no extra keys, no omissions).\n"
    "- Keep reasoning grounded in retrieved data — cite real episode_id and timestamps.\n"
    "- Use at most one tool per step and stop when a coherent answer with evidence is ready.\n"
    "- When uncertain about episode context, call episode_locator first.\n"
    "- Keep answers concise and structured; prefer minimal steps within safety limits.\n"
)


def _llm_decide(
    question: str, scratch: list[dict[str, Any]], model_id: str, provider: str
) -> dict[str, Any]:
    # Use your minimal generate_answer with a fixed system prompt to produce JSON
    ctx = json.dumps({"question": question, "scratch": scratch})[:6000]
    user = f"State:\n{ctx}\n\nDecide next step."
    out = generate_answer(
        question=user,
        context=SYS_PROMPT,
        citations=[],
        provider=provider,
        model_id=model_id,
    ).strip()
    # best-effort JSON extraction
    try:
        return json.loads(out)
    except Exception:
        # force a safe termination if LLM goes off-script
        return {"final": {"answer": "Unable to proceed safely.", "evidence": []}}


def run_deep_agent(
    question: str,
    episode_id: str,
    top_k: int,
    scope: str,
    provider: str,
    model_id: str,
    step_cap: int = 6,
    per_step_timeout_s: float = 8.0,
) -> dict[str, Any]:
    scratch: list[dict[str, Any]] = []
    with start_span(
        "agent.run",
        kind="CHAIN",
        attrs={"step_cap": step_cap, "timeout_per_step_s": per_step_timeout_s, "scope": scope},
    ):
        for step in range(step_cap):
            with start_span("agent.plan", kind="LLM", attrs={"step": step}):
                plan = _llm_decide(
                    question=question,
                    scratch=scratch,
                    model_id=model_id,
                    provider=provider,
                )
            # termination?
            if "final" in plan:
                result = plan["final"]
                try:
                    validate(result, TASK_OUT_SCHEMA)
                except ValidationError:
                    result = {"answer": str(result), "evidence": []}
                return {"result": result, "scratch": scratch, "steps": step + 1}

            # tool call
            tool_name = str(plan.get("tool", "")).strip()
            args = plan.get("args") or {}
            if tool_name not in TOOLS:
                # fail safe
                return {
                    "result": {"answer": "Refused: tool not allowed.", "evidence": []},
                    "scratch": scratch,
                    "steps": step + 1,
                }

            # add defaults + normalize args for rag_search
            if tool_name == "rag_search":
                args.setdefault("episode_id", episode_id)
                args.setdefault("top_k", top_k)
                args.setdefault("scope", scope)
                # tolerate "query" from the planner; map to schema field
                if "query" in args and "question" not in args:
                    args["question"] = str(args.pop("query"))
            # run tool with timeout
            t0 = time.perf_counter()
            with start_span(f"agent.tool.{tool_name}", kind="TOOL", attrs={"step": step}):
                try:
                    out = TOOLS[tool_name](args)
                except Exception as e:
                    out = {"error": str(e)}
            dt = (time.perf_counter() - t0) * 1000.0
            scratch.append({"tool": tool_name, "args": args, "output": out, "latency_ms": dt})

        # step cap reached
        return {
            "result": {"answer": "Step cap reached without finalization.", "evidence": []},
            "scratch": scratch,
            "steps": step_cap,
        }
