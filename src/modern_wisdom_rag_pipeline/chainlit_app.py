from __future__ import annotations

import os
from typing import cast

import chainlit as cl
from spike8_rag_contract import paths

# Your agent/tooling
from spike11_agent.agent import run_deep_agent

# Optional: configure defaults via env
DEFAULT_PROVIDER = os.getenv("MW_LLM_PROVIDER", "openai")  # "mock" or "openai"
DEFAULT_MODEL_ID = os.getenv("MW_LLM_MODEL", "gpt-4o-mini")
DEFAULT_SCOPE = os.getenv("MW_SCOPE", "corpus")  # "episode" | "corpus" | "auto"
DEFAULT_TOP_K = int(os.getenv("MW_TOP_K", "8"))
DEFAULT_EPISODE_ID = os.getenv("MW_EPISODE_ID", "")

WELCOME = (
    "Ask something like:\n"
    "- *How do I stay disciplined when times get tough*\n\n"
    "- *What are the main factors that contribute to happiness and wellbeing?*\n\n"
    "I’ll search the indexed episodes (Qdrant) and return evidence with timestamps."
)


HISTORY_TURNS = 4  # keep the last 4 message pairs (8 turns)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])  # [{"role":"user"|"assistant","content":str}, ...]
    await cl.Message(
        content=f"Modern Wisdom RAG (index: `{paths.INDEX_VERSION}`)\n\n{WELCOME}"
    ).send()


def _clip_url(episode_id: str, ts: float) -> str:
    base = "https://modernwisdom.fm/episode"
    return f"{base}/{episode_id}?t={int(ts)}"


def _trim_history(hist: list[dict], keep_pairs: int = HISTORY_TURNS) -> list[dict]:
    # keep the last N user/assistant pairs (2N turns)
    turns = []
    for h in hist:
        if h.get("role") in ("user", "assistant"):
            turns.append(h)
    # retain the tail (2*keep_pairs turns)
    return turns[-(2 * keep_pairs) :]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    q = (message.content or "").strip()
    if not q:
        await cl.Message(content="Please type a question.").send()
        return

    history = cast(list[dict], cl.user_session.get("history") or [])
    history = _trim_history(history)

    status = await cl.Message(content="Searching…").send()
    try:
        out = run_deep_agent(
            question=q,
            episode_id=DEFAULT_EPISODE_ID,
            top_k=DEFAULT_TOP_K,
            scope=DEFAULT_SCOPE,
            provider=DEFAULT_PROVIDER,
            model_id=DEFAULT_MODEL_ID,
            step_cap=6,
            per_step_timeout_s=8.0,
            chat_history=history,  # <<< NEW
        )

        result = out.get("result") or {}
        answer = (result.get("answer") or "").strip()
        evidence = result.get("evidence") or []

        # Update/trim history for the next follow-up
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        cl.user_session.set("history", _trim_history(history))

        # Render evidence
        seen = set()
        cleaned = []
        for ev in evidence:
            key = (str(ev.get("episode_id", "")), float(ev.get("start_ts", 0.0)))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(ev)
            if len(cleaned) >= 5:
                break

        if cleaned:
            lines = []
            for i, ev in enumerate(cleaned, 1):
                ep = str(ev.get("episode_id", ""))
                t0 = float(ev.get("start_ts", 0.0))
                t1 = float(ev.get("end_ts", 0.0))
                url = _clip_url(ep, t0)
                lines.append(f"{i}. `{ep}` — {t0:.2f}s–{t1:.2f}s → [Open clip]({url})")
            evidence_md = "\n".join(lines)
        else:
            evidence_md = "_No matching clips found._"

        await status.stream_token(f"**Answer**\n\n{answer}\n\n**Evidence**\n\n{evidence_md}")

        await cl.Message(
            content="Debug artifacts attached.",
            elements=[
                cl.File(
                    name="latest_plan.json",
                    path=str((paths.DATA_DIR / "agents" / "plans" / "latest_plan.json").resolve()),
                ),
                cl.File(
                    name="latest_result.json",
                    path=str(
                        (paths.DATA_DIR / "evals" / "agents" / "latest_result.json").resolve()
                    ),
                ),
            ],
        ).send()

    except Exception as e:
        import traceback

        await status.stream_token(f"**Error**: {e}\n\n```\n{traceback.format_exc()}\n```")
