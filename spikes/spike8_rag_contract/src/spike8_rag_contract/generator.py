from __future__ import annotations

import contextlib
import json
from typing import Any

from .models import get_openai_client
from .tracing import start_span


def _mock_generate(question: str, context: str) -> str:
    return f"Answer (mocked) to: {question}\n\nContext: {context[:800]}"


def _openai_generate(question: str, context: str, model_id: str) -> str:
    client = get_openai_client()

    system = (
        "You are a precise assistant. Answer using ONLY the provided context. "
        "Be concise and cite chunk ids if relevant (e.g. [chunk:XXXX])."
    )
    user = f"Question:\n{question}\n\nContext:\n{context}"

    # simple trunc for previews in Phoenix
    def _short(s: str, n: int = 1000) -> str:
        s = s or ""
        return s if len(s) <= n else s[: n - 1] + "…"

    with start_span("llm.openai.chat", kind="LLM", attrs={"llm.model_name": model_id}) as span:
        # OpenInference-friendly attributes
        with contextlib.suppress(Exception):
            span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.provider", "openai")
        span.set_attribute("llm.model_id", model_id)
        span.set_attribute("input.value", _short(user, 2000))

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=500,
            )
            out = resp.choices[0].message.content or ""

            # token usage (best-effort)
            usage = getattr(resp, "usage", None)
            if usage:
                prompt_toks = int(getattr(usage, "prompt_tokens", 0) or 0)
                comp_toks = int(getattr(usage, "completion_tokens", 0) or 0)
                total_toks = int(getattr(usage, "total_tokens", prompt_toks + comp_toks) or 0)
                span.set_attribute("llm.usage.prompt_tokens", prompt_toks)
                span.set_attribute("llm.usage.completion_tokens", comp_toks)
                span.set_attribute("llm.usage.total_tokens", total_toks)

                # optional: rough cost estimator (update prices as needed)
                prices_usd = {
                    # per 1K tokens
                    "gpt-4o-mini": {"input": 0.0005, "output": 0.0015},
                }
                price = prices_usd.get(model_id)
                if price:
                    cost = (prompt_toks / 1000.0) * price["input"] + (comp_toks / 1000.0) * price[
                        "output"
                    ]
                    span.set_attribute("llm.cost.usd", float(cost))

            span.set_attribute("output.value", _short(out, 2000))
            return out

        except Exception as err:
            # make failures visible in Phoenix
            try:
                span.record_exception(err)
                span.set_attribute("error", True)
            finally:
                raise err


def generate_answer(
    question: str,
    context: str,
    citations: list[str],
    provider: str = "mock",
    model_id: str = "gpt-4o-mini",
) -> str:
    p = provider.lower()
    if p == "mock":
        with start_span("llm.mock", kind="LLM", attrs={"llm.model_name": "mock"}):
            return _mock_generate(question, context)
    if p == "openai":
        return _openai_generate(question, context, model_id=model_id)
    raise ValueError(f"Unsupported llm provider: {provider}")


def _fallback_guest_summary_from_text(context: str) -> dict[str, Any]:
    # Super-stable fallback: guest unknown; key points = first 3 sentences-ish
    import re

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", context) if s.strip()]
    key_points = sentences[:3] if sentences else ["No salient points in context."]
    return {"guest": "Unknown", "key_points": key_points}


def generate_guest_summary(
    question: str,
    context: str,
    provider: str = "mock",
    model_id: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Return {"guest": str, "key_points": [str, ...]}.
    Deterministic fallback if LLM fails or provider=mock.
    """
    if provider == "mock":
        return _fallback_guest_summary_from_text(context)

    if provider == "openai":
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Summarize the guest based strictly on the provided context. "
                'Return ONLY JSON with fields: {"guest": string, "key_points": [string, ...]}.\n\n'
                f"Question: {question}\n\nContext:\n{context}\n"
            )
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = r.choices[0].message.content or "{}"
            obj = json.loads(raw)
            # light shape check
            if not isinstance(obj, dict) or "guest" not in obj or "key_points" not in obj:
                return _fallback_guest_summary_from_text(context)
            if not isinstance(obj.get("key_points"), list):
                obj["key_points"] = [str(obj.get("key_points"))]
            return {
                "guest": str(obj.get("guest", "Unknown")),
                "key_points": [str(x) for x in obj["key_points"]],
            }
        except Exception:
            return _fallback_guest_summary_from_text(context)

    # Unknown provider → fallback
    return _fallback_guest_summary_from_text(context)


def _fallback_weekly_digest_from_retrieved(retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    # Make a tiny deterministic digest from top 3 chunks
    def _first_words(s: str, n: int) -> str:
        parts = (s or "").split()
        return " ".join(parts[:n]).strip()

    items: list[dict[str, Any]] = []
    for r in retrieved[:3]:
        txt = r.get("text", "") or ""
        title = _first_words(txt, 8) or "Untitled"
        summary = _first_words(txt, 40) or "No summary available."
        items.append(
            {
                "title": title,
                "summary": summary,
                "episode_id": r.get("episode_id") or "",  # may be empty, envelope allows it
                "timestamp": float(r.get("start_ts", 0.0)),
            }
        )
    return {"items": items}


def generate_weekly_digest(
    prompt: str,
    retrieved: list[dict[str, Any]],
    provider: str = "mock",
    model_id: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Return {"items":[{"title":..., "summary":..., "episode_id":..., "timestamp":...}, ...]}.
    Deterministic fallback if LLM fails or provider=mock.
    """
    if provider == "mock":
        return _fallback_weekly_digest_from_retrieved(retrieved)

    if provider == "openai":
        try:
            from openai import OpenAI

            client = OpenAI()
            # Pack minimal context from retrieved
            ctx_lines = []
            for r in retrieved[:6]:
                ctx_lines.append(f"- [{r.get('chunk_id')}] {(r.get('text') or '').strip()}")
            ctx = "\n".join(ctx_lines)
            sys = (
                "You create a short digest. Return ONLY JSON with: "
                '{"items":[{"title":string,"summary":string,"episode_id":string,"timestamp":number}, ...]}'
            )
            user = f"Prompt: {prompt}\n\nContext:\n{ctx}\n"
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.0,
            )
            raw = r.choices[0].message.content or "{}"
            obj = json.loads(raw)
            # Light validation
            items = obj.get("items", [])
            if not isinstance(items, list) or not items:
                return _fallback_weekly_digest_from_retrieved(retrieved)
            norm = []
            for it in items[:5]:
                norm.append(
                    {
                        "title": str(it.get("title", "Untitled")),
                        "summary": str(it.get("summary", "")),
                        "episode_id": str(it.get("episode_id", "")),
                        "timestamp": float(it.get("timestamp", 0.0)),
                    }
                )
            return {"items": norm}
        except Exception:
            return _fallback_weekly_digest_from_retrieved(retrieved)

    return _fallback_weekly_digest_from_retrieved(retrieved)
