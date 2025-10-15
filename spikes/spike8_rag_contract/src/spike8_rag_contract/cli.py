from __future__ import annotations

import json
import re
import time
import uuid
from argparse import ArgumentParser, Namespace

from jsonschema import ValidationError, validate
from rich.console import Console
from tiktoken import get_encoding

from . import paths
from .generator import generate_answer, generate_guest_summary  # if used
from .io import load_chunks_df
from .qdrant import client, vector_search
from .retrieval import embed_question_fastembed
from .schema import (
    ANSWER_ENVELOPE_SCHEMA,
    GUEST_SUMMARY_ENVELOPE_SCHEMA,
    WEEKLY_DIGEST_ENVELOPE_SCHEMA,
    write_schemas,
)
from .tracing import get_tracer, start_span

tracer = get_tracer()

console = Console()


def _to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _build_context_and_citations(
    retrieved: list[dict], max_ctx_tokens: int = 1200
) -> tuple[str, list[str]]:
    enc = get_encoding("cl100k_base")
    retrieved = sorted(retrieved, key=lambda x: x.get("score", 0.0), reverse=True)
    citations = [r["chunk_id"] for r in retrieved[:3]]
    ctx_parts, used = [], 0
    for r in retrieved:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        t = len(enc.encode(txt))
        if used + t > max_ctx_tokens:
            break
        ctx_parts.append(txt)
        used += t
    return "\n\n---\n\n".join(ctx_parts), citations


def _retrieval_common(args):
    query_text = getattr(args, "question", None) or getattr(args, "prompt", None) or "summary"

    with start_span("retrieve", kind="RETRIEVER") as span:
        span.set_attribute("rag.phase", "retrieve")
        span.set_attribute("rag.index_version", paths.INDEX_VERSION)
        span.set_attribute("rag.episode_id", args.episode_id)
        span.set_attribute("rag.emb_v", args.emb_v)
        span.set_attribute("retrieval.top_k", int(args.top_k))
        span.set_attribute("input.value", query_text)

        cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)
        q_vec = embed_question_fastembed(query_text, model_id=args.query_model_id)

        t0 = time.perf_counter()
        docs, rt_ms = vector_search(
            cli,
            collection=paths.INDEX_VERSION,
            episode_id=args.episode_id,
            q_vector=q_vec,
            top_k=args.top_k,
        )
        span.set_attribute("latency.ms", (time.perf_counter() - t0) * 1000.0)
        span.set_attribute("retrieval.latency.ms", float(rt_ms))

        chunks_df = load_chunks_df(args.method, args.episode_id)
        pld_map = {str(row["chunk_id"]): row.to_dict() for _, row in chunks_df.iterrows()}

        retrieved = []
        for i, d in enumerate(docs):
            did = d["id"] if isinstance(d, dict) else getattr(d, "id", None)
            score = d["score"] if isinstance(d, dict) else getattr(d, "score", None)
            cid = str(did)
            row = pld_map.get(cid, {}) or {}
            item = {
                "chunk_id": cid,
                "start_ts": float(row.get("start_ts", 0.0)),
                "end_ts": float(row.get("end_ts", 0.0)),
                "score": float(score or 0.0),
                "text": row.get("text", ""),
                "episode_id": row.get("episode_id", args.episode_id),
            }
            retrieved.append(item)

            # annotate top results succinctly
            prefix = f"retrieval.documents.{i}.document"
            span.set_attribute(f"{prefix}.id", cid)
            span.set_attribute(f"{prefix}.score", item["score"])
            if i < 5:  # keep attributes small
                preview = (item["text"] or "")[:300]
                span.set_attribute(f"{prefix}.content", preview)

        span.set_attribute("retrieval.documents.count", int(len(retrieved)))

    return retrieved, rt_ms, pld_map


def cmd_guest_summary(args) -> None:
    with start_span("rag.guest_summary", kind="CHAIN") as span:
        span.set_attribute("rag.phase", "pipeline")
        span.set_attribute("rag.index_version", paths.INDEX_VERSION)
        span.set_attribute("rag.episode_id", args.episode_id)
        span.set_attribute("llm.provider", args.llm_provider)
        span.set_attribute("llm.model_id", args.llm_model_id)

        retrieved, rt_ms, _ = _retrieval_common(args)
        context, citations = _build_context_and_citations(
            retrieved, max_ctx_tokens=args.max_ctx_tokens
        )

        with start_span("generate", kind="LLM") as gspan:
            gspan.set_attribute("llm.provider", args.llm_provider)
            gspan.set_attribute("llm.model_id", args.llm_model_id)
            t0 = time.perf_counter()
            body = generate_guest_summary(
                question=args.question,
                context=context,
                provider=args.llm_provider,
                model_id=args.llm_model_id,
            )
            gen_ms = (time.perf_counter() - t0) * 1000.0
            gspan.set_attribute("latency.ms", gen_ms)

        env = {
            "answer": body,
            "citations": citations,
            "trace_id": str(uuid.uuid4()),
            "timings": {"retrieve_ms": rt_ms, "generate_ms": gen_ms},
            "cost": {},
            "model_id": args.llm_model_id,
            "index_version": paths.INDEX_VERSION,
            "retrieved": retrieved,
        }

        with start_span("validate", kind="VALIDATOR") as vspan:
            try:
                validate(instance=env, schema=GUEST_SUMMARY_ENVELOPE_SCHEMA)
                vspan.set_attribute("validation.ok", True)
            except ValidationError as e:
                vspan.set_attribute("validation.ok", False)
                vspan.record_exception(e)
                console.print(f"[red]Schema validation failed:[/red] {e.message}")
                raise SystemExit(1) from e

        out = paths.SAMPLES_DIR / "sample_guest_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(env, indent=2))
        span.set_attribute("output.path", str(out))
        console.print(f"[green]OK →[/green] {out}")


def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # split on sentence-ish boundary; fall back to first ~12 words
    for sep in [". ", "! ", "? ", "\n"]:
        if sep in t:
            return t.split(sep, 1)[0].strip()
    return " ".join(t.split()[:12]).strip()


def _two_sentence_summary(text: str, max_words: int = 80) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    parts = []
    remain = max_words
    for seg in re.split(r"(?<=[\.!?])\s+", t):
        if not seg:
            continue
        words = seg.split()
        if remain <= 0:
            break
        if len(words) <= remain:
            parts.append(seg.strip())
            remain -= len(words)
        else:
            parts.append(" ".join(words[:remain]).strip())
            break
        if len(parts) >= 2:  # roughly two sentences
            break
    return " ".join(parts).strip()


def _is_near_duplicate(a: str, b: str, thresh: float = 0.9) -> bool:
    # very cheap Jaccard on words to avoid repeating the same promo line
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return False
    j = len(wa & wb) / max(1, len(wa | wb))
    return j >= thresh


def cmd_weekly_digest(args: Namespace) -> None:
    with start_span("rag.weekly_digest", kind="CHAIN") as span:
        span.set_attribute("rag.phase", "pipeline")
        span.set_attribute("rag.index_version", paths.INDEX_VERSION)
        span.set_attribute("rag.episode_id", args.episode_id)
        span.set_attribute("llm.provider", args.llm_provider)
        span.set_attribute("llm.model_id", args.llm_model_id)

        retrieved, rt_ms, _ = _retrieval_common(args)

        # Build citations from top-3 chunk_ids
        citations = [r["chunk_id"] for r in retrieved[:3]]

        # Deterministic digest from retrieved (your current logic)
        items = []
        for r in retrieved:
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            if any(_is_near_duplicate(txt, it["summary"]) for it in items):
                continue
            items.append(
                {
                    "title": _first_sentence(txt)[:120],
                    "summary": _two_sentence_summary(txt, max_words=80),
                    "episode_id": args.episode_id,
                    "timestamp": float(r.get("start_ts", 0.0)),
                }
            )
            if len(items) >= 3:
                break

        while len(items) < 3:
            items.append(
                {"title": "", "summary": "", "episode_id": args.episode_id, "timestamp": 0.0}
            )

        env = {
            "answer": {"items": items},
            "citations": citations,
            "trace_id": str(uuid.uuid4()),
            "timings": {"retrieve_ms": rt_ms, "generate_ms": 0.0},
            "cost": {},
            "model_id": args.llm_model_id,
            "index_version": paths.INDEX_VERSION,
            "retrieved": retrieved,
        }

        with start_span("validate", kind="VALIDATOR") as vspan:
            try:
                validate(env, WEEKLY_DIGEST_ENVELOPE_SCHEMA)
                vspan.set_attribute("validation.ok", True)
            except ValidationError as err:
                vspan.set_attribute("validation.ok", False)
                vspan.record_exception(err)
                console.print(f"[red]Schema validation failed:[/red] {err.message}")
                raise SystemExit(1) from err

        out = paths.CONTRACTS_DIR / "sample_responses" / "sample_weekly_digest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(env, indent=2))
        span.set_attribute("output.path", str(out))
        console.print(f"[green]OK →[/green] {out}")


def cmd_dump_schemas(_args: Namespace) -> None:
    written = write_schemas()
    for p in written:
        console.print(f"[green]Wrote schema:[/green] {p}")


def cmd_run(args: Namespace) -> None:
    with start_span("rag.run", kind="CHAIN") as span:
        span.set_attribute("rag.phase", "pipeline")
        span.set_attribute("rag.index_version", paths.INDEX_VERSION)
        span.set_attribute("rag.episode_id", args.episode_id)
        span.set_attribute("rag.emb_v", args.emb_v)
        span.set_attribute("rag.method", args.method)
        span.set_attribute("llm.provider", args.llm_provider)
        span.set_attribute("llm.model_id", args.llm_model_id)
        span.set_attribute("retrieval.top_k", int(args.top_k))

        # 1) Retrieve
        retrieved, rt_ms, _ = _retrieval_common(args)

        # 2) Context
        context, citations = _build_context_and_citations(
            retrieved, max_ctx_tokens=args.max_ctx_tokens
        )
        span.set_attribute(
            "context.tokens.approx", len(get_encoding("cl100k_base").encode(context))
        )

        # 3) Generate
        with start_span("generate", kind="LLM") as gspan:
            gspan.set_attribute("llm.provider", args.llm_provider)
            gspan.set_attribute("llm.model_id", args.llm_model_id)
            t0 = time.perf_counter()
            answer_text = generate_answer(
                question=args.question,
                context=context,
                citations=citations,
                provider=args.llm_provider,
                model_id=args.llm_model_id,
            )
            gen_ms = (time.perf_counter() - t0) * 1000.0
            gspan.set_attribute("latency.ms", gen_ms)
            gspan.set_attribute("citations.count", len(citations))

        # 4) Envelope + validate
        env = {
            "answer": answer_text,
            "citations": citations,
            "trace_id": str(uuid.uuid4()),
            "timings": {"retrieve_ms": rt_ms, "generate_ms": gen_ms},
            "cost": {},
            "model_id": args.llm_model_id,
            "index_version": paths.INDEX_VERSION,
            "retrieved": retrieved,
        }

        with start_span("validate", kind="VALIDATOR") as vspan:
            try:
                validate(instance=env, schema=ANSWER_ENVELOPE_SCHEMA)
                vspan.set_attribute("validation.ok", True)
            except ValidationError as e:
                vspan.set_attribute("validation.ok", False)
                vspan.record_exception(e)
                console.print(f"[red]Schema validation failed:[/red] {e.message}")
                raise SystemExit(1) from e

        # 5) Write
        out = paths.CONTRACTS_DIR / "sample_responses" / "sample_answer.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(env, indent=2))
        span.set_attribute("output.path", str(out))
        console.print(f"[green]OK →[/green] {out}")


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="spike8", description="Spike 8 — RAG Contract (CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- run (existing) ---
    run = sub.add_parser("run", help="Retrieve → Generate → Validate envelope")
    run.add_argument("--episode-id", required=True)
    run.add_argument("--emb-v", required=True)
    run.add_argument("--method", default="sentence_bound")
    run.add_argument("--question", required=True)
    run.add_argument("--top-k", type=int, default=8)
    run.add_argument("--llm-provider", choices=["mock", "openai"], default="mock")
    run.add_argument("--llm-model-id", default="gpt-4o-mini")
    run.add_argument("--query-model-id", default="BAAI/bge-small-en-v1.5")
    run.add_argument("--max-ctx-tokens", type=int, default=1200)
    run.set_defaults(func=cmd_run)

    # --- guest-summary ---
    gs = sub.add_parser("guest-summary", help="Guest summary → envelope")
    gs.add_argument("--episode-id", required=True)
    gs.add_argument("--emb-v", required=True)
    gs.add_argument("--method", default="sentence_bound")
    gs.add_argument("--question", default="Summarize the guest.")
    gs.add_argument("--top-k", type=int, default=8)
    gs.add_argument("--llm-provider", choices=["mock", "openai"], default="mock")
    gs.add_argument("--llm-model-id", default="gpt-4o-mini")
    gs.add_argument("--query-model-id", default="BAAI/bge-small-en-v1.5")
    gs.add_argument("--max-ctx-tokens", type=int, default=1200)
    gs.set_defaults(func=cmd_guest_summary)

    # --- weekly-digest ---
    wd = sub.add_parser("weekly-digest", help="Weekly digest → envelope")
    wd.add_argument("--episode-id", required=True)
    wd.add_argument("--emb-v", required=True)
    wd.add_argument("--method", default="sentence_bound")
    wd.add_argument("--prompt", default="Create a short weekly digest of key highlights.")
    wd.add_argument("--top-k", type=int, default=8)
    wd.add_argument("--llm-provider", choices=["mock", "openai"], default="mock")
    wd.add_argument("--llm-model-id", default="gpt-4o-mini")
    wd.add_argument("--query-model-id", default="BAAI/bge-small-en-v1.5")
    wd.add_argument("--max-ctx-tokens", type=int, default=1200)
    wd.set_defaults(func=cmd_weekly_digest)

    dump = sub.add_parser("dump-schemas", help="Write JSON Schemas to data/contracts/schemas")
    dump.set_defaults(func=cmd_dump_schemas)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
