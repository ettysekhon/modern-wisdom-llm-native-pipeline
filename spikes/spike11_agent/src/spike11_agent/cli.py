from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace

from jsonschema import validate
from rich.console import Console
from spike8_rag_contract import paths

from spike11_agent.agent import TASK_OUT_SCHEMA, run_deep_agent
from spike11_agent.specs import write_specs

console = Console()


def cmd_tools_specs(_: Namespace) -> None:
    paths_written = write_specs()
    for p in paths_written:
        console.print(f"[green]Wrote spec:[/green] {p}")


def cmd_deep_agent(args: Namespace) -> None:
    out = run_deep_agent(
        question=args.question,
        episode_id=args.episode_id,
        top_k=args.top_k,
        scope=args.scope,
        provider=args.llm_provider,
        model_id=args.llm_model_id,
        step_cap=args.step_cap,
        per_step_timeout_s=args.step_timeout_s,
    )
    # validate task result
    validate(out["result"], TASK_OUT_SCHEMA)

    # persist artifacts
    plans_dir = paths.DATA_DIR / "agents" / "plans"
    evals_dir = paths.DATA_DIR / "evals" / "agents"
    plans_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    plan_path = plans_dir / "latest_plan.json"
    res_path = evals_dir / "latest_result.json"
    plan_path.write_text(json.dumps({"question": args.question, **out}, indent=2))
    res_path.write_text(json.dumps(out["result"], indent=2))
    console.print(f"[green]OK →[/green] {res_path}")


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="spike11", description="Spike 11 — Agentic Workflows")
    sub = p.add_subparsers(dest="cmd", required=True)

    specs = sub.add_parser("tools-specs", help="Write tool I/O schemas to data/agents/specs")
    specs.set_defaults(func=cmd_tools_specs)

    run = sub.add_parser("deep-agent", help="Run constrained Deep Agent")
    run.add_argument("--episode-id", required=True)
    run.add_argument("--question", required=True)
    run.add_argument("--top-k", type=int, default=8)
    run.add_argument("--scope", choices=["episode", "corpus", "auto"], default="auto")
    run.add_argument("--llm-provider", choices=["mock", "openai"], default="mock")
    run.add_argument("--llm-model-id", default="gpt-4o-mini")
    run.add_argument("--step-cap", type=int, default=6)
    run.add_argument("--step-timeout-s", type=float, default=8.0)
    run.set_defaults(func=cmd_deep_agent)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
