import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .chunkers import PrepParams, make_rows_fixed, make_rows_sentence_bound, make_rows_time_window
from .configs import write_chunking_toml
from .decision import write_decision_md
from .eval import evaluate_methods, write_eval_report
from .metadata import enrich_chunk_rows, load_episode_meta
from .paths import CHUNKS_DIR, DOCS_DIR, DUCKDB_PATH
from .persist import upsert_duckdb, write_chunks_parquet
from .transcript import load_episode_parquet, validate_transcript_df

console = Console()


def cmd_check(args):
    df = load_episode_parquet(args.episode_id)
    errs = validate_transcript_df(df)
    table = Table(title=f"Transcript check — {args.episode_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Rows", str(len(df)))
    table.add_row("Cols", ", ".join(df.columns))
    console.print(table)
    if errs:
        console.print("[yellow]Issues found:[/yellow]")
        for e in errs:
            console.print(f"- {e}")
        sys.exit(1)


def cmd_chunk(args):
    df = load_episode_parquet(args.episode_id)
    episode_meta = load_episode_meta(args.episode_id, DUCKDB_PATH)
    params = PrepParams(
        method="",
        size_tokens=args.size_tokens,
        overlap_tokens=args.overlap_tokens,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        chunk_v=args.chunk_v,
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        params.method = m
        if m == "fixed":
            rows = make_rows_fixed(df, params)
        elif m == "sentence_bound":
            rows = make_rows_sentence_bound(df, params)
        elif m == "time_window":
            rows = make_rows_time_window(df, params)
        else:
            raise ValueError(f"Unknown method {m}")

        # Enrich the chunk rows with episode metadata
        rows = enrich_chunk_rows(rows, episode_meta)

        path = write_chunks_parquet(rows, m, args.episode_id, out_dir=CHUNKS_DIR)
        if args.duckdb:
            upsert_duckdb(path, db_path=DUCKDB_PATH)
        console.print(f"[green]Wrote {len(rows)} rows for {m} → {path}[/green]")


def cmd_eval(args):
    report = evaluate_methods(
        episode_id=args.episode_id,
        qa_csv=Path(args.qa_csv),
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        k=args.k,
        tol_s=args.tolerance_s,
        prefer_efficient=args.prefer_efficient,
        # chunks_dir defaults to CHUNKS_DIR
    )
    report_path, md_path_basic = write_eval_report(report)
    console.print(f"[green]Eval report complete → {report_path}, {md_path_basic}[/green]")

    # write decision MD & configs based on the *report*
    md_path = write_decision_md(
        report,
        out_path=DOCS_DIR / "0003-chunking.md",
        owner=args.owner,
        qa_csv=Path(args.qa_csv),
        context_window=args.context_window,
    )
    cfg_path = write_chunking_toml(
        episode_id=report["episode_id"],
        winner_method=report["winner"],
        out_path=Path("configs") / "chunking.toml",
        chunks_dir=CHUNKS_DIR,
    )
    console.print(f"[green]Eval decision complete → {report_path}[/green]")
    console.print(f"[green]Decision updated → {md_path}[/green]")
    console.print(f"[green]Config updated → {cfg_path}[/green]")


def build_parser():
    p = argparse.ArgumentParser(prog="spike3", description="Spike 3 — Chunking & metadata")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--episode-id", required=True)
    p_check.set_defaults(func=cmd_check)
    p_chunk = sub.add_parser("chunk")
    p_chunk.add_argument("--episode-id", required=True)
    p_chunk.add_argument("--methods", default="fixed,sentence_bound,time_window")
    p_chunk.add_argument("--size-tokens", type=int, default=700)
    p_chunk.add_argument("--overlap-tokens", type=int, default=100)
    p_chunk.add_argument("--window-seconds", type=int, default=190)
    p_chunk.add_argument("--overlap-seconds", type=int, default=30)
    p_chunk.add_argument("--chunk-v", default="c1")
    p_chunk.add_argument("--duckdb", action="store_true")
    p_chunk.set_defaults(func=cmd_chunk)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--episode-id", required=True)
    p_eval.add_argument("--methods", default="fixed,sentence_bound,time_window")
    p_eval.add_argument("--qa-csv", required=True)
    p_eval.add_argument("--k", type=int, default=20)
    p_eval.add_argument("--tolerance-s", type=int, default=7)
    p_eval.add_argument(
        "--prefer-efficient",
        action="store_true",
        help="When Hit@10 and MRR tie, prefer smaller AvgTokens then AvgDuration",
    )
    p_eval.add_argument("--owner", default="", help="Decision owner for the MD note")
    p_eval.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="LLM context window (tokens)",
    )
    p_eval.set_defaults(func=cmd_eval)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
