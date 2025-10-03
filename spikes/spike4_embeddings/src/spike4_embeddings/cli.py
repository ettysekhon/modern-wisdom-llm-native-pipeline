import argparse
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table

from . import paths
from .embed import embed_chunks_df, filter_idempotent
from .io import load_chunks, read_existing_embeddings, upsert_duckdb_embeddings, write_embeddings
from .manifest import write_embed_manifest
from .metrics import estimate_openai_cost, summarize_batch

console = Console()

SUPPORTED_PROVIDERS = ["openai", "fastembed"]


def cmd_check(args):
    chunks = load_chunks(args.method, args.episode_id, chunks_dir=paths.CHUNKS_DIR)
    table = Table(title=f"Spike 4 Check — {args.episode_id} [{args.method}]")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Chunks", str(len(chunks)))
    table.add_row("Avg tokens", f"{float(chunks['n_tokens'].mean()):.1f}")
    table.add_row("Avg duration (s)", f"{float(chunks['duration_s'].mean()):.1f}")
    console.print(table)


def cmd_embed(args):
    chunks = load_chunks(args.method, args.episode_id, chunks_dir=paths.CHUNKS_DIR)

    existing = read_existing_embeddings(args.emb_v, args.episode_id, emb_dir=paths.EMB_DIR)
    pending = filter_idempotent(chunks, existing)

    if existing is not None:
        console.print(f"[dim]Existing vectors: {len(existing)}; pending: {len(pending)}[/dim]")
    else:
        console.print(
            f"[dim]No existing vectors for emb_v={args.emb_v}; pending: {len(pending)}[/dim]"
        )

    if pending.empty:
        console.print("[green]Nothing to embed. Idempotent skip.[/green]")
        return

    rows = embed_chunks_df(
        pending,
        emb_v=args.emb_v,
        provider=args.provider,
        model_id=args.model_id,
        batch_size=args.batch_size,
        retries=args.retries,
        sleep_base_ms=args.sleep_base_ms,
    )

    out_path = write_embeddings(rows, args.emb_v, args.episode_id, emb_dir=paths.EMB_DIR)
    console.print(f"[green]Wrote embeddings → {out_path}[/green]")

    summary = summarize_batch(rows)
    total_tokens = int(pending["n_tokens"].sum())
    est_cost = None
    if args.provider.lower() == "openai" and args.price_per_1k > 0:
        est_cost = estimate_openai_cost(total_tokens, args.price_per_1k)

    table = Table(title="Batch Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("rows", str(summary["rows"]))
    table.add_row("rows_ok", str(summary["rows_ok"]))
    table.add_row("dim", str(summary["dim"]))
    table.add_row("avg_attempts", f"{summary['avg_attempts']:.2f}")
    table.add_row("tokens_total", str(total_tokens))
    table.add_row("est_cost", f"${est_cost:.4f}" if est_cost is not None else "n/a")
    console.print(table)

    if args.duckdb:
        upsert_duckdb_embeddings(out_path, db_path=paths.DUCKDB_PATH)
        console.print("[green]Upserted into DuckDB table `embeddings`[/green]")


def cmd_stats(args):
    emb_dir = paths.EMB_DIR / args.emb_v
    if not emb_dir.exists():
        console.print(f"[red]No embeddings found for emb_v={args.emb_v}[/red]")
        sys.exit(1)
    parts = list(emb_dir.glob("episode_id=*/part-*.parquet"))
    if not parts:
        console.print(f"[red]No parquet parts found under {emb_dir}[/red]")
        sys.exit(1)
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    table = Table(title=f"Embeddings Stats — {args.emb_v}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("episodes", str(len(set(df["episode_id"]))))
    table.add_row("rows", str(len(df)))
    dim_vals = df["dim"].dropna().unique()
    dim = int(dim_vals[0]) if len(dim_vals) > 0 else 0
    table.add_row("dim", str(dim))
    table.add_row("providers", ", ".join(sorted(set(df["provider"]))))
    table.add_row("models", ", ".join(sorted(set(df["model_id"]))))
    console.print(table)


def cmd_sync(args):
    """
    Idempotently upsert existing embeddings parquet into DuckDB.
    Useful after a crash where parquet was written but DuckDB wasn't updated.
    """
    p = paths.EMB_DIR / args.emb_v / f"episode_id={args.episode_id}" / "part-00000.snappy.parquet"
    if not p.exists():
        console.print(
            f"[red]No parquet found for emb_v={args.emb_v} episode_id={args.episode_id}[/red]"
        )
        sys.exit(1)

    upsert_duckdb_embeddings(p, db_path=paths.DUCKDB_PATH)
    console.print(f"[green]Synced embeddings → {paths.DUCKDB_PATH}[/green]")


def cmd_manifest(args):
    out = write_embed_manifest(args.emb_v)
    console.print(f"[green]Wrote manifest → {out}[/green]")


def build_parser():
    p = argparse.ArgumentParser(prog="spike4", description="Spike 4 — Embeddings (batch) & cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Check chunks ready to embed")
    p_check.add_argument("--episode-id", required=True)
    p_check.add_argument("--method", required=True, help="e.g., sentence_bound")
    p_check.set_defaults(func=cmd_check)

    p_embed = sub.add_parser("embed", help="Embed one episode")
    p_embed.add_argument("--episode-id", required=True)
    p_embed.add_argument("--method", required=True)
    p_embed.add_argument(
        "--emb-v", required=True, help="Embedding version label, e.g., openai_t3small_v1"
    )
    p_embed.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default="fastembed",
    )
    p_embed.add_argument("--model-id", default="text-embedding-3-small")
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--retries", type=int, default=5)
    p_embed.add_argument("--sleep-base-ms", type=int, default=200)
    p_embed.add_argument(
        "--price-per-1k",
        type=float,
        default=0.02,
        help="Optional cost estimate (USD per 1K tokens)",
    )
    p_embed.add_argument("--duckdb", action="store_true")
    p_embed.set_defaults(func=cmd_embed)

    p_stats = sub.add_parser("stats", help="Summarize embeddings for an emb_v")
    p_stats.add_argument("--emb-v", required=True)
    p_stats.set_defaults(func=cmd_stats)

    p_sync = sub.add_parser(
        "sync", help="Idempotently upsert existing embeddings parquet into DuckDB"
    )
    p_sync.add_argument("--emb-v", required=True)
    p_sync.add_argument("--episode-id", required=True)
    p_sync.set_defaults(func=cmd_sync)

    p_manifest = sub.add_parser("manifest", help="Write manifest for an emb_v")
    p_manifest.add_argument("--emb-v", required=True)
    p_manifest.set_defaults(func=cmd_manifest)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
