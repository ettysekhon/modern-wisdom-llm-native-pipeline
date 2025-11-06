from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from . import paths
from .io import load_chunks_df, load_embeddings
from .qdrant_ops import (
    VectorSpec,
    alias_set_live,
    client,
    collection_name_for_emb_v,
    ensure_collection,
    upsert_points,
)

console = Console()

# Configure logging with Rich handler for better output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


def cmd_upsert(args: Namespace) -> None:
    """Upsert embeddings into Qdrant and optionally set live alias."""
    emb_df = load_embeddings(args.emb_v, args.episode_id)
    dim = len(emb_df.iloc[0]["vector"])
    spec = VectorSpec(size=dim, distance="COSINE")
    collection = collection_name_for_emb_v(args.emb_v)

    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)
    ensure_collection(cli, collection, spec)
    upsert_points(cli, collection, emb_df.to_dict("records"))

    if args.set_live:
        alias_set_live(cli, collection, alias=args.live_alias)
        console.print(f"[green]Alias set:[/green] {args.live_alias} → {collection}")


def cmd_upsert_batch(args: Namespace) -> None:
    """Upsert embeddings for multiple episodes from a file."""
    if args.episode_list:
        episode_file = Path(args.episode_list)
        if not episode_file.exists():
            console.print(f"[red]Episode list file not found: {episode_file}[/red]")
            sys.exit(1)
        episode_ids = [
            line.strip() for line in episode_file.read_text().splitlines() if line.strip()
        ]
    else:
        console.print("[red]--episode-list is required for batch upsert[/red]")
        sys.exit(1)

    collection = collection_name_for_emb_v(args.emb_v)
    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    # Determine dimension from first episode
    try:
        first_ep = episode_ids[0]
        emb_df = load_embeddings(args.emb_v, first_ep)
        dim = len(emb_df.iloc[0]["vector"])
        spec = VectorSpec(size=dim, distance="COSINE")
        ensure_collection(cli, collection, spec)
    except Exception as e:
        console.print(f"[red]Failed to initialize collection: {e}[/red]")
        sys.exit(1)

    # Upsert each episode
    for i, ep_id in enumerate(episode_ids, 1):
        try:
            console.print(f"[cyan]Upserting {i}/{len(episode_ids)}: {ep_id}[/cyan]")

            # Load embeddings
            emb_df = load_embeddings(args.emb_v, ep_id)

            # Load chunks to get text field
            try:
                chunks_df = load_chunks_df("sentence_bound", ep_id)
                # Merge embeddings with chunks to include text
                merged_df = emb_df.merge(
                    chunks_df[["chunk_id", "text", "start_ts", "end_ts", "publish_date"]],
                    on="chunk_id",
                    how="left",
                )
                # Fill missing text with empty string
                merged_df["text"] = merged_df["text"].fillna("")
                console.print(f"  [dim]Merged {len(merged_df)} chunks with text[/dim]")
            except Exception as chunk_err:
                console.print(
                    f"  [yellow]Warning: Could not load chunks for {ep_id}: {chunk_err}[/yellow]"
                )
                console.print("  [yellow]Upserting without text field[/yellow]")
                merged_df = emb_df

            upsert_points(cli, collection, merged_df.to_dict("records"))
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to upsert {ep_id}: {e}[/yellow]")
            continue

    if args.set_live:
        alias_set_live(cli, collection, alias=args.live_alias)
        console.print(f"[green]Alias set:[/green] {args.live_alias} → {collection}")


def cmd_check(args: Namespace) -> None:
    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    if args.collection:
        name = args.collection
    elif args.emb_v:
        name = collection_name_for_emb_v(args.emb_v)
    else:
        console.print("[red]You must pass --collection or --emb-v[/red]")
        return

    try:
        info = cli.get_collection(name)
        count = cli.count(name, exact=True)

        # Vector parameters are under config.params.vectors
        vectors_cfg = getattr(info.config.params, "vectors", None)
        vec_size = getattr(vectors_cfg, "size", "unknown")
        vec_distance = getattr(vectors_cfg, "distance", "unknown")

        console.print(
            f"[cyan]{name}[/cyan]: status={info.status}, "
            f"vector_size={vec_size}, distance={vec_distance}, points={count.count}"
        )

        # Per-collection aliases
        aliases_for_collection = cli.get_collection_aliases(collection_name=name)
        alias_names = [a.alias_name for a in aliases_for_collection.aliases]
        console.print(f"Aliases for {name}: {alias_names}")
    except Exception as e:
        console.print(f"[red]Error checking collection: {e}[/red]")


def cmd_clear(args: Namespace) -> None:
    """Clear all points from a collection."""
    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    if args.collection:
        name = args.collection
    elif args.emb_v:
        name = collection_name_for_emb_v(args.emb_v)
    else:
        console.print("[red]You must pass --collection or --emb-v[/red]")
        return

    if not args.yes:
        response = input(f"Are you sure you want to clear collection '{name}'? (yes/no): ")
        if response.lower() != "yes":
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        from qdrant_client.models import FilterSelector

        cli.delete(collection_name=name, points_selector=FilterSelector())  # pyright: ignore[reportCallIssue]
        console.print(f"[green]Cleared collection: {name}[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing collection: {e}[/red]")


def cmd_list_collections(args: Namespace) -> None:
    """List all collections."""
    import time

    from qdrant_client.http.exceptions import ResponseHandlingException

    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    max_retries = 3
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            collections = cli.get_collections()
            if not collections.collections:
                console.print("[yellow]No collections found[/yellow]")
                return

            for coll in collections.collections:
                try:
                    count = cli.count(coll.name, exact=True)
                    aliases = cli.get_collection_aliases(collection_name=coll.name)
                    alias_names = [a.alias_name for a in aliases.aliases]
                    alias_str = f" (aliases: {', '.join(alias_names)})" if alias_names else ""
                    console.print(f"[cyan]{coll.name}[/cyan]: {count.count} points{alias_str}")
                except Exception:
                    console.print(f"[cyan]{coll.name}[/cyan]: (error getting info)")
            return  # Success, exit
        except (ResponseHandlingException, ConnectionError, OSError) as e:
            if attempt < max_retries - 1:
                delay = retry_delay * (attempt + 1)
                console.print(
                    f"[yellow]Connection failed (attempt {attempt + 1}/{max_retries}): {e}[/yellow]"
                )
                console.print(f"[yellow]Retrying in {delay}s...[/yellow]")
                time.sleep(delay)
                continue
            console.print(f"[red]Error listing collections: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Error listing collections: {e}[/red]")
            return


def build_parser() -> ArgumentParser:
    """Build command-line parser."""
    parser = ArgumentParser(
        prog="mw-rag",
        description="Modern Wisdom RAG Pipeline - Embedding Management",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- upsert ---
    p_up = sub.add_parser("upsert", help="Upsert embeddings into Qdrant")
    p_up.add_argument("--episode-id", required=True)
    p_up.add_argument("--emb-v", required=True)
    p_up.add_argument("--set-live", action="store_true")
    p_up.add_argument("--live-alias", default="mw_chunks_live")
    p_up.set_defaults(func=cmd_upsert)

    # --- upsert-batch ---
    p_up_batch = sub.add_parser("upsert-batch", help="Upsert embeddings for multiple episodes")
    p_up_batch.add_argument(
        "--episode-list", required=True, help="Path to file with episode IDs (one per line)"
    )
    p_up_batch.add_argument("--emb-v", required=True)
    p_up_batch.add_argument("--set-live", action="store_true")
    p_up_batch.add_argument("--live-alias", default="mw_chunks_live")
    p_up_batch.set_defaults(func=cmd_upsert_batch)

    # --- check ---
    p_ck = sub.add_parser("check", help="Inspect Qdrant collections and aliases")
    p_ck.add_argument("--emb-v", help="Embedding version to check")
    p_ck.add_argument("--collection", help="Collection name (optional)")
    p_ck.set_defaults(func=cmd_check)

    # --- clear ---
    p_clr = sub.add_parser("clear", help="Clear all points from a collection")
    p_clr.add_argument("--emb-v", help="Embedding version")
    p_clr.add_argument("--collection", help="Collection name")
    p_clr.add_argument("--yes", action="store_true", help="Skip confirmation")
    p_clr.set_defaults(func=cmd_clear)

    # --- list ---
    p_list = sub.add_parser("list", help="List all collections")
    p_list.set_defaults(func=cmd_list_collections)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
