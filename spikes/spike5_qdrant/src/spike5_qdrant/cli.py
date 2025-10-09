from argparse import ArgumentParser, Namespace

from rich.console import Console

from . import paths
from .io import load_embeddings
from .qdrant import alias_set_live, client, ensure_collection, upsert_points
from .schema import VectorSpec, collection_name_for_emb_v

console = Console()


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


def cmd_check(args: Namespace) -> None:
    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)

    if args.collection:
        name = args.collection
    elif args.emb_v:
        name = collection_name_for_emb_v(args.emb_v)
    else:
        console.print("[red]You must pass --collection or --emb-v[/red]")
        return

    info = cli.get_collection(name)
    count = cli.count(name, exact=True)

    # vector params are under config.params.vectors
    vectors_cfg = getattr(info.config.params, "vectors", None)
    vec_size = getattr(vectors_cfg, "size", "unknown")
    vec_distance = getattr(vectors_cfg, "distance", "unknown")

    console.print(
        f"[cyan]{name}[/cyan]: status={info.status}, "
        f"vector_size={vec_size}, distance={vec_distance}, points={count.count}"
    )

    # ✅ per-collection aliases
    aliases_for_collection = cli.get_collection_aliases(collection_name=name)
    alias_names = [a.alias_name for a in aliases_for_collection.aliases]
    console.print(f"Aliases for {name}: {alias_names}")


def cmd_query(args: Namespace) -> None:
    """Run a sample vector query to confirm retrieval."""
    emb_df = load_embeddings(args.emb_v, args.episode_id)
    q_vector = emb_df.iloc[0]["vector"]

    cli = client(paths.QDRANT_URL, paths.QDRANT_API_KEY)
    collection = args.collection or collection_name_for_emb_v(args.emb_v)

    results = cli.search(
        collection_name=collection,
        query_vector=q_vector,
        limit=args.top_k,
    )

    for point in results:
        payload = dict(getattr(point, "payload", {}) or {})
        episode_id = payload.get("episode_id", "n/a")
        console.print(f"id={point.id} score={point.score} episode={episode_id}")


def build_parser() -> ArgumentParser:
    """Build command-line parser."""
    parser = ArgumentParser(
        prog="spike5",
        description="Spike 5 — Qdrant collection & blue/green alias ops",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- upsert ---
    p_up = sub.add_parser("upsert", help="Upsert embeddings into Qdrant")
    p_up.add_argument("--episode-id", required=True)
    p_up.add_argument("--method", required=True)
    p_up.add_argument("--emb-v", required=True)
    p_up.add_argument("--set-live", action="store_true")
    p_up.add_argument("--live-alias", default="mw_chunks_live")
    p_up.set_defaults(func=cmd_upsert)

    # --- check ---
    p_ck = sub.add_parser("check", help="Inspect Qdrant collections and aliases")
    p_ck.add_argument("--emb-v", help="Embedding version to check")
    p_ck.add_argument("--collection", help="Collection name (optional)")
    p_ck.set_defaults(func=cmd_check)

    # --- query ---
    p_q = sub.add_parser("query", help="Query Qdrant with sample embedding")
    p_q.add_argument("--emb-v", required=True)
    p_q.add_argument("--episode-id", required=True)
    p_q.add_argument("--collection")
    p_q.add_argument("--top-k", type=int, default=5)
    p_q.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
