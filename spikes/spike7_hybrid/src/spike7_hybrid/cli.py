from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.console import Console

from . import paths
from .eval import evaluate_hybrid, write_report
from .schema import Filters, HybridParams

console = Console()


def cmd_eval(args: Namespace):
    params = HybridParams(
        k_vec=args.k_vec,
        k_lex=args.k_lex,
        rrf_k=args.rrf_k,
        tolerance_s=args.tolerance_s,
        ks_report=[int(k) for k in args.ks.split(",")],
    )
    filters = Filters(
        guest=args.guest,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    report = evaluate_hybrid(
        episode_id=args.episode_id,
        emb_v=args.emb_v,
        method=args.method,
        qa_csv=Path(args.qa_csv),
        collection=args.collection or paths.LIVE_ALIAS,
        params=params,
        filters=filters,
    )
    write_report(report)


def build_parser():
    p = ArgumentParser(
        prog="spike7", description="Spike 7 — Hybrid retrieval (vector + BM25 + filters)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="Evaluate hybrid retrieval for one episode")
    ev.add_argument("--episode-id", required=True)
    ev.add_argument("--emb-v", required=True)
    ev.add_argument("--method", default="sentence_bound")
    ev.add_argument("--qa-csv", default="data/qa/labels.csv")
    ev.add_argument("--collection", help="Collection or alias (default: mw_chunks_live)")
    ev.add_argument("--ks", default="5,10,20")
    ev.add_argument("--tolerance-s", type=int, default=7)
    ev.add_argument("--k-vec", type=int, default=20)
    ev.add_argument("--k-lex", type=int, default=200)
    ev.add_argument("--rrf-k", type=float, default=60.0)
    # Optional metadata/date filters
    ev.add_argument("--guest")
    ev.add_argument("--date-from")
    ev.add_argument("--date-to")

    ev.add_argument(
        "--embed-questions", action="store_true", help="Use FastEmbed to embed QA questions"
    )

    ev.set_defaults(func=cmd_eval)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)
