from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.console import Console

from .eval import evaluate_episode, write_report

console = Console()


def cmd_eval(args: Namespace):
    report = evaluate_episode(
        episode_id=args.episode_id,
        emb_v=args.emb_v,
        qa_csv=Path(args.qa_csv),
        ks=[int(k) for k in args.ks.split(",")],
        tolerance_s=args.tolerance_s,
        collection=args.collection,
    )
    write_report(report)


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="spike6", description="Spike 6 — Retrieval baseline (vector-only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ev = sub.add_parser("eval", help="Evaluate retrieval on QA set")
    p_ev.add_argument("--episode-id", required=True)
    p_ev.add_argument("--emb-v", required=True)
    p_ev.add_argument("--qa-csv", default="data/qa/labels.csv")
    p_ev.add_argument("--ks", default="5,10,20")
    p_ev.add_argument("--tolerance-s", type=int, default=7)
    p_ev.add_argument("--collection", help="Qdrant collection or alias (default: mw_chunks_live)")
    p_ev.set_defaults(func=cmd_eval)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
