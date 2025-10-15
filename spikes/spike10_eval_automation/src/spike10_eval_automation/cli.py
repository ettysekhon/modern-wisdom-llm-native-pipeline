from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import yaml

from . import paths
from .gate import gate
from .suite import run_suite


def cmd_suite(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = run_suite(cfg)
    print(f"[suite] Wrote {out}")


def cmd_report(_args):
    p = paths.REPORTS_DIR / "suite_report.json"
    print(p.read_text())


def cmd_gate(_args):
    p = paths.REPORTS_DIR / "suite_report.json"
    rc = gate(p)
    raise SystemExit(rc)


def build_parser():
    p = ArgumentParser(
        prog="spike10", description="Spike 10 — Eval Automation & CI Gate"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suite", help="Run batch evals from YAML")
    s.add_argument("--config", required=True)
    s.set_defaults(func=cmd_suite)

    sub.add_parser("report", help="Print last suite report").set_defaults(
        func=cmd_report
    )
    sub.add_parser("gate", help="Enforce thresholds").set_defaults(func=cmd_gate)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)
