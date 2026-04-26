"""Unified user-facing CLI for common GLP workflows.

Subcommands:
- collect: cache activations from a model+dataset
- train: write training config and launch GLP training
- stream: end-to-end memory-efficient streaming pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collect_acts import build_parser as build_collect_parser
    from collect_acts import run as collect_run
    from stream_glp import build_parser as build_stream_parser
    from stream_glp import run as stream_run
else:
    from .collect_acts import build_parser as build_collect_parser
    from .collect_acts import run as collect_run
    from .stream_glp import build_parser as build_stream_parser
    from .stream_glp import run as stream_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GLP user workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "collect",
        help="Collect activation dataset",
        parents=[build_collect_parser(add_help=False)],
    )


    subparsers.add_parser(
        "stream",
        help="Stream activations and train in a single process",
        parents=[build_stream_parser(add_help=False)],
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Delegate to existing scripts so behavior stays identical.
    if args.command == "collect":
        collect_run(args)
        return
    if args.command == "stream":
        stream_run(args)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
