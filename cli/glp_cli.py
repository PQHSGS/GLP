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
    from train_glp import build_parser as build_train_parser
    from train_glp import run as train_run
    from stream_glp import build_parser as build_stream_parser
    from stream_glp import run as stream_run
else:
    from .collect_acts import build_parser as build_collect_parser
    from .collect_acts import run as collect_run
    from .train_glp import build_parser as build_train_parser
    from .train_glp import run as train_run
    from .stream_glp import build_parser as build_stream_parser
    from .stream_glp import run as stream_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GLP user workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_cmd = subparsers.add_parser("collect", help="Collect activation dataset")
    for action in build_collect_parser()._actions:
        if action.dest in {"help"}:
            continue
        collect_cmd._add_action(action)

    train_cmd = subparsers.add_parser("train", help="Write config and run GLP training")
    for action in build_train_parser()._actions:
        if action.dest in {"help"}:
            continue
        train_cmd._add_action(action)

    stream_cmd = subparsers.add_parser("stream", help="Stream activations and train in a single process")
    for action in build_stream_parser()._actions:
        if action.dest in {"help"}:
            continue
        stream_cmd._add_action(action)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Delegate to existing scripts so behavior stays identical.
    if args.command == "collect":
        collect_run(args)
        return
    if args.command == "train":
        train_run(args)
        return
    if args.command == "stream":
        stream_run(args)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
