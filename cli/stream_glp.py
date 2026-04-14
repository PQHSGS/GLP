"""Stream activations from LLM and train GLP in a single process to save disk space."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma2_pipeline.streaming import stream_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream activations and train GLP")
    
    # Model & extraction args
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Precision for the underlying LLM extractor model.")
    
    # Dataset args
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=None, help="Stop streaming texts after reading this many documents")
    
    # Stream/Train args
    parser.add_argument("--stream-chunk-size", type=int, default=64000, help="Number of activations per chunk")
    parser.add_argument("--total-steps", type=int, default=250000, help="Total number of gradient steps")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    
    parser.add_argument("--save-root", default=".")
    parser.add_argument("--run-name", default="glp-stream")
    parser.add_argument("--checkpoint-token-step", type=int, default=None, help="Save a checkpoint every N tokens (e.g. 100000000 for 100M)")
    
    
    parser.add_argument("--denoiser-layers", type=int, default=3)
    parser.add_argument("--d-model-mult", type=int, default=2)
    parser.add_argument("--d-mlp-mult", type=int, default=4)
    
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="glp")
    return parser


def run(args: argparse.Namespace) -> None:
    stream_train(args)


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
