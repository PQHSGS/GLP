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
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--retain", choices=["input", "output"], default="output")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--token-idx", choices=["last", "all", "random_doc"], default="all")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--drop-bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--padding-side", choices=["left", "right"], default="right")
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Precision for the underlying LLM extractor model.")
    parser.add_argument("--storage-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16", help="Temporary memmap dtype for streamed activation chunks.")
    
    # Dataset args
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=50000, help="Stop streaming texts after reading this many documents")
    
    # Stream/Train args
    parser.add_argument("--stream-chunk-size", type=int, default=65536, help="Number of activations per chunk")
    parser.add_argument("--total-steps", type=int, default=244, help="Total number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-clipping-threshold", type=float, default=1.0)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--initial-factor", type=float, default=0.01)
    parser.add_argument("--final-factor", type=float, default=0.1)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stats-max-vectors", type=int, default=1000000, help="Number of vectors used to precompute normalization stats")
    
    parser.add_argument("--save-root", default=".")
    parser.add_argument("--run-name", default="glp-stream")
    parser.add_argument("--checkpoint-token-step", type=int, default=1000000, help="Save a checkpoint every N tokens")
    
    
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
