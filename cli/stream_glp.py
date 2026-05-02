"""Stream activations from LLM and train GLP in a single process to save disk space."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma2_pipeline.settings import (
    make_default_activation_collection_config,
    make_default_model_train_config,
)
from gemma2_pipeline.streaming import stream_train


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    collect_defaults = make_default_activation_collection_config()
    source_defaults = collect_defaults.fineweb
    train_defaults = make_default_model_train_config()
    d_model_mult_default = max(1, train_defaults.d_model // train_defaults.d_input)
    d_mlp_mult_default = max(1, train_defaults.d_mlp // train_defaults.d_input)

    parser = argparse.ArgumentParser(description="Stream activations and train GLP", add_help=add_help)
    
    # Model & extraction args
    parser.add_argument("--model-name", default=collect_defaults.model_name)
    parser.add_argument("--layer", type=int, default=collect_defaults.layer)
    parser.add_argument("--layer-prefix", type=str, default=collect_defaults.layer_prefix, help="Prefix for layers to extract.")
    parser.add_argument("--retain", choices=["input", "output"], default=train_defaults.retain)
    parser.add_argument("--max-length", type=int, default=collect_defaults.max_length)
    parser.add_argument("--token-idx", choices=["last", "all", "random_doc"], default=collect_defaults.token_idx)
    parser.add_argument("--sample-seed", type=int, default=collect_defaults.sample_seed)
    parser.add_argument("--drop-bos", action=argparse.BooleanOptionalAction, default=collect_defaults.drop_bos)
    parser.add_argument("--padding-side", choices=["left", "right"], default=collect_defaults.padding_side)
    parser.add_argument("--document-batch-size", type=int, default=collect_defaults.document_batch_size)
    parser.add_argument("--forward-batch-size", type=int, default=collect_defaults.forward_batch_size)
    parser.add_argument("--device", default=collect_defaults.device)
    parser.add_argument(
        "--phase-switch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload the inactive model between extraction and GLP training phases to reduce peak VRAM.",
    )
    parser.add_argument(
        "--offload-device",
        default="cpu",
        help="Device used to park the inactive model during phase switching (usually cpu).",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default=collect_defaults.torch_dtype,
        help="Precision for the underlying LLM extractor model.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=["float32", "float16", "bfloat16"],
        default=collect_defaults.storage_dtype,
        help="Temporary memmap dtype for streamed activation chunks.",
    )
    
    # Dataset args
    parser.add_argument("--dataset-name", default=source_defaults.dataset_name)
    parser.add_argument("--dataset-config", default=source_defaults.dataset_config)
    parser.add_argument("--dataset-split", default=source_defaults.split, help="Dataset split to stream from.")
    parser.add_argument("--text-field", default=source_defaults.text_field)
    parser.add_argument("--max-documents", type=int, default=source_defaults.max_documents, help="Stop streaming texts after reading this many documents")
    
    # Stream/Train args
    parser.add_argument("--stream-chunk-size", type=int, default=1000000, help="Number of activations per chunk")
    parser.add_argument("--total-steps", type=int, default=244, help="Total number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=train_defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=train_defaults.learning_rate)
    parser.add_argument(
        "--normalization-method",
        default=train_defaults.normalization_method,
        help=(
            "Latent normalization method. Examples: gaussian, log_norm, rmsnorm, "
            "iqr, quantile_99, quantile_97, 99, 97, 0.99"
        ),
    )
    parser.add_argument(
        "--noise-sampling-method",
        dest="noise_sampling_method",
        choices=["uniform", "ot"],
        default=train_defaults.noise_sampling_method,
        help="Training noise pairing method. 'uniform' keeps random batch pairing; 'ot' uses minibatch optimal transport.",
    )
    parser.add_argument(
        "--u-sampling-method",
        choices=["uniform", "beta"],
        default=train_defaults.u_sampling_method,
        help="Curriculum for sampling flow-matching u values. 'beta' uses Beta(5, 1) to bias samples toward the end of the schedule.",
    )
    parser.add_argument(
        "--ot-chunk-size",
        type=int,
        default=train_defaults.ot_chunk_size,
        help="Chunk size for minibatch OT matching. Smaller chunks reduce Hungarian cost; ignored unless --noise-sampling-method=ot.",
    )
    parser.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=train_defaults.split,
        help="Use separate denoiser output heads for top-variance dimensions and remaining dimensions.",
    )
    parser.add_argument(
        "--split-proportion",
        type=float,
        default=train_defaults.split_proportion,
        help="Proportion of highest-variance dimensions routed through the split tail output head.",
    )
    parser.add_argument("--gradient-clipping-threshold", type=float, default=train_defaults.gradient_clipping_threshold)
    parser.add_argument("--log-every-n-steps", type=int, default=train_defaults.log_every_n_steps)
    parser.add_argument("--tail-aware-weight", type=float, default=train_defaults.tail_aware_weight, help="Tail aggression alpha. 0 disables tail-aware weighting.")
    parser.add_argument("--tail-aware-start", type=int, default=train_defaults.tail_aware_start, help="Enable tail-aware loss weighting at this optimizer step (before that: plain MSE).")
    parser.add_argument("--tail-aware-min-weight", type=float, default=train_defaults.tail_aware_min_weight, help="Minimum raw-magnitude loss multiplier.")
    parser.add_argument("--tail-aware-max-weight", type=float, default=train_defaults.tail_aware_max_weight, help="Maximum raw-magnitude loss multiplier. <=0 disables max clamping.")
    parser.add_argument("--warmup-ratio", type=float, default=train_defaults.warmup_ratio)
    parser.add_argument("--initial-factor", type=float, default=train_defaults.initial_factor)
    parser.add_argument("--final-factor", type=float, default=train_defaults.final_factor)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=train_defaults.use_bf16)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=train_defaults.shuffle)
    parser.add_argument("--init-ckpt", default=train_defaults.init_ckpt, help="Optional local checkpoint folder or Hugging Face repo id/subfolder to initialize/finetune from.")
    parser.add_argument("--load-opt", action=argparse.BooleanOptionalAction, default=train_defaults.load_opt, help="Load optimizer/scheduler state from init checkpoint when opt.pt exists.")
    
    parser.add_argument("--save-root", default=".")
    parser.add_argument("--run-name", default="glp-stream")
    parser.add_argument("--checkpoint-token-step", type=int, default=100000000, help="Save a checkpoint every N tokens")
    
    
    parser.add_argument("--denoiser-layers", type=int, default=train_defaults.denoiser_layers)
    parser.add_argument("--d-model-mult", type=int, default=d_model_mult_default)
    parser.add_argument("--d-mlp-mult", type=int, default=d_mlp_mult_default)
    
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=train_defaults.wandb_enabled)
    parser.add_argument("--wandb-project", default=train_defaults.wandb_project)
    return parser


def run(args: argparse.Namespace) -> None:
    stream_train(args)


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
