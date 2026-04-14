"""Write GLP train config from cached activations and optionally launch training.

This script is intended for quick usage on Kaggle or other notebook VMs.
It infers activation dimensionality from the memmap dataset folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gemma2_pipeline.settings import GemmaTrainConfig

def build_train_config_dict(config: GemmaTrainConfig) -> dict:
    run_name = config.run_name
    output_path = f"{config.save_root}/runs/{run_name}"
    rep_statistic = config.rep_statistic or f"{config.train_dataset}/rep_statistics.pt"

    return {
        "save_root": config.save_root,
        "model_name": config.model_name,
        "device": config.device,
        "run_name": run_name,
        "output_path": output_path,
        "wandb_enabled": config.wandb_enabled,
        "wandb_project": config.wandb_project,
        "wandb_run_name": run_name,
        "train_dataset": config.train_dataset,
        "rep_statistic": rep_statistic,
        "num_epochs": config.num_epochs,
        "save_epochs": config.save_epochs,
        "shuffle": config.shuffle,
        "glp_kwargs": {
            "normalizer_config": {
                "rep_statistic": "${rep_statistic}",
            },
            "denoiser_config": {
                "d_input": config.d_input,
                "d_model": config.d_model,
                "d_mlp": config.d_mlp,
                "n_layers": config.denoiser_layers,
                "multi_layer_n_layers": config.multi_layer_n_layers,
            },
            "tracedict_config": {
                "layer_prefix": "model.layers",
                "layers": [config.layer],
                "retain": config.retain,
            },
        },
        "use_bf16": config.use_bf16,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping_threshold": config.gradient_clipping_threshold,
        "log_every_n_steps": config.log_every_n_steps,
        "save_opt_state": config.save_opt_state,
        "lr_scheduler": {
            "scheduler_cls": "cosine_scheduler_with_warmup",
            "warmup_ratio": config.warmup_ratio,
            "initial_factor": config.initial_factor,
            "final_factor": config.final_factor,
        },
    }

def write_train_config(
    config: GemmaTrainConfig,
    config_out_path: str | None = None,
) -> Path:
    out_path = Path(config_out_path or config.config_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_dict = build_train_config_dict(config)
    out_path.write_text(json.dumps(cfg_dict, indent=2))
    return out_path


def infer_hidden_size(dataset_dir: Path) -> int:
    indices_path = dataset_dir / "data_indices.npy"
    if not indices_path.exists():
        raise FileNotFoundError(f"Missing index file: {indices_path}")

    indices = np.load(indices_path)
    if indices.size == 0:
        raise ValueError(f"Activation dataset is empty: {dataset_dir}")

    # Each row is (file_idx, start_idx, end_idx), where chunk length equals d_input.
    first = indices[0]
    return int(first[2] - first[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare GLP config from activation memmaps and run training"
    )
    parser.add_argument("--train-dataset", required=True, help="Path to memmap activation dataset folder")
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--run-name", default="glp-custom-d3-static")
    parser.add_argument("--save-root", default=".")
    parser.add_argument("--config-out", default="configs/train_custom_static.yaml")

    parser.add_argument("--denoiser-layers", type=int, default=3)
    parser.add_argument("--d-model-mult", type=int, default=2)
    parser.add_argument("--d-mlp-mult", type=int, default=4)

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="glp")

    parser.add_argument("--write-only", action="store_true", help="Only write config, do not start training")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--working-dir", default=".")
    return parser


def run(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.train_dataset).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    d_input = infer_hidden_size(dataset_dir)
    d_model = args.d_model_mult * d_input
    d_mlp = args.d_mlp_mult * d_input

    train_cfg = GemmaTrainConfig(
        save_root=args.save_root,
        model_name=args.model_name,
        run_name=args.run_name,
        train_dataset=str(dataset_dir),
        rep_statistic=str(dataset_dir / "rep_statistics.pt"),
        num_epochs=args.num_epochs,
        d_input=d_input,
        d_model=d_model,
        d_mlp=d_mlp,
        denoiser_layers=args.denoiser_layers,
        layer=args.layer,
        device=args.device,
        use_bf16=args.use_bf16,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        config_out_path=args.config_out,
    )

    config_path = write_train_config(train_cfg, args.config_out)
    print(f"Wrote training config: {config_path}")

    summary = {
        "d_input": d_input,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "config": str(config_path),
    }
    print(json.dumps(summary, indent=2))

    if args.write_only:
        return

    from unittest.mock import patch
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from glp_train import main as glp_train_main

    print(f"Launching GLP training natively with config: {config_path}")
    with patch("sys.argv", ["glp_train.py", f"config={config_path}"]):
        glp_train_main()


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
