import json
import subprocess
from pathlib import Path

from .settings import GemmaTrainConfig


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
    # JSON is a valid YAML subset, so glp_train.py (OmegaConf loader) can read this.
    out_path.write_text(json.dumps(cfg_dict, indent=2))
    return out_path


def launch_training(
    config_path: str,
    *,
    python_bin: str = "python3",
    working_dir: str | None = None,
) -> None:
    """Launch the existing GLP trainer with a generated config file."""
    root_dir = Path(working_dir) if working_dir else Path(__file__).resolve().parents[1]
    absolute_config = str(Path(config_path).resolve())
    command = [python_bin, "glp_train.py", f"config={absolute_config}"]
    subprocess.run(command, check=True, cwd=root_dir)
