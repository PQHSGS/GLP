import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from glp.denoiser import GLP
from glp.utils_acts import MemmapWriter
from glp_train import cosine_scheduler_with_warmup, load_activation_dataset, get_activation_dataloader

from .collect_acts import (
    build_tracedict_config,
    extract_activation_vectors,
    write_vectors_to_memmap,
)
from .loading import (
    get_storage_dtype,
    iter_fineweb_texts,
    load_model_and_tokenizer,
)
from .preprocess import batch_items
from .stats import RunningMoments, save_rep_statistics
from .settings import FineWebSourceConfig

LOGGER = logging.getLogger(__name__)

CHECKPOINT_FILES = ("config.yaml", "rep_statistics.pt", "final.safetensors", "opt.pt")



def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def _get_module_device(module: torch.nn.Module) -> str:
    for param in module.parameters():
        return str(param.device)
    for buffer in module.buffers():
        return str(buffer.device)
    return "cpu"


def _move_model_to_device(module: torch.nn.Module, target_device: str, module_name: str) -> None:
    target_device = str(target_device)
    current_device = _get_module_device(module)
    if current_device == target_device:
        return
    LOGGER.info("Moving %s from %s to %s.", module_name, current_device, target_device)
    module.to(target_device)


def _cleanup_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()




def canonicalize_normalization_method(method: str | None) -> str:
    method = "gaussian" if method is None else str(method)
    method = method.strip().lower().replace("-", "_")
    if method in {"lognorm"}:
        method = "log_norm"

    aliases = {
        "rms": "rmsnorm",
        "rms_norm": "rmsnorm",
        "zscore": "gaussian",
        "z_score": "gaussian",
    }
    method = aliases.get(method, method)

    if method in {"gaussian", "log_norm", "rmsnorm", "iqr"}:
        return method

    if method == "quantile":
        return "quantile_99"

    if method.startswith("quantile_"):
        quantile_raw = method.split("_", 1)[1]
        quantile_percent = parse_quantile_percent(quantile_raw)
        return f"quantile_{format_quantile_percent(quantile_percent)}"

    # Allow shorthand numeric input like "99" or "0.99".
    try:
        quantile_percent = parse_quantile_percent(method)
        return f"quantile_{format_quantile_percent(quantile_percent)}"
    except ValueError:
        pass

    raise ValueError(
        f"Unsupported normalization_method '{method}'. "
        "Expected one of ['gaussian', 'log_norm', 'rmsnorm', 'iqr', 'quantile_XX', '99', '0.99']."
    )


def parse_quantile_percent(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid quantile specification '{raw_value}'. "
            "Use values like 99, 97, or 0.99."
        ) from exc

    if value <= 1.0:
        value *= 100.0

    if not (0.0 < value < 100.0):
        raise ValueError(
            f"Quantile percent must be in (0, 100), got {value}."
        )
    return value


def format_quantile_percent(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def normalization_requires_stats(method: str) -> bool:
    return method != "log_norm"


def quantile_percent_from_method(method: str) -> float | None:
    if method.startswith("quantile_"):
        return float(method.split("_", 1)[1])
    return None

def setup_glp_model(hidden_size, args):
    d_model = args.d_model_mult * hidden_size
    d_mlp = args.d_mlp_mult * hidden_size
    normalization_method = canonicalize_normalization_method(
        getattr(args, "normalization_method", "gaussian")
    )
    
    model = GLP(
        normalizer_config={
            "rep_statistic": "",
            "d_input": hidden_size,
            "normalization_method": normalization_method,
        },  # Initializes identity buffer of right shape
        denoiser_config={
            "d_input": hidden_size,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "n_layers": args.denoiser_layers,
            "multi_layer_n_layers": None,
        },
        noise_sampling_method=getattr(args, "noise_sampling_method", "uniform"),
        u_sampling_method=getattr(args, "u_sampling_method", "uniform"),
        ot_chunk_size=getattr(args, "ot_chunk_size", 256),
        tracedict_config={
            "layer_prefix": getattr(args, "layer_prefix", "model.layers"),
            "layers": [args.layer],
            "retain": args.retain,
        }
    )
    return model


def resolve_init_checkpoint(init_ckpt):
    if not init_ckpt:
        return None

    resolved = Path(init_ckpt).expanduser()
    if not resolved.exists():
        repo_id, subfolder = split_hf_checkpoint_ref(init_ckpt)
        allow_patterns = [
            f"{subfolder}/{name}" if subfolder else name
            for name in CHECKPOINT_FILES
        ]
        resolved = Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
        ))
        if subfolder:
            resolved = resolved / subfolder

    return resolved


def split_hf_checkpoint_ref(checkpoint_ref):
    parts = checkpoint_ref.strip("/").split("/")
    if len(parts) <= 2:
        return checkpoint_ref, ""
    return "/".join(parts[:2]), "/".join(parts[2:])


def setup_init_glp_model(init_dir, hidden_size, args):
    config = OmegaConf.load(str(init_dir / "config.yaml"))
    if "glp_kwargs" not in config:
        raise ValueError(f"Checkpoint config at {init_dir / 'config.yaml'} does not contain glp_kwargs.")

    normalizer_config = config.glp_kwargs.normalizer_config
    rep_stats_path = init_dir / "rep_statistics.pt"
    if rep_stats_path.exists():
        normalizer_config.rep_statistic = str(rep_stats_path)
    else:
        normalizer_config.rep_statistic = ""
    normalizer_config.d_input = hidden_size

    tracedict_config = {
        "layer_prefix": getattr(args, "layer_prefix", "model.layers"),
        "layers": [args.layer],
        "retain": args.retain,
    }
    model = GLP(
        normalizer_config=normalizer_config,
        denoiser_config=config.glp_kwargs.denoiser_config,
        noise_sampling_method=getattr(
            config.glp_kwargs,
            "noise_sampling_method",
            getattr(args, "noise_sampling_method", "uniform"),
        ),
        u_sampling_method=getattr(config.glp_kwargs, "u_sampling_method", getattr(args, "u_sampling_method", "uniform")),
        ot_chunk_size=getattr(config.glp_kwargs, "ot_chunk_size", getattr(args, "ot_chunk_size", 256)),
        tracedict_config=tracedict_config,
    )
    model.load_pretrained(init_dir, name="final")
    return model

def stream_train(args):
    device = args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    offload_device = str(getattr(args, "offload_device", "cpu"))
    phase_switch = bool(getattr(args, "phase_switch", False))
    if phase_switch and not _is_cuda_device(device):
        LOGGER.warning("Phase switching was enabled but training device is %s; disabling phase switch.", device)
        phase_switch = False
    if phase_switch and str(device) == offload_device:
        LOGGER.warning(
            "Phase switching requested with identical train/offload device (%s); disabling phase switch.",
            device,
        )
        phase_switch = False

    init_dir = resolve_init_checkpoint(getattr(args, "init_ckpt", None))
    if init_dir is not None:
        init_config = OmegaConf.load(str(init_dir / "config.yaml"))
        init_normalizer_config = init_config.glp_kwargs.normalizer_config
        normalization_method = canonicalize_normalization_method(
            getattr(init_normalizer_config, "normalization_method", getattr(args, "normalization_method", "gaussian"))
        )
    else:
        normalization_method = canonicalize_normalization_method(
            getattr(args, "normalization_method", "gaussian")
        )
    use_stats = normalization_requires_stats(normalization_method)
    use_gaussian_stats = normalization_method == "gaussian"
    use_rmsnorm_stats = normalization_method == "rmsnorm"
    use_iqr_stats = normalization_method == "iqr"
    quantile_percent = quantile_percent_from_method(normalization_method)
    use_quantile_stats = quantile_percent is not None
    split_requested = bool(getattr(args, "split", False))
    
    # 1. Load Gemma Extractor
    LOGGER.info(f"Loading extractor model {args.model_name}")
    hf_model, hf_tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=device,
        torch_dtype_name=getattr(args, "torch_dtype", "bfloat16"),
    )
    hf_model.requires_grad_(False)
    hidden_size = int(hf_model.config.hidden_size)
    
    LOGGER.info("Setting up GLP denoiser")
    if init_dir is not None:
        LOGGER.info("Initializing GLP from %s.", init_dir)
        glp_model = setup_init_glp_model(init_dir, hidden_size, args).to(device)
    else:
        glp_model = setup_glp_model(hidden_size, args).to(device)
    total_steps = args.total_steps

    from functools import partial

    def create_optimizer_and_scheduler():
        # Baseline optimizer: plain AdamW over all trainable GLP parameters.
        optimizer = torch.optim.AdamW(glp_model.parameters(), lr=args.learning_rate)
        lr_lambda = partial(
            cosine_scheduler_with_warmup,
            warmup_steps=int(getattr(args, "warmup_ratio", 0.01) * total_steps),
            max_steps=total_steps,
            initial_factor=getattr(args, "initial_factor", 0.01),
            final_factor=getattr(args, "final_factor", 0.01),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    opt_adamw = None
    sched_adamw = None

    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    else:
        wandb_run = None

    tmp_dir = Path(f"data/tmp_stream_{args.run_name}")
    stats = RunningMoments(hidden_size) if (use_gaussian_stats or use_quantile_stats) else None
    second_moment_sum = np.zeros(hidden_size, dtype=np.float64) if use_rmsnorm_stats else None
    second_moment_count = 0
    iqr_q25 = np.zeros(hidden_size, dtype=np.float64) if use_iqr_stats else None
    iqr_median = np.zeros(hidden_size, dtype=np.float64) if use_iqr_stats else None
    iqr_q75 = np.zeros(hidden_size, dtype=np.float64) if use_iqr_stats else None
    iqr_count = 0
    quantile_scale = np.ones(hidden_size, dtype=np.float64) if use_quantile_stats else None
    quantile_count = 0
    fineweb = FineWebSourceConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=getattr(args, "dataset_split", "train"),
        text_field=args.text_field,
        max_documents=args.max_documents,
        streaming=True
    )

    def build_batch_iterator():
        text_iter = iter_fineweb_texts(fineweb)
        return batch_items(text_iter, args.document_batch_size)

    rng = np.random.default_rng(getattr(args, "sample_seed", 0))
    
    global_step = 0
    pbar = tqdm(total=total_steps, desc="Streaming GLP")
    
    total_tokens_collected = 0
    next_checkpoint_target = args.checkpoint_token_step if getattr(args, "checkpoint_token_step", None) else float('inf')
    
    def save_glp_checkpoint(save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        glp_model.save_pretrained(path=save_dir, name="final")
        if use_stats:
            torch.save(
                {
                    "mean": glp_model.normalizer.mean.cpu(),
                    "var": glp_model.normalizer.var.cpu(),
                    "normalization_method": normalization_method,
                },
                save_dir / "rep_statistics.pt",
            )
        
        import yaml
        denoiser_model = glp_model.denoiser.model
        split_tail_indices = list(glp_model.denoiser.model.split_tail_indices)
        config_dict = {
            "model_name": args.model_name,
            "glp_kwargs": {
                "normalizer_config": {
                    "rep_statistic": "rep_statistics.pt",
                    "d_input": hidden_size,
                    "normalization_method": normalization_method,
                },
                "denoiser_config": {
                    "d_input": hidden_size,
                    "d_model": denoiser_model.d_model,
                    "d_mlp": denoiser_model.d_mlp,
                    "n_layers": denoiser_model.n_layers,
                    "multi_layer_n_layers": denoiser_model.multi_layer_n_layers,
                    "split": bool(denoiser_model.split),
                    "split_tail_indices": split_tail_indices,
                },
                "noise_sampling_method": glp_model.noise_sampling_method,
                "ot_chunk_size": glp_model.ot_chunk_size,
                "tracedict_config": {
                    "layer_prefix": getattr(args, "layer_prefix", "model.layers"),
                    "layers": [args.layer],
                    "retain": args.retain,
                }
            }
        }
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)

        if opt_adamw is not None and sched_adamw is not None:
            torch.save(
                {
                    "optimizer": opt_adamw.state_dict(),
                    "scheduler": sched_adamw.state_dict(),
                    "global_step": global_step,
                    "total_tokens_collected": total_tokens_collected,
                },
                save_dir / "opt.pt",
            )
    
    tracedict_config = build_tracedict_config(layer=args.layer, retain=args.retain, layer_prefix=getattr(args, "layer_prefix", "model.layers"))
    use_autocast = bool(getattr(args, "use_bf16", True) and ("cuda" in str(device)))

    # Use a single streaming pass and update normalization stats cumulatively per chunk.
    batch_iterator = build_batch_iterator()
    
    while global_step < total_steps:
        if phase_switch:
            _move_model_to_device(glp_model, offload_device, "GLP model")
            _move_model_to_device(hf_model, device, "Extractor LLM")
            _cleanup_cuda_cache()

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        stream_storage_dtype = getattr(args, "storage_dtype", "bfloat16")
        file_size = args.stream_chunk_size * hidden_size
        np_dtype, dtype_label = get_storage_dtype(stream_storage_dtype)
        writer = MemmapWriter(output_dir=tmp_dir, file_size=file_size, dtype=np_dtype)
        (tmp_dir / "dtype.txt").write_text(dtype_label)
        
        vectors_written = 0
        while vectors_written < args.stream_chunk_size:
            text_batch = next(batch_iterator, None)
            if not text_batch:
                LOGGER.warning("Ran out of text batch data.")
                break
            
            vectors = extract_activation_vectors(
                hf_model=hf_model,
                hf_tokenizer=hf_tokenizer,
                text_batch=text_batch,
                tracedict_config=tracedict_config,
                padding_side=args.padding_side,
                token_idx=args.token_idx,
                forward_batch_size=args.forward_batch_size,
                max_length=args.max_length,
                drop_bos=args.drop_bos,
                rng=rng,
            )
            if vectors.numel() == 0: continue
            
            remaining = args.stream_chunk_size - vectors_written
            vectors = vectors[:remaining]
            if vectors.numel() == 0:
                continue

            if use_stats:
                vectors_np = vectors.detach().float().cpu().numpy().astype(np.float64, copy=False)

                if stats is not None:
                    stats.update(vectors_np)

                if use_rmsnorm_stats and second_moment_sum is not None:
                    second_moment_sum += np.square(vectors_np).sum(axis=0)
                    second_moment_count += vectors_np.shape[0]

                if use_iqr_stats and iqr_q25 is not None and iqr_median is not None and iqr_q75 is not None:
                    batch_count = vectors_np.shape[0]
                    chunk_q25 = np.percentile(vectors_np, 25, axis=0)
                    chunk_median = np.percentile(vectors_np, 50, axis=0)
                    chunk_q75 = np.percentile(vectors_np, 75, axis=0)
                    if iqr_count == 0:
                        iqr_q25 = chunk_q25
                        iqr_median = chunk_median
                        iqr_q75 = chunk_q75
                    else:
                        total = iqr_count + batch_count
                        iqr_q25 = (iqr_q25 * iqr_count + chunk_q25 * batch_count) / total
                        iqr_median = (iqr_median * iqr_count + chunk_median * batch_count) / total
                        iqr_q75 = (iqr_q75 * iqr_count + chunk_q75 * batch_count) / total
                    iqr_count += batch_count

                if use_quantile_stats and quantile_scale is not None and quantile_percent is not None:
                    # Center the vectors using the current running mean before quantile calculation
                    centered_vectors = vectors_np - stats.mean
                    chunk_q = np.percentile(np.abs(centered_vectors), quantile_percent, axis=0)
                    batch_count = vectors_np.shape[0]
                    if quantile_count == 0:
                        quantile_scale = chunk_q
                    else:
                        total = quantile_count + batch_count
                        quantile_scale = (quantile_scale * quantile_count + chunk_q * batch_count) / total
                    quantile_count += batch_count

            written = write_vectors_to_memmap(
                writer,
                vectors,
                stream_storage_dtype,
            )
            if written == 0:
                continue

            vectors_written += written
            
        writer.flush()
        if vectors_written == 0:
            LOGGER.warning("No vectors generated in this chunk (dataset exhausted?). Halting training loop early.")
            break

        normalizer_device = glp_model.normalizer.mean.device

        if use_gaussian_stats and stats is not None:
            mean, var = stats.finalize()
            glp_model.normalizer.mean = torch.tensor(mean, dtype=torch.float32, device=normalizer_device)
            glp_model.normalizer.var = torch.tensor(var, dtype=torch.float32, device=normalizer_device)
            LOGGER.info("Updated cumulative gaussian stats from %d vectors.", stats.count)
            
        elif use_rmsnorm_stats and second_moment_sum is not None:
            rms_sq = np.maximum(second_moment_sum / max(second_moment_count, 1), 1e-8)
            glp_model.normalizer.mean = torch.zeros(hidden_size, dtype=torch.float32, device=normalizer_device)
            glp_model.normalizer.var = torch.tensor(rms_sq, dtype=torch.float32, device=normalizer_device)
            LOGGER.info("Updated cumulative rmsnorm stats from %d vectors.", second_moment_count)

        elif use_iqr_stats and iqr_median is not None and iqr_q25 is not None and iqr_q75 is not None:
            iqr = np.maximum(iqr_q75 - iqr_q25, 1e-6)
            glp_model.normalizer.mean = torch.tensor(iqr_median, dtype=torch.float32, device=normalizer_device)
            glp_model.normalizer.var = torch.tensor(iqr * iqr, dtype=torch.float32, device=normalizer_device)
            LOGGER.info("Updated cumulative iqr stats from %d vectors.", iqr_count)
            
        elif use_quantile_stats and quantile_scale is not None:
            scale = np.maximum(quantile_scale, 1e-6)
            glp_model.normalizer.mean = torch.tensor(stats.mean, dtype=torch.float32, device=normalizer_device)
            glp_model.normalizer.var = torch.tensor(scale * scale, dtype=torch.float32, device=normalizer_device)
            LOGGER.info("Updated cumulative quantile-%s stats from %d vectors.", format_quantile_percent(quantile_percent), quantile_count)
            
        else:
            LOGGER.info("Using normalization_method=%s; skipped cumulative stat updates.", normalization_method)

        if phase_switch:
            _move_model_to_device(hf_model, offload_device, "Extractor LLM")
            _move_model_to_device(glp_model, device, "GLP model")
            _cleanup_cuda_cache()

        if opt_adamw is None:
            if split_requested and not glp_model.denoiser.model.split:
                if use_stats:
                    split_tail_indices = glp_model.configure_split_output_from_normalizer(
                        proportion=getattr(args, "split_proportion", 0.1)
                    )
                    LOGGER.info(
                        "Configured split output projection with %d/%d top-variance dimensions.",
                        len(split_tail_indices),
                        hidden_size,
                    )
                else:
                    LOGGER.warning(
                        "Requested split output projection with normalization_method=%s, "
                        "but this mode has no variance stats; using the normal output projection.",
                        normalization_method,
                    )
            opt_adamw, sched_adamw = create_optimizer_and_scheduler()
            if init_dir is not None and getattr(args, "load_opt", False):
                opt_path = init_dir / "opt.pt"
                if opt_path.exists():
                    opt_state = torch.load(opt_path, map_location=device)
                    opt_adamw.load_state_dict(opt_state["optimizer"])
                    if "scheduler" in opt_state:
                        sched_adamw.load_state_dict(opt_state["scheduler"])
                    LOGGER.info("Loaded optimizer state from %s.", opt_path)
                else:
                    LOGGER.warning("Requested --load-opt but no optimizer state found at %s.", opt_path)
        
        
        train_dataset = load_activation_dataset(str(tmp_dir))
        train_dataloader = get_activation_dataloader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            normalizer=glp_model.normalizer,
            shuffle=getattr(args, "shuffle", True),
        )
        
        glp_model.train()
        for batch in train_dataloader:
            if global_step >= total_steps: break
            
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}

            loss_kwargs = {
                "tail_aware_weight": getattr(args, "tail_aware_weight", 0.0),
                "tail_aware_start": getattr(args, "tail_aware_start", 1000),
                "tail_aware_min_weight": getattr(args, "tail_aware_min_weight", 0.1),
                "tail_aware_max_weight": getattr(args, "tail_aware_max_weight", 10.0),
            }

            with torch.autocast(device_type="cuda" if use_autocast else "cpu", dtype=torch.bfloat16, enabled=use_autocast):
                outputs = glp_model(
                    **batch,
                    global_step=global_step,
                    total_steps=total_steps,
                    loss_kwargs=loss_kwargs,
                )
                loss = outputs.loss
                tgt_norm = outputs.tgt_norm
                latent_pre_l2 = outputs.latent_pre_l2
                latent_post_l2 = outputs.latent_post_l2
                latent_pre_l1 = outputs.latent_pre_l1
                latent_post_l1 = outputs.latent_post_l1
                cos_sim = outputs.cos_sim
                loss_rel = outputs.loss_rel
                loss_raw = outputs.loss_raw
                
            loss.backward()
            grad_clip_threshold = float(getattr(args, "gradient_clipping_threshold", 1.0))
            max_grad_norm = grad_clip_threshold if grad_clip_threshold > 0.0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(glp_model.parameters(), max_grad_norm)
            grad_norm_value = float(grad_norm.detach().float().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            
            opt_adamw.step()
            opt_adamw.zero_grad()
            sched_adamw.step()
            
            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Streaming step {global_step}/{total_steps} (loss: {loss.item():.4f})")
            
            log_every_n_steps = max(1, int(getattr(args, "log_every_n_steps", 10)))
            if wandb_run and global_step % log_every_n_steps == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/loss_rel": loss_rel.item(),
                    "train/loss_raw": loss_raw.item(),
                    "train/grad_norm": grad_norm_value,
                    "train/cos_sim": cos_sim.item(),
                    "train/target_norm": tgt_norm.item(),
                    "train/latent_pre_l2": latent_pre_l2.item(),
                    "train/latent_post_l2": latent_post_l2.item(),
                    "train/latent_pre_l1": latent_pre_l1.item(),
                    "train/latent_post_l1": latent_post_l1.item(),
                    "train/batch_mean": outputs.batch_mean.item(),
                    "train/batch_var_max": outputs.batch_var.item(),
                    "train/global_mean": outputs.global_mean.item(),
                    "train/global_var_max": outputs.global_var.item(),
                    "train/tail_fraction": outputs.tail_fraction.item(),
                    "train/tail_weight_mean": outputs.tail_weight_mean.item(),
                    "train/tail_weight_max": outputs.tail_weight_max.item(),
                    "train/tail_base_mse": outputs.tail_base_mse.item(),
                    "train/tail_weighted_mse": outputs.tail_weighted_mse.item(),
                    "train/tail_region_mse": outputs.tail_region_mse.item(),
                    "train/non_tail_region_mse": outputs.non_tail_region_mse.item(),
                }
                log_dict["train/lr_adamw"] = sched_adamw.get_last_lr()[0]
                wandb_run.log(log_dict, step=global_step)

        total_tokens_collected += vectors_written
        if total_tokens_collected >= next_checkpoint_target:
            def format_tokens(t):
                if t >= 1_000_000 and t % 1_000_000 == 0:
                    return f"{t // 1_000_000}M"
                if t >= 1_000 and t % 1_000 == 0:
                    return f"{t // 1_000}K"
                return str(t)
            
            milestone_str = format_tokens(int(next_checkpoint_target))
            milestone_dir = Path(args.save_root) / args.run_name / milestone_str
            save_glp_checkpoint(milestone_dir)
            LOGGER.info(f"Saved checkpoint at {total_tokens_collected} tokens to {milestone_dir}")
            
            while next_checkpoint_target <= total_tokens_collected:
                next_checkpoint_target += getattr(args, "checkpoint_token_step", float('inf'))

    pbar.close()
    if wandb_run: wandb_run.finish()
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        
    out_dir = Path(args.save_root) / args.run_name
    save_glp_checkpoint(out_dir)
    
    LOGGER.info(f"Stream training complete! Model saved to {out_dir}")
