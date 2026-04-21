import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
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

def setup_glp_model(hidden_size, args):
    d_model = args.d_model_mult * hidden_size
    d_mlp = args.d_mlp_mult * hidden_size
    
    model = GLP(
        normalizer_config={"rep_statistic": "", "d_input": hidden_size}, # Initializes identity buffer of right shape
        denoiser_config={
            "d_input": hidden_size,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "n_layers": args.denoiser_layers,
            "multi_layer_n_layers": None,
        },
        tracedict_config={
            "layer_prefix": "model.layers",
            "layers": [args.layer],
            "retain": args.retain,
        }
    )
    return model

def stream_train(args):
    device = args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Gemma Extractor
    LOGGER.info(f"Loading extractor model {args.model_name}")
    hf_model, hf_tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=device,
        torch_dtype_name=getattr(args, "torch_dtype", "bfloat16"),
    )
    hidden_size = int(hf_model.config.hidden_size)
    
    # 2. Setup GLP Trainer
    LOGGER.info("Setting up GLP denoiser")
    glp_model = setup_glp_model(hidden_size, args).to(device)
    optimizer = torch.optim.AdamW(glp_model.parameters(), lr=args.learning_rate)
    
    from functools import partial
    total_steps = args.total_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(
            cosine_scheduler_with_warmup,
            warmup_steps=int(getattr(args, "warmup_ratio", 0.01) * total_steps),
            max_steps=total_steps,
            initial_factor=getattr(args, "initial_factor", 0.01),
            final_factor=getattr(args, "final_factor", 0.1),
        )
    )

    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    else:
        wandb_run = None

    tmp_dir = Path("data/tmp_stream")
    stats = RunningMoments(hidden_size)
    fineweb = FineWebSourceConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
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
        save_rep_statistics(stats, save_dir / "rep_statistics.pt")
        
        import yaml
        d_model = args.d_model_mult * hidden_size
        d_mlp = args.d_mlp_mult * hidden_size
        config_dict = {
            "model_name": args.model_name,
            "glp_kwargs": {
                "normalizer_config": {
                    "rep_statistic": "rep_statistics.pt"
                },
                "denoiser_config": {
                    "d_input": hidden_size,
                    "d_model": d_model,
                    "d_mlp": d_mlp,
                    "n_layers": args.denoiser_layers,
                    "multi_layer_n_layers": None,
                },
                "tracedict_config": {
                    "layer_prefix": "model.layers",
                    "layers": [args.layer],
                    "retain": args.retain,
                }
            }
        }
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
    
    tracedict_config = build_tracedict_config(layer=args.layer, retain=args.retain)
    use_autocast = bool(getattr(args, "use_bf16", True) and ("cuda" in str(device)))

    # Use a single streaming pass and update normalization stats cumulatively per chunk.
    batch_iterator = build_batch_iterator()
    
    while global_step < total_steps:
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

            stats.update(vectors.detach().float().cpu().numpy())

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

        mean, var = stats.finalize()
        # rep_statistics stores one scalar per hidden dimension across all seen samples.
        if mean.ndim != 1 or var.ndim != 1 or mean.shape[0] != hidden_size or var.shape[0] != hidden_size:
            raise RuntimeError(
                f"Expected 1D mean/var vectors of length {hidden_size}, got mean={mean.shape}, var={var.shape}"
            )
        glp_model.normalizer.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        glp_model.normalizer.var = torch.tensor(var, dtype=torch.float32, device=device)
        LOGGER.info(
            "Updated cumulative normalizer stats from %d vectors (latest chunk: %d).",
            stats.count,
            vectors_written,
        )
        
        
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
            with torch.autocast(device_type="cuda" if use_autocast else "cpu", dtype=torch.bfloat16, enabled=use_autocast):
                outputs = glp_model(**batch)
                loss = outputs.loss
                
            loss.backward()
            grad_clip_threshold = float(getattr(args, "gradient_clipping_threshold", 1.0))
            max_grad_norm = grad_clip_threshold if grad_clip_threshold > 0.0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(glp_model.parameters(), max_grad_norm)
            grad_norm_value = float(grad_norm.detach().float().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Streaming step {global_step}/{total_steps} (loss: {loss.item():.4f})")
            
            log_every_n_steps = max(1, int(getattr(args, "log_every_n_steps", 10)))
            if wandb_run and global_step % log_every_n_steps == 0:
                wandb_run.log(
                    {
                        "train/step": global_step,
                        "train/loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm_value,
                    },
                    step=global_step,
                )

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
