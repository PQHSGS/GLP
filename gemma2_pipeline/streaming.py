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
from glp.utils_acts import MemmapWriter, save_acts
from glp_train import cosine_scheduler_with_warmup, load_activation_dataset, get_activation_dataloader

from .loading import (
    get_storage_dtype,
    iter_fineweb_texts,
    load_model_and_tokenizer,
    to_storage_array,
)
from .preprocess import batch_items, flatten_layer_activations
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
            "retain": "output",
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
            warmup_steps=int(0.01 * total_steps),
            max_steps=total_steps,
            initial_factor=0.01,
            final_factor=0.1,
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
    text_iter = iter_fineweb_texts(fineweb)
    batch_iterator = batch_items(text_iter, 16)
    
    global_step = 0
    pbar = tqdm(total=total_steps, desc="Streaming GLP")
    
    tracedict_config = {
        "layer_prefix": "model.layers",
        "layers": [args.layer],
        "retain": "output",
    }
    
    use_autocast = ("cuda" in str(device))
    
    while global_step < total_steps:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        file_size = args.stream_chunk_size * hidden_size
        np_dtype, dtype_label = get_storage_dtype("float32")
        writer = MemmapWriter(output_dir=tmp_dir, file_size=file_size, dtype=np_dtype)
        (tmp_dir / "dtype.txt").write_text(dtype_label)
        
        vectors_written = 0
        while vectors_written < args.stream_chunk_size:
            text_batch = next(batch_iterator, None)
            if not text_batch:
                LOGGER.warning("Ran out of text batch data.")
                break
            
            activations = save_acts(
                hf_model=hf_model,
                hf_tokenizer=hf_tokenizer,
                text=text_batch,
                tracedict_config=tracedict_config,
                padding_side="right",
                token_idx="all",
                batch_size=1,
                max_length=args.max_length,
            )
            vectors = flatten_layer_activations(activations, drop_bos=True)
            if vectors.numel() == 0: continue
            
            remaining = args.stream_chunk_size - vectors_written
            vectors = vectors[:remaining]
            
            storage_vectors = to_storage_array(vectors, "float32")
            stats.update(storage_vectors)
            
            for row in storage_vectors:
                writer.write(np.ascontiguousarray(row))
            
            vectors_written += int(storage_vectors.shape[0])
            
        writer.flush()
        if vectors_written == 0:
            LOGGER.warning("No vectors generated in this chunk (dataset exhausted?). Halting training loop early.")
            break
        
        
        # Inject dynamic stats
        if stats.count > 0:
            mean, var = stats.finalize()
            glp_model.normalizer.mean = torch.tensor(mean, dtype=torch.float32, device=device)
            glp_model.normalizer.var = torch.tensor(var, dtype=torch.float32, device=device)
        
        train_dataset = load_activation_dataset(str(tmp_dir))
        train_dataloader = get_activation_dataloader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            normalizer=glp_model.normalizer,
            shuffle=True,
        )
        
        glp_model.train()
        for batch in train_dataloader:
            if global_step >= total_steps: break
            
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if use_autocast else "cpu", dtype=torch.bfloat16, enabled=use_autocast):
                outputs = glp_model(**batch)
                loss = outputs.loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(glp_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Streaming step {global_step}/{total_steps} (loss: {loss.item():.4f})")
            
            if wandb_run and global_step % 10 == 0:
                wandb_run.log({"train/step": global_step, "train/loss": loss.item(), "train/learning_rate": scheduler.get_last_lr()[0]}, step=global_step)

    pbar.close()
    if wandb_run: wandb_run.finish()
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        
    out_dir = Path(args.save_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    glp_model.save_pretrained(path=out_dir, name="final")
    save_rep_statistics(stats, out_dir / "rep_statistics.pt")
    
    LOGGER.info(f"Stream training complete! Model saved to {out_dir}")
