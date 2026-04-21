# Quickstart: GLP Streaming Training

This guide focuses on the current Llama-1B streaming setup in this repo. The streaming pipeline collects activations on the fly and trains GLP without storing the full activation corpus on disk.

## 1. Environment Setup

```bash
git clone <repository_url>
cd GLP

conda env create -f environment.yaml
conda activate glp

pip install -e .
```

## 2. Authentication

```bash
# Hugging Face (for gated model access)
hf auth login

# Weights & Biases
wandb login
```

## 3. Streaming Defaults (Current)

The stream CLI now defaults to the validated Llama-1B collection/training profile:

- model: meta-llama/Llama-3.2-1B
- layer: 7
- retain: output
- max_length: 2048
- token_idx: all
- drop_bos: true
- padding_side: right
- document_batch_size: 16
- forward_batch_size: 1
- storage_dtype: bfloat16
- max_documents: 50000
- batch_size: 4096
- learning_rate: 5e-5
- gradient_clipping_threshold: 1.0
- log_every_n_steps: 10
- warmup_ratio: 0.01
- initial_factor: 0.01
- final_factor: 0.1

Normalization behavior in stream mode:

- Mean/variance are updated cumulatively after each streamed chunk.
- With stream_chunk_size 1000000, each update is based on a large 1M-vector window, then accumulated over the run.

WandB logs now include:

- train/loss
- train/learning_rate
- train/grad_norm

## 4. Recommended Runs

### 4.1 Sanity Run (~1M activations)

```bash
python cli/glp_cli.py stream \
  --run-name 1m \
  --stream-chunk-size 1000000 \
  --total-steps 244 \
  --checkpoint-token-step 1000000 \
  --wandb
```

### 4.2 Large Run (~1B activations)

Use the maintained script:

```bash
bash scripts/run_1b.sh
```

Equivalent key settings used by that script:

- total_steps: 244141
- batch_size: 4096
- stream_chunk_size: 1000000
- max_documents: 1000000
- checkpoint_token_step: 100000000

## 5. Push Checkpoints to Hugging Face

```bash
# Push a milestone checkpoint (example: 100M)
python cli/push_to_hf.py \
  --repo-id <your_user>/glp-llama \
  --folder ./1b/100M
```

```bash
# Push final run
python cli/push_to_hf.py \
  --repo-id <your_user>/glp-llama \
  --folder ./1b
```

## 6. Notes for Stable Reproducibility

- Keep token mode at all and drop_bos enabled for consistency with your validated localcollect setup.
- Keep layer fixed at 7 for Llama-1B to match the accepted training profile.
- For static-like 1M comparison, set stream_chunk_size to 1000000 so shuffle happens across the full 1M window instead of per 65,536 chunk.
- If GPU memory is tight, reduce batch_size first.
- If you change batch_size, recalculate total_steps for your target activation budget.
