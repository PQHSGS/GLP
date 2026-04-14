# Quickstart: GLP Streaming Mode Training

This guide provides the exact commands needed to set up the environment and launch the memory-efficient streaming pipeline for training Generative Latent Priors (GLPs). This mode generates activations on-the-fly and trains dynamically, keeping disk usage strictly near 0%.

## 1. Setup Environment
First, clone the repository and set up the Python environment using Conda.
```bash
git clone <repository_url>
cd GLP

conda env create -f environment.yaml
conda activate glp

pip install vllm==0.9.2 
pip install transformers==4.47.0
pip install -e .
```

## 2. Authentication
You need to authenticate to Hugging Face to download the language models (like Gemma) and Weights & Biases to keep track of the training loss curve.
```bash
# Login to Hugging Face (will prompt for a read token from huggingface.co/settings/tokens)
hf auth login

# Login to Weights & Biases
wandb login 
```

## 3. Run Training
Copy and paste the command below into your terminal to start the streaming pipeline. It will handle loading the Gemma model, iteratively reading the FineWeb dataset without hoarding space, and optimizing the diffusion GLP network natively.

```bash
python3 scripts/glp_cli.py stream \
  --device cuda \
  --stream-chunk-size 50000 \
  --total-steps 250000 \
  --batch-size 4096 \
  --dataset-config sample-10BT \
  --model-name google/gemma-2-2b-it \
  --learning-rate 5e-5 \
  --layer 14 \
  --run-name glp-streaming-run \
  --wandb
```

---

## Hyperparameter Guide (Ranked by Importance)

Below are the parameters you might want to adjust, ranked from the most commonly changed (to avoid memory/GPU issues) to the least.

### 1. `--batch-size` (Default: 4096)
**What it does:** The number of activation vectors loaded simultaneously per forward/backward pass of the GLP training step.
**How to adjust:** 
- If you encounter a **CUDA Out of Memory (OOM)** error, drastically reduce this (e.g., to `2048` or `1024`). The GLP model is small enough that `4096` easily fits into an NVIDIA A30 (24GB) alongside Gemma-2B, but lowering it is your primary defense against OOM crashes on smaller cards (like RTX 3090/4090s).

### 2. `--total-steps` (Default: 10000)
**What it does:** The absolute number of gradient steps the model will loop through before halting the training process completely.
**How to adjust:** 
- Scale this up to train longer for better convergence. 
- *Time Estimation:* Multiply your GPU's speed (e.g. `1.5s/step` shown in the `tqdm` progress bar) by `--total-steps` to calculate exactly how long the execution will take.

### 3. `--stream-chunk-size` (Default: 50000)
**What it does:** Regulates the buffer interval. This is precisely how many raw activations the LLM sequentially generates and flushes natively *before* switching over and letting the GLP model train against that extracted slice.
**How to adjust:** 
- Lowering it (e.g. to `10000`) causes the script to alternate between generating and training much more aggressively. 
- Leaving it between `50,000` to `100,000` is highly recommended, as it minimizes the context-switching latency so the GPU operates optimally.

### 4. `--model-name` & `--layer` (Default: google/gemma-2-2b-it @ layer 14)
**What it does:** The target LLM text model and the specific neural network layer where the script intercepts the semantic activations.
**How to adjust:** 
- Target any model on HuggingFace by swapping `--model-name`. Change `--layer` depending on if you want to model "early" syntactical activations or "late" highly conceptual activations. Layer 14 is strictly the mathematical center of Gemma-2-2B.

### 5. `--learning-rate` (Default: 5e-5)
**What it does:** The peak parameter tuning rate for the AdamW optimizer following a quick cosine warmup.
**How to adjust:** 
- Only alter this if you open WandB and observe the `train/loss` curve flatlining too early or suffering massive geometric spikes. `5e-5` achieves beautiful downward loss 99% of the time seamlessly.

### 6. `--denoiser-layers` (Default: 3)
**What it does:** The internal architectural depth of the GLP diffusion network.
**How to adjust:** 
- Bumping to `6` or `12` scales the latent capacity of the GLP at the expense of needing heavier GPU load. The original paper baseline defaults to `3`.
