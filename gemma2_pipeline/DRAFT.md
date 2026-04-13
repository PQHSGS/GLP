# Gemma-2-2B GLP Draft Workflow

This draft keeps everything inside GLP and avoids any Steering integration.

## 1) Collect activations (FineWeb -> memmap)

Run from GLP root:

```bash
python3 -m gemma2_pipeline.cli collect \
  --model-name google/gemma-2-2b \
  --layer 12 \
  --device auto \
  --torch-dtype float32 \
  --output-dir data/gemma2-2b-layer12-fineweb-1M \
  --dataset-name HuggingFaceFW/fineweb \
  --dataset-config sample-10BT \
  --max-documents 1000 \
  --max-vectors 1000000 \
  --token-idx all \
  --drop-bos \
  --storage-dtype float32
```

Artifacts created in the output dir:

- data_0000.npy, data_0001.npy, ...
- data_indices.npy
- dtype.txt
- rep_statistics.pt
- collection_summary.json

## 2) Write Gemma training config

```bash
python3 -m gemma2_pipeline.cli write-train-config \
  --train-dataset ./data/gemma2-2b-layer12-fineweb-1M \
  --layer 12 \
  --device auto \
  --denoiser-layers 3 \
  --wandb false \
  --config-out configs/train_gemma2_2b_static.yaml
```

## 3) Train

```bash
python3 -m gemma2_pipeline.cli train \
  --config-path configs/train_gemma2_2b_static.yaml
```

## 4) Test (FD on held-out vectors)

Use a reference memmap folder that follows the same format.

```bash
python3 -m gemma2_pipeline.cli test \
  --weights-folder runs/glp-gemma2-2b-d3_static-1M \
  --reference-data-dir data/gemma2-2b-layer12-fineweb-1M \
  --device auto \
  --checkpoint final \
  --sample-size 50000
```

## Notes

- Device `auto` uses GPU when available and falls back to CPU otherwise.
- The collector defaults to token_idx=all and drops BOS to match paper-style data handling.
- For even-layer models, layer 12 is a practical middle-layer default for Gemma-2-2B.
- You can later compare layer 12 vs 13 with the same pipeline.
