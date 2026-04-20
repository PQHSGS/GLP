"""Collect activation datasets for GLP training.

This script is a thin, Kaggle-friendly wrapper around gemma2_pipeline
activation collection utilities. Despite the module name, the dataset input
is generic HuggingFace datasets as long as a text field is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma2_pipeline.settings import ActivationCollectionConfig, FineWebSourceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect LLM activations into GLP memmap dataset format"
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--output-dir", default="data/llama1b-layer07-fineweb-localcollect-1M-all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")

    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=50000)

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--token-idx", choices=["last", "all", "random_doc"], default="all")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--drop-bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--padding-side", choices=["left", "right"], default="right")
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--vectors-per-file", type=int, default=50000)
    parser.add_argument("--max-vectors", type=int, default=1000000)
    parser.add_argument("--storage-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    return parser


def run(args: argparse.Namespace) -> None:
    from gemma2_pipeline.collect_acts import collect_activations

    source_cfg = FineWebSourceConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        streaming=True,
        max_documents=args.max_documents,
    )
    collect_cfg = ActivationCollectionConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        layer=args.layer,
        max_length=args.max_length,
        token_idx=args.token_idx,
        sample_seed=args.sample_seed,
        drop_bos=args.drop_bos,
        padding_side=args.padding_side,
        document_batch_size=args.document_batch_size,
        forward_batch_size=args.forward_batch_size,
        vectors_per_file=args.vectors_per_file,
        max_vectors=args.max_vectors,
        storage_dtype=args.storage_dtype,
        device=args.device,
        torch_dtype=args.torch_dtype,
        fineweb=source_cfg,
    )

    summary = collect_activations(collect_cfg)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
