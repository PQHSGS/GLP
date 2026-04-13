from typing import Iterator, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .preprocess import clean_text, is_valid_text
from .settings import FineWebSourceConfig


def resolve_device(device: str) -> str:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"

    return device


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(
    model_name: str,
    *,
    device: str,
    torch_dtype_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    resolved_device = resolve_device(device)
    torch_dtype = parse_torch_dtype(torch_dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(resolved_device)
    model.eval()
    return model, tokenizer


def iter_fineweb_texts(cfg: FineWebSourceConfig) -> Iterator[str]:
    if cfg.dataset_config:
        dataset = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            split=cfg.split,
            streaming=cfg.streaming,
        )
    else:
        dataset = load_dataset(
            cfg.dataset_name,
            split=cfg.split,
            streaming=cfg.streaming,
        )

    seen = 0
    for row in dataset:
        text = row.get(cfg.text_field)
        if not is_valid_text(text):
            continue

        yield clean_text(text)
        seen += 1
        if cfg.max_documents is not None and seen >= cfg.max_documents:
            break


def get_storage_dtype(storage_dtype: str) -> tuple[np.dtype, str]:
    if storage_dtype == "float32":
        return np.dtype(np.float32), "float32"
    if storage_dtype == "float16":
        return np.dtype(np.float16), "float16"
    if storage_dtype == "bfloat16":
        # Stored as int16 and viewed back as bfloat16 when loaded.
        return np.dtype(np.int16), "int16"
    raise ValueError(f"Unsupported storage dtype: {storage_dtype}")


def to_storage_array(vectors: torch.Tensor, storage_dtype: str) -> np.ndarray:
    if storage_dtype == "bfloat16":
        return vectors.to(torch.bfloat16).view(torch.int16).cpu().numpy()

    if storage_dtype == "float16":
        return vectors.to(torch.float16).cpu().numpy()

    if storage_dtype == "float32":
        return vectors.to(torch.float32).cpu().numpy()

    raise ValueError(f"Unsupported storage dtype: {storage_dtype}")
