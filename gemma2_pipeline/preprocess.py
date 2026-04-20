from typing import Iterable, Iterator, Sequence

import numpy as np

import torch


def is_valid_text(text: object) -> bool:
    return isinstance(text, str) and bool(text.strip())


def clean_text(text: str) -> str:
    # Collapse whitespace to reduce tokenizer variance from malformed web lines.
    return " ".join(text.split())


def batch_items(items: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    bucket: list[str] = []
    for item in items:
        bucket.append(item)
        if len(bucket) >= batch_size:
            yield bucket
            bucket = []

    if bucket:
        yield bucket


def flatten_layer_activations(activations: torch.Tensor, drop_bos: bool) -> torch.Tensor:
    """
    Convert save_acts outputs into token vectors shaped (n_vectors, d_model).

    Expected input shapes:
    - token_idx='all':  (batch, layers, seq, dim)
    - token_idx='last': (batch, layers, dim)
    """
    if activations.ndim == 4:
        if activations.shape[1] != 1:
            raise ValueError("collector expects a single layer in tracedict_config")
        vectors = activations[:, 0, :, :]
        if drop_bos and vectors.shape[1] > 0:
            vectors = vectors[:, 1:, :]
        return vectors.reshape(-1, vectors.shape[-1])

    if activations.ndim == 3:
        if activations.shape[1] != 1:
            raise ValueError("collector expects a single layer in tracedict_config")
        vectors = activations[:, 0, :]
        return vectors.reshape(-1, vectors.shape[-1])

    raise ValueError(f"Unexpected activation tensor rank: {activations.ndim}")


def sample_random_token_per_document(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    drop_bos: bool,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Select exactly one random valid token vector per document.

    Expected input:
    - activations: (batch, layers, seq, dim) from save_acts with token_idx='all'
    - attention_mask: (batch, seq) with 1 for valid tokens and 0 for padding
    """
    if activations.ndim != 4:
        raise ValueError(
            "sample_random_token_per_document expects token_idx='all' activations with shape (batch, layers, seq, dim)"
        )
    if activations.shape[1] != 1:
        raise ValueError("collector expects a single layer in tracedict_config")
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask rank 2, got {attention_mask.ndim}")
    if attention_mask.shape[0] != activations.shape[0] or attention_mask.shape[1] != activations.shape[2]:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match activations {(activations.shape[0], activations.shape[2])}"
        )

    vectors = activations[:, 0, :, :]
    attention_mask = attention_mask.to(dtype=torch.long, device=vectors.device)

    picks = []
    for i in range(vectors.shape[0]):
        valid_len = int(attention_mask[i].sum().item())
        start = 1 if drop_bos else 0
        if valid_len <= start:
            continue
        token_idx = int(rng.integers(start, valid_len))
        picks.append(vectors[i, token_idx, :])

    if not picks:
        return torch.empty((0, vectors.shape[-1]), dtype=vectors.dtype, device=vectors.device)
    return torch.stack(picks, dim=0)
