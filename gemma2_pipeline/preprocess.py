from typing import Iterable, Iterator, Sequence

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
