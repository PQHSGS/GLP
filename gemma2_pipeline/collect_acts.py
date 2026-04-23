import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from glp.utils_acts import MemmapWriter, save_acts

from .loading import (
    get_storage_dtype,
    iter_fineweb_texts,
    load_model_and_tokenizer,
    to_storage_array,
)
from .preprocess import batch_items, flatten_layer_activations, sample_random_token_per_document
from .settings import ActivationCollectionConfig
from .stats import RunningMoments, save_rep_statistics

LOGGER = logging.getLogger(__name__)


def _init_writer(config: ActivationCollectionConfig, hidden_size: int) -> tuple[MemmapWriter, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np_dtype, dtype_label = get_storage_dtype(config.storage_dtype)
    file_size = config.vectors_per_file * hidden_size
    writer = MemmapWriter(output_dir=output_dir, file_size=file_size, dtype=np_dtype)
    (output_dir / "dtype.txt").write_text(dtype_label)
    return writer, output_dir


def build_tracedict_config(layer: int, retain: str = "output", layer_prefix: str = "model.layers") -> dict:
    return {
        "layer_prefix": layer_prefix,
        "layers": [layer],
        "retain": retain,
    }


def extract_activation_vectors(
    *,
    hf_model,
    hf_tokenizer,
    text_batch: list[str],
    tracedict_config: dict,
    padding_side: str,
    token_idx: str,
    forward_batch_size: int,
    max_length: int,
    drop_bos: bool,
    rng: np.random.Generator,
) -> torch.Tensor:
    save_token_idx = "all" if token_idx == "random_doc" else token_idx
    activations = save_acts(
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        text=text_batch,
        tracedict_config=tracedict_config,
        padding_side=padding_side,
        token_idx=save_token_idx,
        batch_size=forward_batch_size,
        max_length=max_length,
    )

    if token_idx == "random_doc":
        tokenized = hf_tokenizer(
            text_batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        return sample_random_token_per_document(
            activations,
            tokenized["attention_mask"],
            drop_bos=drop_bos,
            rng=rng,
        )
    return flatten_layer_activations(activations, drop_bos=drop_bos)


def write_vectors_to_memmap(
    writer: MemmapWriter,
    vectors: torch.Tensor,
    storage_dtype: str,
    *,
    max_rows: int | None = None,
) -> int:
    if max_rows is not None:
        if max_rows <= 0:
            return 0
        vectors = vectors[:max_rows]

    if vectors.numel() == 0 or vectors.shape[0] == 0:
        return 0

    storage_vectors = to_storage_array(vectors, storage_dtype)
    for row in storage_vectors:
        writer.write(np.ascontiguousarray(row))
    return int(storage_vectors.shape[0])


def collect_activations(config: ActivationCollectionConfig) -> dict:
    """Extract Gemma activations from FineWeb and store in GLP memmap format."""
    hf_model, hf_tokenizer = load_model_and_tokenizer(
        config.model_name,
        device=config.device,
        torch_dtype_name=config.torch_dtype,
    )

    hidden_size = int(hf_model.config.hidden_size)
    writer, output_dir = _init_writer(config, hidden_size)
    stats = RunningMoments(hidden_size)

    tracedict_config = build_tracedict_config(layer=config.layer, retain="output", layer_prefix=config.layer_prefix)

    documents_processed = 0
    vectors_written = 0
    rng = np.random.default_rng(config.sample_seed)

    text_iter = iter_fineweb_texts(config.fineweb)
    for text_batch in tqdm(
        batch_items(text_iter, config.document_batch_size),
        desc="Collecting activations",
        dynamic_ncols=True,
    ):
        documents_processed += len(text_batch)

        vectors = extract_activation_vectors(
            hf_model=hf_model,
            hf_tokenizer=hf_tokenizer,
            text_batch=text_batch,
            tracedict_config=tracedict_config,
            padding_side=config.padding_side,
            token_idx=config.token_idx,
            forward_batch_size=config.forward_batch_size,
            max_length=config.max_length,
            drop_bos=config.drop_bos,
            rng=rng,
        )
        if vectors.numel() == 0:
            continue

        if config.max_vectors is not None:
            remaining = config.max_vectors - vectors_written
            if remaining <= 0:
                break
            vectors = vectors[:remaining]

        if vectors.shape[0] == 0:
            break

        stats.update(vectors.detach().float().cpu().numpy())
        vectors_written += write_vectors_to_memmap(
            writer,
            vectors,
            config.storage_dtype,
        )
        if config.max_vectors is not None and vectors_written >= config.max_vectors:
            break

    writer.flush()

    rep_statistics_path = output_dir / "rep_statistics.pt"
    save_rep_statistics(stats, rep_statistics_path)

    summary = {
        "model_name": config.model_name,
        "layer": config.layer,
        "hidden_size": hidden_size,
        "documents_processed": documents_processed,
        "vectors_written": vectors_written,
        "output_dir": str(output_dir),
        "rep_statistics": str(rep_statistics_path),
    }
    with (output_dir / "collection_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Activation collection complete: %s", summary)
    return summary
