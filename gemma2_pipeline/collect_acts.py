import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from glp.utils_acts import MemmapWriter, save_acts

from .loading import (
    get_storage_dtype,
    iter_fineweb_texts,
    load_model_and_tokenizer,
    to_storage_array,
)
from .preprocess import batch_items, flatten_layer_activations
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

    tracedict_config = {
        "layer_prefix": "model.layers",
        "layers": [config.layer],
        "retain": "output",
    }

    documents_processed = 0
    vectors_written = 0

    text_iter = iter_fineweb_texts(config.fineweb)
    for text_batch in tqdm(
        batch_items(text_iter, config.document_batch_size),
        desc="Collecting activations",
        dynamic_ncols=True,
    ):
        documents_processed += len(text_batch)

        activations = save_acts(
            hf_model=hf_model,
            hf_tokenizer=hf_tokenizer,
            text=text_batch,
            tracedict_config=tracedict_config,
            padding_side=config.padding_side,
            token_idx=config.token_idx,
            batch_size=config.forward_batch_size,
            max_length=config.max_length,
        )

        vectors = flatten_layer_activations(activations, drop_bos=config.drop_bos)
        if vectors.numel() == 0:
            continue

        if config.max_vectors is not None:
            remaining = config.max_vectors - vectors_written
            if remaining <= 0:
                break
            vectors = vectors[:remaining]

        if vectors.shape[0] == 0:
            break

        stats.update(vectors.detach().cpu().numpy())
        storage_vectors = to_storage_array(vectors, config.storage_dtype)
        for row in storage_vectors:
            writer.write(np.ascontiguousarray(row))

        vectors_written += int(storage_vectors.shape[0])
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
