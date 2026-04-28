from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class FineWebSourceConfig:
    """Data source settings for activation extraction."""

    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: Optional[str] = "sample-10BT"
    split: str = "train"
    text_field: str = "text"
    streaming: bool = True
    max_documents: Optional[int] = 50000

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActivationCollectionConfig:
    """Configuration for collecting Gemma activations into memmaps."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "data/llama1b-layer07-fineweb-localcollect-1M-all"
    layer: int = 7
    layer_prefix: str = "model.layers"
    max_length: int = 2048
    token_idx: Literal["last", "all", "random_doc"] = "all"
    sample_seed: int = 0
    drop_bos: bool = True
    padding_side: Literal["left", "right"] = "right"
    document_batch_size: int = 16
    forward_batch_size: int = 1
    vectors_per_file: int = 50000
    max_vectors: Optional[int] = 1000000
    storage_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    device: str = "cuda:0"
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    fineweb: FineWebSourceConfig = field(default_factory=FineWebSourceConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fineweb"] = self.fineweb.to_dict()
        return data


@dataclass
class ModelTrainConfig:
    """Configuration for writing and launching GLP training."""

    save_root: str = "."
    model_name: str = "meta-llama/Llama-3.2-1B"
    run_name: str = "glp-llama1b-d3_static-1B"
    train_dataset: str = "./data/llama1b-layer07-fineweb-localcollect-1M-all"
    rep_statistic: Optional[str] = None
    num_epochs: int = 1
    save_epochs: list[int] = field(default_factory=lambda: [1])
    shuffle: bool = True
    d_input: int = 2048
    d_model: int = 4096
    d_mlp: int = 8192
    denoiser_layers: int = 3
    use_spectral_norm: bool = False
    multi_layer_n_layers: Optional[int] = None
    layer: int = 7
    layer_prefix: str = "model.layers"
    retain: str = "output"
    device: str = "auto"
    use_bf16: bool = True
    learning_rate: float = 5e-5
    batch_size: int = 4096
    gradient_accumulation_steps: int = 1
    gradient_clipping_threshold: float = 1.0
    log_every_n_steps: int = 10
    save_opt_state: bool = True
    normalization_method: str = "gaussian"
    sampling_method: str = "uniform"
    ot_chunk_size: int = 256
    tail_aware_weight: float = 0.0
    tail_aware_start: int = 1000
    tail_aware_min_weight: float = 0.1
    tail_aware_max_weight: float = 10.0
    warmup_ratio: float = 0.01
    initial_factor: float = 0.01
    final_factor: float = 0.1
    wandb_enabled: bool = False
    wandb_project: str = "glp"
    config_out_path: str = "configs/train_llama1b_our.json"

    def to_dict(self) -> dict:
        return asdict(self)


def make_default_fineweb_source_config() -> FineWebSourceConfig:
    """Return a fresh default source config for CLI default wiring."""
    return FineWebSourceConfig()


def make_default_activation_collection_config() -> ActivationCollectionConfig:
    """Return a fresh default activation-collection config for CLI defaults."""
    return ActivationCollectionConfig()


def make_default_model_train_config() -> ModelTrainConfig:
    """Return a fresh default train config for CLI defaults."""
    return ModelTrainConfig()
