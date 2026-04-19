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
    max_documents: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActivationCollectionConfig:
    """Configuration for collecting Gemma activations into memmaps."""

    model_name: str = "google/gemma-2-2b-it"
    output_dir: str = "data/gemma2-2b-layer14-fineweb-1M"
    layer: int = 14
    max_length: int = 2048
    token_idx: Literal["last", "all"] = "all"
    drop_bos: bool = True
    padding_side: Literal["left", "right"] = "right"
    document_batch_size: int = 128
    forward_batch_size: int = 8
    vectors_per_file: int = 50000
    max_vectors: Optional[int] = 1000000
    storage_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: str = "auto"
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    fineweb: FineWebSourceConfig = field(default_factory=FineWebSourceConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fineweb"] = self.fineweb.to_dict()
        return data


@dataclass
class GemmaTrainConfig:
    """Configuration for writing and launching GLP training."""

    save_root: str = "."
    model_name: str = "google/gemma-2-2b-it"
    run_name: str = "glp-gemma2-2b-d3_static-1M"
    train_dataset: str = "./data/gemma2-2b-layer14-fineweb-1M"
    rep_statistic: Optional[str] = None
    num_epochs: int = 1
    save_epochs: list[int] = field(default_factory=lambda: [1])
    shuffle: bool = True
    d_input: int = 2304
    d_model: int = 4608
    d_mlp: int = 9216
    denoiser_layers: int = 3
    multi_layer_n_layers: Optional[int] = None
    layer: int = 14
    retain: str = "output"
    device: str = "auto"
    use_bf16: bool = False
    learning_rate: float = 5e-5
    batch_size: int = 4096
    gradient_accumulation_steps: int = 1
    gradient_clipping_threshold: float = 1.0
    log_every_n_steps: int = 10
    save_opt_state: bool = True
    warmup_ratio: float = 0.01
    initial_factor: float = 0.01
    final_factor: float = 0.1
    wandb_enabled: bool = True
    wandb_project: str = "glp"
    config_out_path: str = "configs/train_gemma2_2b_static.yaml"

    def to_dict(self) -> dict:
        return asdict(self)
