import argparse
import json

from .settings import ActivationCollectionConfig, FineWebSourceConfig, GemmaTrainConfig


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return True

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use true/false."
    )


def _collect_subcommand(args: argparse.Namespace) -> None:
    from .collect_acts import collect_activations

    fineweb_cfg = FineWebSourceConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        streaming=args.streaming,
        max_documents=args.max_documents,
    )
    collect_cfg = ActivationCollectionConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        layer=args.layer,
        max_length=args.max_length,
        token_idx=args.token_idx,
        drop_bos=args.drop_bos,
        padding_side=args.padding_side,
        document_batch_size=args.document_batch_size,
        forward_batch_size=args.forward_batch_size,
        vectors_per_file=args.vectors_per_file,
        max_vectors=args.max_vectors,
        storage_dtype=args.storage_dtype,
        device=args.device,
        torch_dtype=args.torch_dtype,
        fineweb=fineweb_cfg,
    )
    summary = collect_activations(collect_cfg)
    print(json.dumps(summary, indent=2))


def _write_config_subcommand(args: argparse.Namespace) -> None:
    from .training import write_train_config

    train_cfg = GemmaTrainConfig(
        save_root=args.save_root,
        model_name=args.model_name,
        run_name=args.run_name,
        train_dataset=args.train_dataset,
        rep_statistic=args.rep_statistic,
        denoiser_layers=args.denoiser_layers,
        layer=args.layer,
        device=args.device,
        config_out_path=args.config_out,
        wandb_enabled=args.wandb,
    )
    config_path = write_train_config(train_cfg, args.config_out)
    print(str(config_path))


def _train_subcommand(args: argparse.Namespace) -> None:
    from .training import launch_training

    launch_training(
        args.config_path,
        python_bin=args.python_bin,
        working_dir=args.working_dir,
    )


def _test_subcommand(args: argparse.Namespace) -> None:
    from .testing import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        weights_folder=args.weights_folder,
        reference_data_dir=args.reference_data_dir,
        checkpoint=args.checkpoint,
        sample_size=args.sample_size,
        num_timesteps=args.num_timesteps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        layer_idx=args.layer_idx,
        save_path=args.save_path,
    )
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Gemma-2-2B-IT GLP pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect FineWeb activations")
    collect.add_argument("--model-name", default="google/gemma-2-2b-it")
    collect.add_argument("--output-dir", default="data/gemma2-2b-layer14-fineweb-1M")
    collect.add_argument("--layer", type=int, default=12)
    collect.add_argument("--max-length", type=int, default=2048)
    collect.add_argument("--token-idx", choices=["last", "all"], default="all")
    collect.add_argument("--drop-bos", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--padding-side", choices=["left", "right"], default="right")
    collect.add_argument("--document-batch-size", type=int, default=128)
    collect.add_argument("--forward-batch-size", type=int, default=8)
    collect.add_argument("--vectors-per-file", type=int, default=50000)
    collect.add_argument("--max-vectors", type=int, default=1000000)
    collect.add_argument("--storage-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    collect.add_argument("--device", default="auto")
    collect.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    collect.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    collect.add_argument("--dataset-config", default="sample-10BT")
    collect.add_argument("--split", default="train")
    collect.add_argument("--text-field", default="text")
    collect.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--max-documents", type=int, default=1000)
    collect.set_defaults(func=_collect_subcommand)

    write_cfg = subparsers.add_parser("write-train-config", help="Write Gemma GLP YAML config")
    write_cfg.add_argument("--save-root", default=".")
    write_cfg.add_argument("--model-name", default="google/gemma-2-2b-it")
    write_cfg.add_argument("--run-name", default="glp-gemma2-2b-d3_static-1M")
    write_cfg.add_argument("--train-dataset", default="./data/gemma2-2b-layer14-fineweb-1M")
    write_cfg.add_argument("--rep-statistic", default=None)
    write_cfg.add_argument("--denoiser-layers", type=int, default=3)
    write_cfg.add_argument("--layer", type=int, default=12)
    write_cfg.add_argument("--device", default="auto")
    write_cfg.add_argument("--config-out", default="configs/train_gemma2_2b_static.yaml")
    write_cfg.add_argument(
        "--wandb",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Enable or disable wandb (e.g. --wandb false)",
    )
    write_cfg.add_argument("--no-wandb", dest="wandb", action="store_false")
    write_cfg.set_defaults(func=_write_config_subcommand)

    train = subparsers.add_parser("train", help="Launch GLP training with a config")
    train.add_argument("--config-path", required=True)
    train.add_argument("--python-bin", default="python3")
    train.add_argument("--working-dir", default=None)
    train.set_defaults(func=_train_subcommand)

    test = subparsers.add_parser("test", help="Evaluate a GLP checkpoint with FD")
    test.add_argument("--weights-folder", required=True)
    test.add_argument("--reference-data-dir", required=True)
    test.add_argument("--checkpoint", default="final")
    test.add_argument("--sample-size", type=int, default=50000)
    test.add_argument("--num-timesteps", type=int, default=1000)
    test.add_argument("--batch-size", type=int, default=256)
    test.add_argument("--device", default="auto")
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--layer-idx", type=int, default=None)
    test.add_argument("--save-path", default=None)
    test.set_defaults(func=_test_subcommand)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
