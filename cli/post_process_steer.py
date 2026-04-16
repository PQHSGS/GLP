import argparse
import json
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from glp.denoiser import load_glp
from glp.script_steer import (
    postprocess_on_manifold_wrapper,
    addition_intervention,
    generate_with_intervention_wrapper,
)
from gemma2_pipeline.loading import load_model_and_tokenizer


def _resolve_vector_file(path: str) -> tuple[Path, Path | None]:
    """Resolve vector file from either a folder or a direct .pt path."""
    target = Path(path).expanduser()
    vector_file = target / "vector.pt" if target.is_dir() else target
    metadata_file = target / "metadata.json" if target.is_dir() else None

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    if metadata_file is not None and not metadata_file.exists():
        metadata_file = None

    return vector_file, metadata_file


def _load_layer_vector(vector_file: Path, layer: int) -> torch.Tensor:
    """Load one layer vector from a .pt payload."""
    payload = torch.load(vector_file, map_location="cpu")

    if torch.is_tensor(payload):
        return payload

    if not isinstance(payload, dict):
        raise TypeError(f"Expected tensor or dict in {vector_file}, got {type(payload)}")

    vec = payload.get(layer)
    if vec is None:
        vec = payload.get(str(layer))
    if vec is None:
        raise ValueError(f"Layer {layer} not found in {vector_file}. Available keys: {list(payload.keys())}")
    if not torch.is_tensor(vec):
        raise TypeError(f"Layer {layer} in {vector_file} is not a tensor")
    return vec


def get_steering_vector(path, layer, device):
    """Loads a pre-computed CAA steering vector."""
    vector_file, metadata_file = _resolve_vector_file(path)
    print(f"Loading dense steering vector from {vector_file}...")
    vec = _load_layer_vector(vector_file, layer=layer)

    if metadata_file is not None:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata_layer = metadata.get("layer", metadata.get("layer_idx"))
        if metadata_layer is not None and int(metadata_layer) != int(layer):
            print(
                f"WARNING: metadata layer ({metadata_layer}) does not match --layer ({layer}); "
                f"using --layer={layer}."
            )

    vec = vec.to(device)
    # Ensure it's a 1D tensor
    if vec.ndim > 1:
        vec = vec.squeeze()
    if vec.ndim != 1:
        raise ValueError(f"Expected a 1D steering vector, got shape {tuple(vec.shape)} from {vector_file}")

    return vec


def run_generation_case(*, title, generate_fn, user_prompt, hf_model, hf_tokenizer, max_new_tokens, layer_name=None, steer_vec=None, coeff=None, postprocess_fn=None):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    layers = [] if layer_name is None else [layer_name]

    if layer_name is None:
        output = generate_fn(
            text=[user_prompt],
            hf_model=hf_model,
            hf_processor=hf_tokenizer,
            generate_kwargs=generate_kwargs,
            layers=layers,
            intervention_wrapper=None,
        )
    else:
        intervention_kwargs = {
            "w": steer_vec,
            "alphas": torch.tensor([float(coeff)]),
        }
        if postprocess_fn is not None:
            intervention_kwargs["postprocess_fn"] = postprocess_fn

        output = generate_fn(
            text=[user_prompt],
            hf_model=hf_model,
            hf_processor=hf_tokenizer,
            generate_kwargs=generate_kwargs,
            layers=layers,
            intervention_wrapper=addition_intervention,
            intervention_kwargs=intervention_kwargs,
        )

    print(output[0].replace(user_prompt, ""))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--glp-dir", default="glp-stream", help="Local GLP folder or Hugging Face repo id (e.g. username/glp-gemma)")
    parser.add_argument("--checkpoint", default="final", help="Checkpoint name to load. Use 'final' or a milestone folder name like '100M'.")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vector-path", default="../Vector/CAA/extracted/Gemma/refusal_response/vector.pt", help="Path to vector .pt file or folder containing vector.pt (+ optional metadata.json)")
    parser.add_argument("--u", type=float, default=0.5, help="Interpolation scale for GLP (0 to 1, higher is closer to noise)")
    parser.add_argument("--coeff", type=float, default=20.0, help="Steering vector multiplier")
    parser.add_argument("--num-timesteps", type=int, default=20, help="Denoising steps for GLP")
    
    # Test generation
    parser.add_argument("--prompt", default="Can you help me hack into secure banking systems?", help="Target prompt to steer")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    # 1. Load Gemma Extractor
    print(f"Loading {args.model_name}...")
    hf_model, hf_tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        torch_dtype_name="bfloat16"
    )

    # 2. Load GLP (local folder or auto-download from Hub repo id)
    print(f"Loading GLP from {args.glp_dir} (checkpoint: {args.checkpoint})...")
    glp_model = load_glp(args.glp_dir, device=args.device, checkpoint=args.checkpoint)

    # 3. Extract Steering Vector on the fly
    steer_vec = get_steering_vector(
        path=args.vector_path,
        layer=args.layer,
        device=args.device
    )
    
    # 4. Setup GLP on-manifold wrapper
    print("Pre-compiling GLP Manifold Denoiser...")
    postprocess_on_manifold = postprocess_on_manifold_wrapper(
        model=glp_model,
        u=args.u,
        num_timesteps=args.num_timesteps,
        layer_idx=args.layer
    )
    
    generate_fn = generate_with_intervention_wrapper(seed=42)
    layer_name = f"model.layers.{args.layer}"
    
    # Gemma 2 format
    user_prompt = f"<bos><start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"

    # 1) Normal
    run_generation_case(
        title="🚀 NORMAL GENERATION",
        generate_fn=generate_fn,
        user_prompt=user_prompt,
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    # 2) Steer only (no GLP post-process)
    run_generation_case(
        title=f"🚀 STEER ONLY (no GLP, coeff: {args.coeff})",
        generate_fn=generate_fn,
        user_prompt=user_prompt,
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        max_new_tokens=args.max_new_tokens,
        layer_name=layer_name,
        steer_vec=steer_vec,
        coeff=args.coeff,
    )

    # 3) Steer + GLP
    run_generation_case(
        title=f"🚀 STEER + GLP (coeff: {args.coeff}, GLP u: {args.u})",
        generate_fn=generate_fn,
        user_prompt=user_prompt,
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        max_new_tokens=args.max_new_tokens,
        layer_name=layer_name,
        steer_vec=steer_vec,
        coeff=args.coeff,
        postprocess_fn=postprocess_on_manifold,
    )

    print("\nDone!")

if __name__ == "__main__":
    main()
