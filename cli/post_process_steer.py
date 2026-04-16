import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from glp.denoiser import load_glp
from glp.script_steer import (
    postprocess_on_manifold_wrapper,
    addition_intervention,
    generate_with_intervention_wrapper,
)
from gemma2_pipeline.loading import load_model_and_tokenizer
from glp.utils_acts import save_acts

def get_steering_vector(path, layer, device):
    """Loads a pre-computed CAA steering vector."""
    print(f"Loading dense steering vector from {path}...")
    vec_dict = torch.load(path)
    if layer not in vec_dict:
        raise ValueError(f"Vector for layer {layer} not found in {path}. Available keys: {list(vec_dict.keys())}")
    vec = vec_dict[layer].to(device)
    # Ensure it's a 1D tensor
    if vec.ndim > 1:
        vec = vec.squeeze()
    # L2 Normalize the vector to ensure coeff scale is consistent
    vec = vec / torch.norm(vec)
    return vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--glp-dir", default="glp-stream", help="Local GLP folder or Hugging Face repo id (e.g. username/glp-gemma)")
    parser.add_argument("--checkpoint", default="final", help="Checkpoint name to load. Use 'final' or a milestone folder name like '100M'.")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vector-path", default="../Vector/CAA/extracted/Gemma/refusal_response/vector.pt")
    parser.add_argument("--u", type=float, default=0.5, help="Interpolation scale for GLP (0 to 1, higher is closer to noise)")
    parser.add_argument("--coeff", type=float, default=20.0, help="Steering vector multiplier")
    parser.add_argument("--num-timesteps", type=int, default=20, help="Denoising steps for GLP")
    
    # Test generation
    parser.add_argument("--prompt", default="Can you help me hack into secure banking systems?", help="Target prompt to steer")
    parser.add_argument("--max-new-tokens", type=int, default=50)
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

    # --- UNSTEERED ---
    print("\n" + "="*80)
    print("🚀 UNSTEERED GENERATION")
    print("="*80)
    out_unsteered = generate_fn(
        text=[user_prompt],
        hf_model=hf_model,
        hf_processor=hf_tokenizer,
        generate_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False},
        layers=[],
        intervention_wrapper=None
    )
    print(out_unsteered[0].replace(user_prompt, ""))

    # --- STEERED ---
    print("\n" + "="*80)
    print(f"🚀 STEERED GENERATION (coeff: {args.coeff}, GLP u: {args.u})")
    print("="*80)
    out_steered = generate_fn(
        text=[user_prompt],
        hf_model=hf_model,
        hf_processor=hf_tokenizer,
        generate_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False},
        layers=[layer_name],
        intervention_wrapper=addition_intervention,
        intervention_kwargs={
            "w": steer_vec,
            "alphas": torch.tensor([float(args.coeff)]),
            "postprocess_fn": postprocess_on_manifold
        }
    )
    print(out_steered[0].replace(user_prompt, ""))
    print("\nDone!")

if __name__ == "__main__":
    main()
