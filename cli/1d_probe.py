import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import pandas as pd
from datasets import load_dataset
from gemma2_pipeline.loading import load_model_and_tokenizer
from glp.utils_acts import save_acts

def prepare_dataset(dataset_name, split, n_samples):
    print(f"Loading {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=f"{split}")
    # Convert to pandas, keep text and label
    df = ds.to_pandas()
    # Ensure standard names
    if "label" in df.columns:
        df = df.rename(columns={"label": "target"})
    
    # Stratified shuffle (to keep targets balanced)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(n_samples)
    return df

def extract_and_save(hf_model, hf_tokenizer, df, out_dir, split_name, layer):
    print(f"Extracting activations for {split_name} ({len(df)} samples)...")
    texts = df['text'].tolist()
    
    tracedict_config = {
        "layer_prefix": "model.layers",
        "layers": [layer],
        "retain": "output",
    }
    
    # We do a small batch size for safety with varying sequence lengths
    acts = save_acts(
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        text=texts,
        tracedict_config=tracedict_config,
        token_idx="last",
        batch_size=8,
        max_length=512
    )
    
    # Acts shape: (N, 1, d_model) because token_idx='last'
    print(f"Saving {split_name} acts shape: {acts.shape}")
    
    os.makedirs(out_dir, exist_ok=True)
    torch.save(acts.float(), os.path.join(out_dir, f"X_{split_name}.pt"))
    
    # Create indices mapping
    indices = list(range(len(df)))
    with open(os.path.join(out_dir, f"indices_{split_name}.json"), "w") as f:
        json.dump(indices, f)
        
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-name", default="imdb", help="HuggingFace dataset path")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training samples to extract")
    parser.add_argument("--n-test", type=int, default=500, help="Number of test samples to extract")
    parser.add_argument("--out-acts", default="data/gemma-probes", help="Output root directory for activations")
    parser.add_argument("--out-df", default="data/gemma-probes-df", help="Output root directory for CSVs")
    args = parser.parse_args()

    print("Loading LLM Backend to extract activations...")
    hf_model, hf_tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        torch_dtype_name="bfloat16"
    )

    df_train = prepare_dataset(args.dataset_name, "train", args.n_train)
    df_test = prepare_dataset(args.dataset_name, "test", args.n_test)
    
    dataset_out_acts = os.path.join(args.out_acts, args.dataset_name)
    df_train = extract_and_save(hf_model, hf_tokenizer, df_train, dataset_out_acts, "train", args.layer)
    df_test = extract_and_save(hf_model, hf_tokenizer, df_test, dataset_out_acts, "test", args.layer)
    
    # Save combined Dataframe (Probe logic dictates indices point to a single master DF)
    os.makedirs(args.out_df, exist_ok=True)
    df_out_path = os.path.join(args.out_df, f"{args.dataset_name}.csv")
    
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all.to_csv(df_out_path, index=False)
    
    # Offset the test indices so they map correctly to the bottom half of df_all
    test_start_idx = len(df_train)
    test_indices = list(range(test_start_idx, test_start_idx + len(df_test)))
    with open(os.path.join(dataset_out_acts, f"indices_test.json"), "w") as f:
        json.dump(test_indices, f)
    
    print(f"\nDone! Pre-computations saved perfectly for Generative Latent Probing!")
    print(f"You can now run the paper's probe evaluation using:")
    print(f"python glp/script_probe.py --cached_acts_folder {args.out_acts} --df_folder {args.out_df} --weights_folder glp-stream")

if __name__ == "__main__":
    main()
