import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, login

def build_parser():
    parser = argparse.ArgumentParser(description="Push a folder to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo ID like 'username/my-model'")
    parser.add_argument("--folder", required=True, help="Local folder to push, e.g. glp_stream/10M")
    parser.add_argument("--token", default=None, help="HF Token (optional, otherwise reads from HF_TOKEN or credentials)")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    return parser

def main():
    args = build_parser().parse_args()
    
    if args.token:
        login(token=args.token)
        
    api = HfApi()
    
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder '{args.folder}' does not exist or is not a directory.")
        sys.exit(1)
        
    # Check if necessary files are present
    print(f"Validating folder {folder_path} contents...")
    required_files = ["final.safetensors", "rep_statistics.pt", "config.yaml"]
    missing_files = []
    for f in required_files:
        if not (folder_path / f).exists():
            missing_files.append(f)
            
    if missing_files:
        print(f"WARNING: The following standard GLP files are missing from {folder_path}:")
        for f in missing_files:
            print(f"  - {f}")
        print("The model might not load properly via `load_glp` without these files.")
        response = input("Do you want to continue pushing? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            sys.exit(0)
            
    print(f"Creating repo {args.repo_id} (if it doesn't exist)...")
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
    
    print(f"Uploading files from {args.folder} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()
