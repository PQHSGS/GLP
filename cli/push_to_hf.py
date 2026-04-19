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
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Target subfolder in the HF repo. Defaults to local folder name to avoid root-level overwrite.",
    )
    parser.add_argument(
        "--allow-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow uploading to an existing target path in the repo (can overwrite files).",
    )
    return parser


def _normalize_repo_path(path_in_repo: str) -> str:
    # Normalize user-provided path for consistent overlap checks and Hub upload.
    normalized = str(path_in_repo).replace("\\", "/").strip()
    normalized = normalized.strip("/")
    if normalized in {"", "."}:
        raise ValueError("path_in_repo resolves to repository root; provide a non-empty subfolder")
    return normalized


def _target_path_exists(existing_files: set[str], target_path: str) -> bool:
    prefix = f"{target_path}/"
    return any(path == target_path or path.startswith(prefix) for path in existing_files)

def main():
    args = build_parser().parse_args()
    
    if args.token:
        login(token=args.token)
        
    api = HfApi()
    
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder '{args.folder}' does not exist or is not a directory.")
        sys.exit(1)

    target_path = args.path_in_repo or folder_path.name
    try:
        target_path = _normalize_repo_path(target_path)
    except ValueError as exc:
        print(f"Error: {exc}")
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

    existing_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))
    if _target_path_exists(existing_files, target_path) and not args.allow_overlap:
        print(
            f"Error: target path '{target_path}' already exists in {args.repo_id}. "
            "Refusing to overwrite existing checkpoint files."
        )
        print("Choose a different --path-in-repo (or local folder name), or pass --allow-overlap if overwrite is intended.")
        sys.exit(1)
    
    print(f"Uploading files from {args.folder} to {args.repo_id}:{target_path}...")
    api.upload_folder(
        folder_path=str(folder_path),
        path_in_repo=target_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()
