from huggingface_hub import snapshot_download, login
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)

    local_dir = "./data/shapenet_v2_subset"
    os.makedirs(local_dir, exist_ok=True)

    target_synsets = [
        "02747177",  # Chair
        "02691156",  # Airplane
        "04379243",  # Table
        "02958343",  # Car
    ]

    patterns = [f"{sid}.zip" for sid in target_synsets]    
    # Also include the taxonomy file so you know which ID is which
    patterns.append("taxonomy.json")

    print(f"Starting download of {len(target_synsets)} categories...")

    snapshot_download(
        repo_id="ShapeNet/ShapeNetCore",
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=patterns,
        token=True
    )

    print(f"Download complete! Files are in: {local_dir}")

if __name__ == "__main__":
    main()