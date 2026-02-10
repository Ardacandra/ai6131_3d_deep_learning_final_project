from huggingface_hub import snapshot_download, login
import os
from dotenv import load_dotenv
from config import DATA_DIR, TARGET_SYNSETS, SHAPENET_CATEGORIES

def main():
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)

    local_dir = str(DATA_DIR)
    os.makedirs(local_dir, exist_ok=True)

    patterns = [f"{sid}.zip" for sid in TARGET_SYNSETS]    
    # Also include the taxonomy file so you know which ID is which
    patterns.append("taxonomy.json")

    print(f"Starting download of {len(TARGET_SYNSETS)} categories...")
    print("Categories to download:")
    for synset_id in TARGET_SYNSETS:
        print(f"  • {SHAPENET_CATEGORIES.get(synset_id, 'Unknown')} ({synset_id})")

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