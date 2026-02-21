from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DEEPSDF_DATASET


class DeepSDFDataset(Dataset):
    """Dataset that lists preprocessed SDF files per-shape.

    Each shape is expected to live at: <root>/<category>/<model_id>/sdf.npz or sdf.npy
    """

    def __init__(
        self,
        root: str,
        extensions: Optional[List[str]] = None,
        objects_per_category: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        self.root = Path(root)
        self.extensions = extensions or DEEPSDF_DATASET["extensions"]
        self.objects_per_category = objects_per_category
        self.random_seed = random_seed
        self.shape_files = []
        category_shape_files: Dict[str, List[Path]] = {}

        # Find files grouped by category.
        for category in sorted(self.root.iterdir()):
            if not category.is_dir():
                continue

            category_files: List[Path] = []
            for model_dir in sorted(category.iterdir()):
                if not model_dir.is_dir():
                    continue
                for ext in self.extensions:
                    candidate = model_dir / f"sdf{ext}"
                    if candidate.exists():
                        category_files.append(candidate)
                        break

            if category_files:
                category_shape_files[category.name] = category_files

        if self.objects_per_category is None:
            for category_name in sorted(category_shape_files):
                self.shape_files.extend(category_shape_files[category_name])
        else:
            rng = np.random.default_rng(self.random_seed)
            for category_name in sorted(category_shape_files):
                category_files = category_shape_files[category_name]
                sample_count = min(self.objects_per_category, len(category_files))
                if sample_count == len(category_files):
                    selected_files = category_files
                else:
                    selected_indices = rng.choice(
                        len(category_files), size=sample_count, replace=False
                    )
                    selected_files = [category_files[idx] for idx in sorted(selected_indices)]
                self.shape_files.extend(selected_files)

    def __len__(self):
        return len(self.shape_files)

    def get_shape_paths(self) -> List[Path]:
        return list(self.shape_files)

    def get_shape_ids(self) -> List[Dict[str, str]]:
        shape_ids = []
        for path in self.shape_files:
            shape_ids.append(
                {
                    "category_id": path.parent.parent.name,
                    "model_id": path.parent.name,
                    "relative_path": str(path.relative_to(self.root)),
                }
            )
        return shape_ids

    def __getitem__(self, idx):
        path = self.shape_files[idx]
        ext = path.suffix.lower()

        if ext == ".npz":
            data = np.load(path)
            # Try to support datasets with separate pos/neg or combined array
            if "pos" in data and "neg" in data:
                pos = data["pos"]
                neg = data["neg"]
                arr = np.vstack([pos, neg]) if len(pos) + len(neg) > 0 else np.zeros((0, 4))
            else:
                # fallback: all samples in single array
                arr = data[list(data.files)[0]]
        else:
            arr = np.load(path)

        # arr shape: [N,4] or [N,] - ensure shape
        if arr.ndim == 1:
            arr = arr.reshape(-1, 4)

        # Provide as numpy arrays; training loop will sample points
        return {
            "path": str(path),
            "data": arr.astype(np.float32)
        }
