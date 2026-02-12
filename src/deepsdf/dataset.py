import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DEEPSDF_DATASET


class DeepSDFDataset(Dataset):
    """Dataset that lists preprocessed SDF files per-shape.

    Each shape is expected to live at: <root>/<category>/<model_id>/sdf.npz or sdf.npy
    """

    def __init__(self, root: str, extensions: Optional[List[str]] = None):
        self.root = Path(root)
        self.extensions = extensions or DEEPSDF_DATASET["extensions"]
        self.shape_files = []
        # find files
        for category in sorted(self.root.iterdir()):
            if not category.is_dir():
                continue
            for model_dir in sorted(category.iterdir()):
                if not model_dir.is_dir():
                    continue
                for ext in self.extensions:
                    candidate = model_dir / f"sdf{ext}"
                    if candidate.exists():
                        self.shape_files.append(candidate)
                        break

    def __len__(self):
        return len(self.shape_files)

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
