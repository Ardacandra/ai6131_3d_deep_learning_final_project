#!/usr/bin/env python3
"""
Visualization script for VQ-DeepSDF model results.

Produces:
  1. Mesh reconstructions  – predicted vs ground-truth, per shape.
  2. Codebook usage        – bar chart of how often each code is used across all shapes.
  3. Index heatmap         – matrix (shapes × codebook groups) of assigned code indices.
  4. Training loss curves  – total / reconstruction / VQ loss + perplexity over epochs.

Usage
-----
# All shapes (default: 1 per category):
python -m visualization.visualize_deepsdf_vq

# Specific shapes:
python -m visualization.visualize_deepsdf_vq --shape-indices 0 1 2

# Point-cloud mode (no marching cubes):
python -m visualization.visualize_deepsdf_vq --pointcloud

# Custom checkpoint / output:
python -m visualization.visualize_deepsdf_vq \
    --checkpoint out/deepsdf_vq/deepsdf_vq_latest.pth \
    --output-dir out/viz_vq
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import trimesh

from src.deepsdf_vq.evaluate import load_vq_checkpoint, _resolve_checkpoint_path
from src.deepsdf_vq.model import DeepSDFDecoder
from src.deepsdf.evaluate import sdf_to_mesh, sdf_to_pointcloud, load_ground_truth_mesh
from src.deepsdf.dataset import DeepSDFDataset
from src.deepsdf.evaluate import _load_selected_shape_paths
from config import DEEPSDF_VQ_EVALUATION, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _plot_mesh_3d(ax, mesh: trimesh.Trimesh, color: str, title: str) -> None:
    verts = mesh.vertices
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=mesh.faces, color=color, alpha=0.25,
        edgecolor="none", linewidth=0,
    )
    lim = float(np.abs(verts).max()) if len(verts) else 1.0
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=11, fontweight="bold")


def _plot_pointcloud_3d(ax, pts: np.ndarray, color: str, title: str) -> None:
    max_disp = 20_000
    if len(pts) > max_disp:
        pts = pts[np.random.choice(len(pts), max_disp, replace=False)]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=1, alpha=0.5)
    lim = float(np.abs(pts).max()) if len(pts) else 1.0
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=11, fontweight="bold")


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------

class VQDeepSDFVisualizer:
    """Visualize reconstructions and VQ codebook statistics for VQ-DeepSDF."""

    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        gt_data_root: str,
        output_dir: Path,
        device: Optional[str] = None,
    ):
        self.device = device or _get_device()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gt_data_root = Path(gt_data_root)

        print(f"Loading VQ-DeepSDF checkpoint from {checkpoint_path}")
        self.decoder, self.latents, self.meta = load_vq_checkpoint(checkpoint_path, device=self.device)
        # latents: [num_shapes, latent_size] float tensor (already quantized)

        # Also load the raw checkpoint for diagnostics (indices, loss log, etc.)
        resolved = _resolve_checkpoint_path(checkpoint_path)
        self._raw_ckpt = torch.load(resolved, map_location="cpu", weights_only=False)

        self.num_codebooks = int(self.meta.get("num_codebooks", 16))
        self.codebook_size = int(self.meta.get("codebook_size", 512))

        # Shape paths
        selected = _load_selected_shape_paths(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
        )
        if selected is not None:
            self.shape_paths = selected
            print(f"  {len(self.shape_paths)} shapes from selected_samples manifest")
        else:
            ds = DeepSDFDataset(data_root)
            self.shape_paths = ds.get_shape_paths()
            print(f"  {len(self.shape_paths)} shapes from dataset")

        print(f"Output directory: {self.output_dir.resolve()}")

    # ------------------------------------------------------------------
    # Reconstruction (mesh / point-cloud)
    # ------------------------------------------------------------------

    def visualize_shape_mesh(self, shape_index: int, resolution: int = 128) -> Optional[str]:
        """Render predicted mesh vs. ground-truth for one shape. Returns saved path."""
        shape_path = Path(self.shape_paths[shape_index])
        latent = self.latents[shape_index].unsqueeze(0)  # [1, L]

        print(f"\nShape {shape_index}: {shape_path.parent.parent.name}/{shape_path.parent.name}")

        pred_mesh = sdf_to_mesh(self.decoder, latent, resolution=resolution, device=self.device)
        if pred_mesh is None:
            print("  [skip] mesh extraction failed")
            return None
        print(f"  Predicted  – {len(pred_mesh.vertices)} verts, {len(pred_mesh.faces)} faces")

        gt_mesh = load_ground_truth_mesh(shape_path, self.gt_data_root)
        if gt_mesh is None:
            print("  [skip] ground-truth mesh not found")
            return None
        print(f"  GT mesh    – {len(gt_mesh.vertices)} verts, {len(gt_mesh.faces)} faces")

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        _plot_mesh_3d(ax1, pred_mesh, "steelblue", f"VQ-DeepSDF (shape {shape_index})")
        _plot_mesh_3d(ax2, gt_mesh, "coral", f"Ground Truth (shape {shape_index})")
        plt.tight_layout()

        out = self.output_dir / f"vq_mesh_{shape_index:03d}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")
        return str(out)

    def visualize_shape_pointcloud(self, shape_index: int, resolution: int = 128) -> Optional[str]:
        """Render predicted point cloud vs. ground-truth for one shape."""
        shape_path = Path(self.shape_paths[shape_index])
        latent = self.latents[shape_index].unsqueeze(0)

        print(f"\nShape {shape_index}: {shape_path.parent.parent.name}/{shape_path.parent.name}")

        pred_pts = sdf_to_pointcloud(self.decoder, latent, resolution=resolution, device=self.device)
        if pred_pts is None:
            print("  [skip] point cloud extraction failed")
            return None
        print(f"  Predicted  – {len(pred_pts)} points")

        gt_mesh = load_ground_truth_mesh(shape_path, self.gt_data_root)
        if gt_mesh is None:
            print("  [skip] ground-truth mesh not found")
            return None
        gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, min(len(pred_pts), 30_000))
        gt_pts = gt_pts.astype(np.float32)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        _plot_pointcloud_3d(ax1, pred_pts, "steelblue", f"VQ-DeepSDF (shape {shape_index})")
        _plot_pointcloud_3d(ax2, gt_pts, "coral", f"Ground Truth (shape {shape_index})")
        plt.tight_layout()

        out = self.output_dir / f"vq_pointcloud_{shape_index:03d}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")
        return str(out)

    def visualize_multiple(
        self,
        shape_indices: Optional[List[int]] = None,
        num_per_category: int = 1,
        resolution: int = 128,
        pointcloud: bool = False,
    ) -> List[str]:
        """Visualize a list of shapes (or 1 per category if indices omitted)."""
        if shape_indices is None:
            shape_indices = self._indices_per_category(num_per_category)

        saved = []
        for idx in shape_indices:
            if pointcloud:
                path = self.visualize_shape_pointcloud(idx, resolution=resolution)
            else:
                path = self.visualize_shape_mesh(idx, resolution=resolution)
            if path:
                saved.append(path)
        print(f"\n{len(saved)}/{len(shape_indices)} shapes visualised → {self.output_dir}")
        return saved

    # ------------------------------------------------------------------
    # VQ codebook diagnostics
    # ------------------------------------------------------------------

    def plot_codebook_usage(self) -> str:
        """Bar chart: how many shapes use each code index (aggregated over groups)."""
        indices_np = self._raw_ckpt.get("shape_code_indices")  # [num_shapes, num_codebooks]
        if indices_np is None:
            print("[skip] no shape_code_indices in checkpoint")
            return ""

        counts = np.zeros(self.codebook_size, dtype=np.int64)
        for code_idx in range(self.num_codebooks):
            col = indices_np[:, code_idx]
            for v in col:
                counts[v] += 1

        active = int((counts > 0).sum())
        total = self.codebook_size
        perplexity = float(np.exp(-(counts / counts.sum() * np.log(counts / counts.sum() + 1e-10)).sum()))

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(np.arange(total), counts, width=1.0, color="steelblue", alpha=0.8)
        ax.set_xlabel("Code index")
        ax.set_ylabel("Frequency (summed over groups)")
        ax.set_title(
            f"Codebook usage  |  active codes: {active}/{total}  |  perplexity: {perplexity:.1f}",
            fontsize=12,
        )
        ax.set_xlim([0, total])
        plt.tight_layout()

        out = self.output_dir / "vq_codebook_usage.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Codebook usage saved → {out}")
        return str(out)

    def plot_index_heatmap(self) -> str:
        """Heat-map of assigned code indices: rows = shapes, columns = codebook groups."""
        indices_np = self._raw_ckpt.get("shape_code_indices")  # [S, M]
        if indices_np is None:
            print("[skip] no shape_code_indices in checkpoint")
            return ""

        num_shapes, num_groups = indices_np.shape
        fig_h = max(4, num_shapes * 0.35)
        fig, ax = plt.subplots(figsize=(min(18, num_groups * 0.6 + 2), fig_h))

        im = ax.imshow(
            indices_np,
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=self.codebook_size - 1,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Code index")
        ax.set_xlabel("Codebook group")
        ax.set_ylabel("Shape index")
        ax.set_title("Per-shape discrete code assignments (VQ-DeepSDF)", fontsize=12)

        ax.set_yticks(np.arange(num_shapes))
        ax.set_yticklabels(np.arange(1, num_shapes + 1), fontsize=7)

        plt.tight_layout()
        out = self.output_dir / "vq_index_heatmap.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Index heatmap saved → {out}")
        return str(out)

    def plot_training_curves(self) -> str:
        """Four-panel training curves: total loss, reconstruction, VQ, and perplexity."""
        loss_log = self._raw_ckpt.get("loss_log", [])
        if not loss_log:
            print("[skip] no loss_log in checkpoint")
            return ""

        # loss_log entries are dicts with keys: total, recon, vq, entropy, perplexity
        epochs = np.arange(1, len(loss_log) + 1)

        def _extract(key: str) -> np.ndarray:
            return np.array([
                e[key] if isinstance(e, dict) else float("nan")
                for e in loss_log
            ])

        total      = _extract("total")
        recon      = _extract("recon")
        vq         = _extract("vq")
        perplexity = _extract("perplexity")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("VQ-DeepSDF Training Curves", fontsize=14, fontweight="bold")

        for ax, values, label, color in [
            (axes[0, 0], total,      "Total loss",         "royalblue"),
            (axes[0, 1], recon,      "Reconstruction loss","steelblue"),
            (axes[1, 0], vq,         "VQ loss",            "darkorange"),
            (axes[1, 1], perplexity, "Codebook perplexity","seagreen"),
        ]:
            ax.plot(epochs, values, color=color, linewidth=1.2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / "vq_training_curves.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved → {out}")
        return str(out)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def visualize_all(self, resolution: int = 128, pointcloud: bool = False, num_per_category: int = 1) -> None:
        """Run all visualizations: shapes + codebook diagnostics + training curves."""
        print("\n=== Shape reconstructions (mesh) ===")
        self.visualize_multiple(resolution=resolution, pointcloud=False, num_per_category=num_per_category)

        print("\n=== Shape reconstructions (point cloud) ===")
        self.visualize_multiple(resolution=resolution, pointcloud=True, num_per_category=num_per_category)

        print("\n=== Codebook usage ===")
        self.plot_codebook_usage()

        print("\n=== Index heatmap ===")
        self.plot_index_heatmap()

        print("\n=== Training curves ===")
        self.plot_training_curves()

        print(f"\nAll outputs in: {self.output_dir.resolve()}")

    def _indices_per_category(self, n: int) -> List[int]:
        cats: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(self.shape_paths):
            cats[Path(p).parent.parent.name].append(i)
        out = []
        for cat_indices in cats.values():
            out.extend(cat_indices[:n])
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-DeepSDF results")
    parser.add_argument(
        "--checkpoint",
        default=str(DEEPSDF_VQ_EVALUATION["checkpoint"]),
        help="Path to VQ-DeepSDF checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEEPSDF_VQ_EVALUATION["data_root"]),
        help="Root directory for SDF data (default: %(default)s)",
    )
    parser.add_argument(
        "--gt-data-root",
        default=str(DEEPSDF_VQ_EVALUATION["gt_data_root"]),
        help="Root directory for ground-truth meshes (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR / "viz_vq"),
        help="Output directory for saved images (default: %(default)s)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEEPSDF_VQ_EVALUATION["resolution"],
        help="Marching-cubes grid resolution (default: %(default)s)",
    )
    parser.add_argument(
        "--shape-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit shape indices to visualise (e.g. 0 1 2). "
             "If omitted, one shape per category is used.",
    )
    parser.add_argument(
        "--num-per-category",
        type=int,
        default=1,
        help="Shapes per category when --shape-indices is not set (default: %(default)s)",
    )
    parser.add_argument(
        "--pointcloud",
        action="store_true",
        help="Use point-cloud mode instead of marching cubes.",
    )
    parser.add_argument(
        "--codebook-only",
        action="store_true",
        help="Skip mesh rendering; only produce codebook / training-curve plots.",
    )

    args = parser.parse_args()

    viz = VQDeepSDFVisualizer(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        gt_data_root=args.gt_data_root,
        output_dir=Path(args.output_dir),
        device=args.device,
    )

    if args.codebook_only:
        viz.plot_codebook_usage()
        viz.plot_index_heatmap()
        viz.plot_training_curves()
    else:
        viz.visualize_all(
            resolution=args.resolution,
            pointcloud=args.pointcloud,
            num_per_category=args.num_per_category,
        )


if __name__ == "__main__":
    main()
