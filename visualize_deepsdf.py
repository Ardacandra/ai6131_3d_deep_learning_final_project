#!/usr/bin/env python3
"""
Visualization script for DeepSDF model reconstructions.

Displays:
- Extracted mesh from trained DeepSDF model
- Ground truth mesh from dataset
- Side-by-side comparison
- Saves visualizations to output directory
"""

import argparse
from pathlib import Path
import json
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

from src.deepsdf.evaluate import load_checkpoint, sdf_to_mesh, load_ground_truth_mesh
from src.deepsdf.dataset import DeepSDFDataset
from config import DEEPSDF_EVALUATION, OUTPUT_DIR


class DeepSDFVisualizer:
    """Visualize DeepSDF model reconstructions"""
    
    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        gt_data_root: str,
        output_dir: Path = None,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.gt_data_root = Path(gt_data_root)
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        print(f"Loading checkpoint from {checkpoint_path}")
        self.decoder, self.latent_embeddings, self.meta = load_checkpoint(checkpoint_path, device=device)
        
        print(f"Loading dataset from {data_root}")
        self.dataset = DeepSDFDataset(data_root)
        
        print(f"Output directory: {self.output_dir.resolve()}")
    
    def get_shape_info(self, shape_index: int) -> Tuple[str, int]:
        """Get shape path and latent code for a given index."""
        if shape_index >= len(self.dataset):
            raise IndexError(f"Shape index {shape_index} out of range (dataset has {len(self.dataset)} shapes)")
        
        shape_data = self.dataset[shape_index]
        shape_path = Path(shape_data["path"])
        return shape_path, shape_index
    
    def visualize_shape(
        self,
        shape_index: int,
        resolution: int = DEEPSDF_EVALUATION["resolution"],
        save: bool = True,
    ) -> Optional[Tuple[trimesh.Trimesh, trimesh.Trimesh]]:
        """
        Visualize a single shape: predicted mesh vs ground truth.
        
        Args:
            shape_index: Index of shape in dataset
            resolution: Resolution for marching cubes
            save: Whether to save visualization to file
            
        Returns:
            Tuple of (predicted_mesh, ground_truth_mesh) or None if visualization fails
        """
        shape_path, idx = self.get_shape_info(shape_index)
        
        print(f"\n{'='*70}")
        print(f"Visualizing shape {shape_index}: {shape_path.parent.name}/{shape_path.parent.parent.name}")
        print(f"{'='*70}")
        
        # Extract predicted mesh
        latent_code = self.latent_embeddings.weight[idx].unsqueeze(0)
        pred_mesh = sdf_to_mesh(self.decoder, latent_code, resolution=resolution, device=self.device)
        
        if pred_mesh is None or len(pred_mesh.vertices) == 0:
            print("❌ Failed to extract predicted mesh")
            return None
        
        print(f"✅ Predicted mesh: {len(pred_mesh.vertices)} vertices, {len(pred_mesh.faces)} faces")
        
        # Load ground truth mesh
        gt_mesh = load_ground_truth_mesh(shape_path, self.gt_data_root)
        
        if gt_mesh is None or len(gt_mesh.vertices) == 0:
            print("❌ Failed to load ground truth mesh")
            return None
        
        print(f"✅ Ground truth mesh: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.faces)} faces")
        
        return pred_mesh, gt_mesh
    
    def plot_mesh_comparison(
        self,
        pred_mesh: trimesh.Trimesh,
        gt_mesh: trimesh.Trimesh,
        shape_index: int,
        title_suffix: str = "",
    ) -> str:
        """
        Create side-by-side 3D visualization of predicted vs ground truth meshes.
        
        Args:
            pred_mesh: Predicted mesh from model
            gt_mesh: Ground truth mesh
            shape_index: Index of shape for filename
            title_suffix: Additional text for title
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 7))
        
        # Predicted mesh
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_mesh_3d(ax1, pred_mesh, color='steelblue', title=f"Predicted Mesh{title_suffix}")
        
        # Ground truth mesh
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_mesh_3d(ax2, gt_mesh, color='coral', title=f"Ground Truth Mesh{title_suffix}")
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"deepsdf_reconstruction_{shape_index:03d}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"💾 Saved visualization to {output_path}")
        plt.close()
        
        return str(output_path)
    
    @staticmethod
    def _plot_mesh_3d(ax, mesh: trimesh.Trimesh, color: str = 'steelblue', title: str = ""):
        """Helper to plot a mesh in 3D."""
        # Sample surface points for visualization
        if len(mesh.vertices) > 5000:
            indices = np.random.choice(len(mesh.vertices), 5000, replace=False)
            verts = mesh.vertices[indices]
        else:
            verts = mesh.vertices
        
        # Plot as scatter
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=color, s=1, alpha=0.6)
        
        # Optionally plot wireframe for small meshes
        if len(mesh.faces) < 10000:
            ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                mesh.vertices[:, 2],
                triangles=mesh.faces,
                color=color,
                alpha=0.1,
                edgecolor='gray',
                linewidth=0.1
            )
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Auto scale
        lim = np.max(np.abs(mesh.vertices))
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
    
    def visualize_multiple(
        self,
        shape_indices: Optional[List[int]] = None,
        num_random: int = 5,
        resolution: int = DEEPSDF_EVALUATION["resolution"],
    ):
        """
        Visualize multiple shapes.
        
        Args:
            shape_indices: List of specific shape indices to visualize.
                          If None and num_random is set, randomly select shapes.
            num_random: Number of random shapes to visualize (if shape_indices is None)
            resolution: Resolution for marching cubes
        """
        # Determine which shapes to visualize
        if shape_indices is None:
            # Randomly select shapes
            total_shapes = len(self.dataset)
            num_random = min(num_random, total_shapes)
            shape_indices = np.random.choice(total_shapes, num_random, replace=False)
        
        print(f"\nVisualizing {len(shape_indices)} shapes...")
        
        successful = 0
        failed = 0
        
        for shape_idx in shape_indices:
            try:
                result = self.visualize_shape(shape_idx, resolution=resolution)
                if result is not None:
                    pred_mesh, gt_mesh = result
                    self.plot_mesh_comparison(pred_mesh, gt_mesh, shape_idx)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Error visualizing shape {shape_idx}: {e}")
                failed += 1
        
        # Summary
        print(f"\n{'='*70}")
        print(f"VISUALIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {self.output_dir.resolve()}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DeepSDF model reconstructions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEEPSDF_EVALUATION["checkpoint"]),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEEPSDF_EVALUATION["data_root"]),
        help="Root directory for SDF data"
    )
    parser.add_argument(
        "--gt-data-root",
        type=str,
        default=str(DEEPSDF_EVALUATION["gt_data_root"]),
        help="Root directory for ground truth meshes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda/cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEEPSDF_EVALUATION["resolution"],
        help="Grid resolution for marching cubes"
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=5,
        help="Number of random shapes to visualize"
    )
    parser.add_argument(
        "--shape-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific shape indices to visualize (e.g., 0 1 2 3)"
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = DeepSDFVisualizer(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        gt_data_root=args.gt_data_root,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device=args.device,
    )
    
    # Visualize shapes
    if args.shape_indices:
        visualizer.visualize_multiple(
            shape_indices=args.shape_indices,
            resolution=args.resolution,
        )
    else:
        visualizer.visualize_multiple(
            shape_indices=None,
            num_random=args.num_shapes,
            resolution=args.resolution,
        )


if __name__ == "__main__":
    main()
