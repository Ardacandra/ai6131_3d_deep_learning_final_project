"""
Evaluation script for DeepSDF model.

Computes the following metrics:
- Chamfer Distance (Mean and Median)
- Earth Mover's Distance (Mean and Median)
- Mesh accuracy (minimum distance d such that 90% of generated points are within d of the ground truth mesh)
"""

import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
try:
    import skimage
    from skimage import measure
except ImportError:
    import skimage.measure as measure

from .dataset import DeepSDFDataset
from .model import DeepSDFDecoder
from config import DEEPSDF_TRAINING, DEEPSDF_EVALUATION


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.exists():
        return path

    parent = path.parent
    candidates = sorted(
        parent.glob("deepsdf_epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if candidates:
        fallback = candidates[-1]
        print(
            f"Checkpoint not found: {path}. Falling back to latest epoch checkpoint: {fallback}"
        )
        return fallback

    raise FileNotFoundError(
        f"Checkpoint not found: {path}. No fallback checkpoint matching deepsdf_epoch_*.pth in {parent}"
    )


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Tuple[DeepSDFDecoder, torch.nn.Embedding, Dict]:
    """Load trained DeepSDF model and latent codes from checkpoint."""
    resolved_checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint_path, map_location=device, weights_only=False)
    
    # Load metadata
    meta_path = resolved_checkpoint_path.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        meta = {}
    
    # Get model configuration from meta.json or use defaults
    latent_size = meta.get("latent_size", DEEPSDF_TRAINING["latent_size"])
    hidden_size = meta.get("hidden_size", DEEPSDF_TRAINING["hidden_size"])
    
    # Create and load decoder
    decoder = DeepSDFDecoder(latent_size=latent_size, hidden_size=hidden_size).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    decoder.eval()
    
    # Load latent codes
    latents = checkpoint["latents"]
    if isinstance(latents, torch.Tensor):
        latents_tensor = latents.to(device)
    else:
        # Convert numpy array to tensor
        latents_tensor = torch.from_numpy(latents).float().to(device)
    
    num_shapes = latents_tensor.shape[0]
    latent_embeddings = torch.nn.Embedding(num_shapes, latent_size).to(device)
    latent_embeddings.weight.data = latents_tensor
    
    return decoder, latent_embeddings, meta


def sdf_to_mesh(
    decoder: DeepSDFDecoder,
    latent_code: torch.Tensor,
    resolution: int = DEEPSDF_EVALUATION["resolution"],
    bounds: Tuple[float, float] = (-1.0, 1.0),
    device: str = "cpu",
    batch_size: int = DEEPSDF_EVALUATION["batch_size"],
) -> Optional[trimesh.Trimesh]:
    """
    Extract mesh from DeepSDF using marching cubes.
    
    Args:
        decoder: DeepSDF decoder model
        latent_code: Latent code for the shape
        resolution: Grid resolution for marching cubes
        bounds: Bounding box (min, max)
        device: Device for computation
        batch_size: Batch size for SDF queries
    
    Returns:
        Trimesh mesh or None if extraction fails
    """
    # Create grid
    grid_points = torch.linspace(bounds[0], bounds[1], resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    # Query SDF values in batches
    sdf_values = []
    decoder.eval()
    with torch.no_grad():
        for i in range(0, len(grid_xyz), batch_size):
            batch_xyz = grid_xyz[i:i+batch_size]
            batch_latent = latent_code.expand(len(batch_xyz), -1)
            batch_sdf = decoder(batch_latent, batch_xyz)
            sdf_values.append(batch_sdf.squeeze(-1))
    
    sdf_values = torch.cat(sdf_values, dim=0)
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution).cpu().numpy()
    
    # Check SDF range and determine appropriate level
    sdf_min, sdf_max = sdf_grid.min(), sdf_grid.max()
    
    # If 0 is not in the range, use a level that is (e.g., median or mean)
    if sdf_min > 0 or sdf_max < 0:
        # All values are on one side of zero - use median as level
        level = np.median(sdf_grid)
        print(f"  Warning: SDF range [{sdf_min:.4f}, {sdf_max:.4f}] doesn't cross 0. Using level={level:.4f}")
    else:
        level = 0.0
    
    # Apply marching cubes
    try:
        verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=level)
        
        # Scale vertices back to original bounds
        verts = verts / (resolution - 1) * (bounds[1] - bounds[0]) + bounds[0]
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Verify mesh is valid
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            print(f"  Marching cubes produced empty mesh")
            return None
            
        return mesh
    except (ValueError, RuntimeError) as e:
        print(f"  Marching cubes failed: {e}")
        return None


def load_ground_truth_mesh(shape_path: Path, data_root: Path) -> Optional[trimesh.Trimesh]:
    """
    Load ground truth mesh for a shape.
    
    Args:
        shape_path: Path to the SDF file (e.g., data/shapenet_sdf/02691156/xxx/sdf.npz)
        data_root: Root directory for ground truth meshes (e.g., data/shapenet_v2_subset)
    
    Returns:
        Trimesh mesh or None if loading fails
    """
    # Parse category and model ID from path
    parts = shape_path.parts
    category_idx = None
    for i, part in enumerate(parts):
        if part == "shapenet_sdf":
            category_idx = i + 1
            break
    
    if category_idx is None or category_idx + 1 >= len(parts):
        print(f"Cannot parse category/model from path: {shape_path}")
        return None
    
    category = parts[category_idx]
    model_id = parts[category_idx + 1]
    
    # Construct path to ground truth mesh (mesh is in models/ subdirectory)
    gt_mesh_path = data_root / category / model_id / "models" / "model_normalized.obj"
    
    if not gt_mesh_path.exists():
        print(f"Ground truth mesh not found: {gt_mesh_path}")
        return None
    
    try:
        mesh = trimesh.load(gt_mesh_path, force='mesh', process=False)
        return mesh
    except Exception as e:
        print(f"Failed to load ground truth mesh {gt_mesh_path}: {e}")
        return None


def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int = 10000) -> np.ndarray:
    """
    Sample points uniformly from mesh surface.
    
    Args:
        mesh: Trimesh mesh
        num_points: Number of points to sample
    
    Returns:
        Point cloud of shape (num_points, 3)
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def chamfer_distance(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[float, float]:
    """
    Compute bidirectional Chamfer Distance.
    
    Args:
        pts1: Point cloud 1, shape (N, 3)
        pts2: Point cloud 2, shape (M, 3)
    
    Returns:
        (mean_chamfer, median_chamfer)
    """
    # Build KD-trees
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    
    # Nearest neighbor distances
    dist1, _ = tree2.query(pts1)  # pts1 to pts2
    dist2, _ = tree1.query(pts2)  # pts2 to pts1
    
    # Bidirectional Chamfer Distance
    chamfer_distances = np.concatenate([dist1, dist2])
    
    mean_chamfer = np.mean(chamfer_distances)
    median_chamfer = np.median(chamfer_distances)
    
    return mean_chamfer, median_chamfer


def earth_movers_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """
    Compute Earth Mover's Distance (Wasserstein distance) between two point clouds.
    
    This is computed as the average EMD across the three coordinate dimensions.
    
    Args:
        pts1: Point cloud 1, shape (N, 3)
        pts2: Point cloud 2, shape (M, 3)
    
    Returns:
        Average EMD across dimensions
    """
    emds = []
    for dim in range(3):
        emd = wasserstein_distance(pts1[:, dim], pts2[:, dim])
        emds.append(emd)
    
    return np.mean(emds)


def mesh_accuracy(pred_pts: np.ndarray, gt_pts: np.ndarray, percentile: float = DEEPSDF_EVALUATION["percentile"]) -> float:
    """
    Compute mesh accuracy: minimum distance d such that {percentile}% of 
    generated points are within d of the ground truth mesh.
    
    Args:
        pred_pts: Predicted point cloud, shape (N, 3)
        gt_pts: Ground truth point cloud, shape (M, 3)
        percentile: Percentile threshold (default 90%)
    
    Returns:
        Distance threshold at the given percentile
    """
    # Build KD-tree for ground truth
    tree_gt = cKDTree(gt_pts)
    
    # Find nearest neighbor distances from predicted to ground truth
    distances, _ = tree_gt.query(pred_pts)
    
    # Compute percentile
    accuracy = np.percentile(distances, percentile)
    
    return accuracy


def evaluate_shape(
    decoder: DeepSDFDecoder,
    latent_code: torch.Tensor,
    gt_mesh_path: Path,
    data_root: Path,
    device: str = "cpu",
    resolution: int = 128,
    num_sample_points: int = 10000,
) -> Optional[Dict]:
    """
    Evaluate a single shape.
    
    Args:
        decoder: DeepSDF decoder
        latent_code: Latent code for the shape
        gt_mesh_path: Path to SDF file (used to locate ground truth mesh)
        data_root: Root directory for ground truth meshes
        device: Computation device
        resolution: Resolution for marching cubes
        num_sample_points: Number of points to sample from meshes
    
    Returns:
        Dictionary of metrics or None if evaluation fails
    """
    # Extract predicted mesh
    pred_mesh = sdf_to_mesh(decoder, latent_code, resolution=resolution, device=device)
    if pred_mesh is None:
        print(f"  Skipped: Failed to extract mesh")
        return None
    if len(pred_mesh.vertices) == 0:
        print(f"  Skipped: Extracted mesh is empty")
        return None
    
    # Load ground truth mesh
    gt_mesh = load_ground_truth_mesh(gt_mesh_path, data_root)
    if gt_mesh is None:
        print(f"  Skipped: Failed to load ground truth mesh")
        return None
    if len(gt_mesh.vertices) == 0:
        print(f"  Skipped: Ground truth mesh is empty")
        return None
    
    # Sample points from both meshes
    pred_pts = sample_points_from_mesh(pred_mesh, num_sample_points)
    gt_pts = sample_points_from_mesh(gt_mesh, num_sample_points)
    
    # Compute metrics
    mean_cd, median_cd = chamfer_distance(pred_pts, gt_pts)
    emd = earth_movers_distance(pred_pts, gt_pts)
    acc_90 = mesh_accuracy(pred_pts, gt_pts, percentile=90.0)
    
    return {
        "chamfer_distance_mean": float(mean_cd),
        "chamfer_distance_median": float(median_cd),
        "earth_movers_distance": float(emd),
        "mesh_accuracy_90": float(acc_90),
    }


def evaluate_dataset(
    checkpoint_path: str,
    data_root: str,
    gt_data_root: str,
    device: str = "cpu",
    resolution: int = 128,
    num_sample_points: int = 10000,
    max_shapes: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate DeepSDF on entire dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Root directory for SDF data (for indexing)
        gt_data_root: Root directory for ground truth meshes
        device: Computation device
        resolution: Resolution for marching cubes
        num_sample_points: Number of points to sample from meshes
        max_shapes: Maximum number of shapes to evaluate (None for all)
        output_path: Path to save detailed results JSON
    
    Returns:
        Dictionary of aggregate metrics
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    decoder, latent_embeddings, meta = load_checkpoint(checkpoint_path, device=device)
    
    print(f"Loading dataset from {data_root}")
    dataset = DeepSDFDataset(data_root)
    
    num_shapes = len(dataset)
    if max_shapes is not None:
        num_shapes = min(num_shapes, max_shapes)
    
    print(f"Evaluating {num_shapes} shapes...")
    
    results = []
    chamfer_means = []
    chamfer_medians = []
    emds = []
    accuracies = []
    
    start_time = time.time()
    
    for idx in range(num_shapes):
        shape_data = dataset[idx]
        shape_path = Path(shape_data["path"])
        
        print(f"[{idx+1}/{num_shapes}] Evaluating {shape_path.name} from {shape_path.parent.name}/{shape_path.parent.parent.name}")
        
        # Get latent code
        latent_code = latent_embeddings.weight[idx].unsqueeze(0)
        
        # Evaluate shape
        metrics = evaluate_shape(
            decoder=decoder,
            latent_code=latent_code,
            gt_mesh_path=shape_path,
            data_root=Path(gt_data_root),
            device=device,
            resolution=resolution,
            num_sample_points=num_sample_points,
        )
        
        if metrics is None:
            print(f"  Skipped due to error")
            continue
        
        # Collect metrics
        chamfer_means.append(metrics["chamfer_distance_mean"])
        chamfer_medians.append(metrics["chamfer_distance_median"])
        emds.append(metrics["earth_movers_distance"])
        accuracies.append(metrics["mesh_accuracy_90"])
        
        # Store detailed result
        results.append({
            "shape_index": idx,
            "shape_path": str(shape_path),
            **metrics
        })
        
        print(f"  CD mean: {metrics['chamfer_distance_mean']:.6f}, "
              f"CD median: {metrics['chamfer_distance_median']:.6f}, "
              f"EMD: {metrics['earth_movers_distance']:.6f}, "
              f"Acc@90: {metrics['mesh_accuracy_90']:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    
    # Compute aggregate metrics (handle empty lists gracefully)
    if len(results) > 0:
        aggregate_metrics = {
            "num_shapes_evaluated": len(results),
            "num_shapes_total": num_shapes,
            "chamfer_distance_mean_avg": float(np.mean(chamfer_means)),
            "chamfer_distance_mean_median": float(np.median(chamfer_means)),
            "chamfer_distance_median_avg": float(np.mean(chamfer_medians)),
            "chamfer_distance_median_median": float(np.median(chamfer_medians)),
            "earth_movers_distance_mean": float(np.mean(emds)),
            "earth_movers_distance_median": float(np.median(emds)),
            "mesh_accuracy_90_mean": float(np.mean(accuracies)),
            "mesh_accuracy_90_median": float(np.median(accuracies)),
            "evaluation_time_seconds": elapsed,
        }
    else:
        # No shapes were successfully evaluated
        aggregate_metrics = {
            "num_shapes_evaluated": 0,
            "num_shapes_total": num_shapes,
            "chamfer_distance_mean_avg": None,
            "chamfer_distance_mean_median": None,
            "chamfer_distance_median_avg": None,
            "chamfer_distance_median_median": None,
            "earth_movers_distance_mean": None,
            "earth_movers_distance_median": None,
            "mesh_accuracy_90_mean": None,
            "mesh_accuracy_90_median": None,
            "evaluation_time_seconds": elapsed,
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Shapes evaluated: {aggregate_metrics['num_shapes_evaluated']}/{aggregate_metrics['num_shapes_total']}")
    
    if aggregate_metrics['num_shapes_evaluated'] > 0:
        print(f"\nChamfer Distance:")
        print(f"  Mean:   {aggregate_metrics['chamfer_distance_mean_avg']:.6f} (avg), "
              f"{aggregate_metrics['chamfer_distance_mean_median']:.6f} (median)")
        print(f"  Median: {aggregate_metrics['chamfer_distance_median_avg']:.6f} (avg), "
              f"{aggregate_metrics['chamfer_distance_median_median']:.6f} (median)")
        print(f"\nEarth Mover's Distance:")
        print(f"  Mean:   {aggregate_metrics['earth_movers_distance_mean']:.6f}")
        print(f"  Median: {aggregate_metrics['earth_movers_distance_median']:.6f}")
        print(f"\nMesh Accuracy @ 90%:")
        print(f"  Mean:   {aggregate_metrics['mesh_accuracy_90_mean']:.6f}")
        print(f"  Median: {aggregate_metrics['mesh_accuracy_90_median']:.6f}")
    else:
        print(f"\nNo metrics available (no shapes were successfully evaluated)")
    print("=" * 60)
    
    # Warn if no shapes were evaluated
    if len(results) == 0:
        print("\n⚠️  WARNING: No shapes were successfully evaluated!")
        print("   This may indicate:")
        print("   1. The model wasn't trained long enough (try more epochs)")
        print("   2. The model failed to learn meaningful SDF representations")
        print("   3. There's a mismatch between training and evaluation data")
        print("   Suggestion: Check training logs and try training for more epochs.")
        print("=" * 60)
    
    # Save detailed results if requested
    if output_path:
        output = {
            "aggregate_metrics": aggregate_metrics,
            "per_shape_results": results,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")
    
    return aggregate_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSDF model")
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cuda/cpu)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEEPSDF_EVALUATION["resolution"],
        help="Grid resolution for marching cubes"
    )
    parser.add_argument(
        "--num-sample-points",
        type=int,
        default=DEEPSDF_EVALUATION["num_sample_points"],
        help="Number of points to sample from meshes"
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=DEEPSDF_EVALUATION["max_shapes"],
        help="Maximum number of shapes to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEEPSDF_EVALUATION["output_file"],
        help="Path to save detailed results"
    )
    
    args = parser.parse_args()
    
    evaluate_dataset(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        gt_data_root=args.gt_data_root,
        device=args.device,
        resolution=args.resolution,
        num_sample_points=args.num_sample_points,
        max_shapes=args.max_shapes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
