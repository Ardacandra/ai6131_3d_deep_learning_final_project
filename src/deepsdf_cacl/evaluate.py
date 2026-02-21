"""
Evaluation script for DeepSDF+CACL model.

Computes:
- Chamfer Distance (Mean and Median)
- Earth Mover's Distance (Mean and Median)
- Mesh accuracy (minimum distance d such that 90% of generated points are within d of ground truth mesh)
"""

import argparse
from pathlib import Path
import json
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.deepsdf.dataset import DeepSDFDataset
from src.deepsdf.model import DeepSDFDecoder
from src.deepsdf.evaluate import (
    _load_selected_shape_paths,
    evaluate_shape,
)
from config import DEEPSDF_CACL_TRAINING, DEEPSDF_CACL_EVALUATION


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.exists():
        return path

    parent = path.parent
    candidates = sorted(
        parent.glob("deepsdf_cacl_epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if candidates:
        fallback = candidates[-1]
        print(
            f"Checkpoint not found: {path}. Falling back to latest epoch checkpoint: {fallback}"
        )
        return fallback

    raise FileNotFoundError(
        f"Checkpoint not found: {path}. No fallback checkpoint matching deepsdf_cacl_epoch_*.pth in {parent}"
    )


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[DeepSDFDecoder, torch.nn.Embedding, Dict]:
    """Load trained DeepSDF+CACL model and latent codes from checkpoint."""
    resolved_checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint_path, map_location=device, weights_only=False)

    meta_path = resolved_checkpoint_path.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    latent_size = meta.get("latent_size", DEEPSDF_CACL_TRAINING["latent_size"])
    hidden_size = meta.get("hidden_size", DEEPSDF_CACL_TRAINING["hidden_size"])

    decoder = DeepSDFDecoder(latent_size=latent_size, hidden_size=hidden_size).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    decoder.eval()

    latents = checkpoint["latents"]
    if isinstance(latents, torch.Tensor):
        latents_tensor = latents.to(device)
    else:
        latents_tensor = torch.from_numpy(latents).float().to(device)

    num_shapes = latents_tensor.shape[0]
    latent_embeddings = torch.nn.Embedding(num_shapes, latent_size).to(device)
    latent_embeddings.weight.data = latents_tensor

    return decoder, latent_embeddings, meta


def evaluate_dataset(
    checkpoint_path: str,
    data_root: str,
    gt_data_root: str,
    samples_manifest_path: Optional[str] = None,
    device: str = "cpu",
    resolution: int = 128,
    num_sample_points: int = 10000,
    max_shapes: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """Evaluate DeepSDF+CACL on an SDF dataset."""
    print(f"Loading checkpoint from {checkpoint_path}")
    decoder, latent_embeddings, meta = load_checkpoint(checkpoint_path, device=device)

    print(f"Loading dataset from {data_root}")
    selected_shape_paths = _load_selected_shape_paths(
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        samples_manifest_path=samples_manifest_path,
    )

    if selected_shape_paths is not None:
        shape_paths = selected_shape_paths
        print(f"Using selected samples manifest with {len(shape_paths)} shapes")
    else:
        dataset = DeepSDFDataset(data_root)
        shape_paths = dataset.get_shape_paths()
        print(f"Selected samples manifest not found; using full dataset ({len(shape_paths)} shapes)")

    num_shapes = len(shape_paths)
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
        shape_path = shape_paths[idx]

        print(
            f"[{idx + 1}/{num_shapes}] Evaluating {shape_path.name} "
            f"from {shape_path.parent.name}/{shape_path.parent.parent.name}"
        )

        latent_code = latent_embeddings.weight[idx].unsqueeze(0)

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
            print("  Skipped due to error")
            continue

        chamfer_means.append(metrics["chamfer_distance_mean"])
        chamfer_medians.append(metrics["chamfer_distance_median"])
        emds.append(metrics["earth_movers_distance"])
        accuracies.append(metrics["mesh_accuracy_90"])

        results.append({
            "shape_index": idx,
            "shape_path": str(shape_path),
            **metrics,
        })

        print(
            f"  CD mean: {metrics['chamfer_distance_mean']:.6f}, "
            f"CD median: {metrics['chamfer_distance_median']:.6f}, "
            f"EMD: {metrics['earth_movers_distance']:.6f}, "
            f"Acc@90: {metrics['mesh_accuracy_90']:.6f}"
        )

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.2f}s")

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
            "triplet_margin": meta.get("triplet_margin"),
            "triplet_lambda": meta.get("triplet_lambda"),
        }
    else:
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
            "triplet_margin": meta.get("triplet_margin"),
            "triplet_lambda": meta.get("triplet_lambda"),
        }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (DEEPSDF + CACL)")
    print("=" * 60)
    print(
        f"Shapes evaluated: {aggregate_metrics['num_shapes_evaluated']}/"
        f"{aggregate_metrics['num_shapes_total']}"
    )

    if aggregate_metrics["num_shapes_evaluated"] > 0:
        print("\nChamfer Distance:")
        print(
            f"  Mean:   {aggregate_metrics['chamfer_distance_mean_avg']:.6f} (avg), "
            f"{aggregate_metrics['chamfer_distance_mean_median']:.6f} (median)"
        )
        print(
            f"  Median: {aggregate_metrics['chamfer_distance_median_avg']:.6f} (avg), "
            f"{aggregate_metrics['chamfer_distance_median_median']:.6f} (median)"
        )
        print("\nEarth Mover's Distance:")
        print(f"  Mean:   {aggregate_metrics['earth_movers_distance_mean']:.6f}")
        print(f"  Median: {aggregate_metrics['earth_movers_distance_median']:.6f}")
        print("\nMesh Accuracy @ 90%:")
        print(f"  Mean:   {aggregate_metrics['mesh_accuracy_90_mean']:.6f}")
        print(f"  Median: {aggregate_metrics['mesh_accuracy_90_median']:.6f}")
    else:
        print("\nNo metrics available (no shapes were successfully evaluated)")

    if aggregate_metrics.get("triplet_margin") is not None:
        print("\nCACL Metadata:")
        print(f"  triplet_margin: {aggregate_metrics['triplet_margin']}")
        print(f"  triplet_lambda: {aggregate_metrics['triplet_lambda']}")

    print("=" * 60)

    if len(results) == 0:
        print("\n⚠️  WARNING: No shapes were successfully evaluated!")
        print("   This may indicate:")
        print("   1. The model wasn't trained long enough (try more epochs)")
        print("   2. The model failed to learn meaningful SDF representations")
        print("   3. There's a mismatch between training and evaluation data")
        print("   Suggestion: Check training logs and try training for more epochs.")
        print("=" * 60)

    if output_path:
        output = {
            "aggregate_metrics": aggregate_metrics,
            "per_shape_results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")

    return aggregate_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSDF+CACL model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEEPSDF_CACL_EVALUATION["checkpoint"]),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEEPSDF_CACL_EVALUATION["data_root"]),
        help="Root directory for SDF data",
    )
    parser.add_argument(
        "--gt-data-root",
        type=str,
        default=str(DEEPSDF_CACL_EVALUATION["gt_data_root"]),
        help="Root directory for ground truth meshes",
    )
    parser.add_argument(
        "--samples-manifest",
        type=str,
        default=None,
        help="Path to selected_samples.json. Defaults to checkpoint directory if present",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cuda/cpu)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEEPSDF_CACL_EVALUATION["resolution"],
        help="Grid resolution for marching cubes",
    )
    parser.add_argument(
        "--num-sample-points",
        type=int,
        default=DEEPSDF_CACL_EVALUATION["num_sample_points"],
        help="Number of points to sample from meshes",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=DEEPSDF_CACL_EVALUATION["max_shapes"],
        help="Maximum number of shapes to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEEPSDF_CACL_EVALUATION["output_file"],
        help="Path to save detailed results",
    )

    args = parser.parse_args()

    evaluate_dataset(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        gt_data_root=args.gt_data_root,
        samples_manifest_path=args.samples_manifest,
        device=args.device,
        resolution=args.resolution,
        num_sample_points=args.num_sample_points,
        max_shapes=args.max_shapes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
