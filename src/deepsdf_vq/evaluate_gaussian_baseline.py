"""Gaussian-noise baseline for generative evaluation.

Instead of sampling from a learned AR Transformer prior, this script draws
random latent vectors from N(0, std²·I) and decodes them with the frozen
VQ-DeepSDF decoder.  All downstream evaluation (mesh extraction, point-cloud
sampling, MMD-CD, COV-CD, MMD-EMD, COV-EMD) is identical to evaluate_prior.py
so results are directly comparable.

Usage
-----
# Evaluate with defaults from config.py:
python -m src.deepsdf_vq.evaluate_gaussian_baseline

# Custom paths / sample count:
python -m src.deepsdf_vq.evaluate_gaussian_baseline \\
    --vq-checkpoint out/deepsdf_vq/deepsdf_vq_latest.pth  \\
    --gt-data-root  data/shapenet_v2_subset               \\
    --n-samples     50                                    \\
    --std           1.0                                   \\
    --output-file   out/deepsdf_vq_gaussian_baseline/gaussian_baseline_evaluation_results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.deepsdf.evaluate import (
    chamfer_distance,
    earth_movers_distance,
    sample_points_from_mesh,
    sdf_to_mesh,
)
from src.deepsdf_vq.evaluate import load_vq_checkpoint, _resolve_checkpoint_path
from src.deepsdf_vq.evaluate_prior import (
    _save_pointcloud_png,
    _save_mesh_png,
    _collect_gt_mesh_paths,
    _load_reference_pointclouds,
    compute_mmd_cov,
    compute_mmd_cov_emd,
)
from config import DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION, DEEPSDF_VQ_MODEL, SHAPENET_CATEGORIES


# ---------------------------------------------------------------------------
# Gaussian sampling
# ---------------------------------------------------------------------------

def _sample_gaussian_pointclouds(
    decoder,
    latent_size: int,
    n_samples: int,
    std: float,
    resolution: int,
    num_sample_points: int,
    batch_size: int,
    device: str,
    png_output_dir: Optional[Path] = None,
    max_attempts_multiplier: int = 20,
) -> List[np.ndarray]:
    """Sample latents from N(0, std²·I) and retry until n_samples valid meshes are found.

    Parameters
    ----------
    max_attempts_multiplier : int
        Maximum total attempts = n_samples * max_attempts_multiplier.
        Prevents infinite loops when virtually no latent produces a valid mesh.
    """
    max_attempts = n_samples * max_attempts_multiplier
    print(
        f"Sampling {n_samples} shapes from N(0, {std}²·I) "
        f"(latent_size={latent_size}, max_attempts={max_attempts})..."
    )

    if png_output_dir is not None:
        png_output_dir.mkdir(parents=True, exist_ok=True)

    pointclouds: List[np.ndarray] = []
    attempts = 0
    while len(pointclouds) < n_samples and attempts < max_attempts:
        attempts += 1
        z = torch.randn(1, latent_size, device=device) * std
        mesh = sdf_to_mesh(
            decoder, z, resolution=resolution, batch_size=batch_size, device=device
        )
        if mesh is None or len(mesh.vertices) == 0:
            print(f"  Attempt {attempts}: mesh extraction failed, retrying...")
            continue
        pts = sample_points_from_mesh(mesh, num_sample_points)
        i = len(pointclouds)
        if png_output_dir is not None:
            _save_mesh_png(mesh, i, png_output_dir)
            _save_pointcloud_png(pts, i, png_output_dir)
            obj_path = png_output_dir / f"generated_{i:04d}.obj"
            mesh.export(str(obj_path))
            print(f"  Saved mesh OBJ: {obj_path}")
        pointclouds.append(pts)
        print(f"  Shape {i} (attempt {attempts}): {len(pts)} surface points")

    if len(pointclouds) < n_samples:
        print(
            f"  Warning: only {len(pointclouds)}/{n_samples} valid meshes found "
            f"after {attempts} attempts. Consider reducing --std."
        )

    return pointclouds


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_gaussian_baseline(
    vq_checkpoint: str = str(DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["vq_checkpoint"]),
    gt_data_root: str = str(DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["gt_data_root"]),
    n_samples: int = DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["n_samples"],
    std: float = DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["std"],
    resolution: int = DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["resolution"],
    num_sample_points: int = DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["num_sample_points"],
    batch_size: int = DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["batch_size"],
    output_file: Optional[str] = str(DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["output_file"]),
    save_png: bool = True,
    device: Optional[str] = None,
    max_attempts_multiplier: int = 20,
) -> Dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device          : {device}")
    print(f"VQ checkpoint   : {vq_checkpoint}")
    print(f"GT data root    : {gt_data_root}")
    print(f"n_samples       : {n_samples}")
    print(f"std             : {std}")

    t0 = time.time()

    # ---- Load decoder ------------------------------------------------------
    decoder, _, meta = load_vq_checkpoint(vq_checkpoint, device=device)

    num_codebooks = int(meta.get("num_codebooks", DEEPSDF_VQ_MODEL["num_codebooks"]))
    code_dim = int(meta.get("code_dim", DEEPSDF_VQ_MODEL["code_dim"]))
    latent_size = num_codebooks * code_dim

    # ---- Generate shapes via Gaussian sampling -----------------------------
    png_dir = Path(output_file).parent / "generated_shapes" if (save_png and output_file) else None
    valid_gen = _sample_gaussian_pointclouds(
        decoder=decoder,
        latent_size=latent_size,
        n_samples=n_samples,
        std=std,
        resolution=resolution,
        num_sample_points=num_sample_points,
        batch_size=batch_size,
        device=device,
        png_output_dir=png_dir,
        max_attempts_multiplier=max_attempts_multiplier,
    )
    print(f"\n{len(valid_gen)}/{n_samples} generated shapes produced valid meshes")

    # ---- Load reference shapes ---------------------------------------------
    gt_paths = _collect_gt_mesh_paths(gt_data_root, vq_checkpoint)
    if not gt_paths:
        raise RuntimeError(f"No ground-truth meshes found under {gt_data_root}")

    ref_clouds, _ = _load_reference_pointclouds(gt_paths, num_sample_points)
    if not ref_clouds:
        raise RuntimeError("Failed to load any reference meshes")

    # ---- Compute MMD + Coverage (CD) ---------------------------------------
    print("\nComputing MMD-CD and COV-CD...")
    cd_metrics = compute_mmd_cov(valid_gen, ref_clouds, chamfer_distance)

    # ---- Compute MMD + Coverage (EMD) --------------------------------------
    print("\nComputing MMD-EMD and COV-EMD...")
    emd_metrics = compute_mmd_cov_emd(valid_gen, ref_clouds)

    elapsed = time.time() - t0

    results = {
        "mmd_cd": cd_metrics["mmd"],
        "coverage_cd": cd_metrics["coverage"],
        "mmd_emd": emd_metrics["mmd"],
        "coverage_emd": emd_metrics["coverage"],
        "n_samples_requested": n_samples,
        "n_samples_valid": len(valid_gen),
        "n_reference_shapes": len(ref_clouds),
        "std": std,
        "elapsed_seconds": elapsed,
    }

    print("\n===== Gaussian Baseline Evaluation Results =====")
    print(f"  MMD-CD   : {results['mmd_cd']:.6f}")
    print(f"  COV-CD   : {results['coverage_cd']:.4f}  ({results['coverage_cd']*100:.1f}%)")
    print(f"  MMD-EMD  : {results['mmd_emd']:.6f}")
    print(f"  COV-EMD  : {results['coverage_emd']:.4f}  ({results['coverage_emd']*100:.1f}%)")
    print(f"  Valid shapes : {results['n_samples_valid']} / {results['n_samples_requested']}")
    print(f"  Elapsed      : {elapsed:.1f}s")

    if output_file is not None:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
    if png_dir is not None:
        print(f"PNGs saved to    {png_dir}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gaussian-noise baseline: MMD-CD, COV-CD, MMD-EMD, COV-EMD"
    )
    parser.add_argument(
        "--vq-checkpoint",
        type=str,
        default=str(DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["vq_checkpoint"]),
    )
    parser.add_argument(
        "--gt-data-root",
        type=str,
        default=str(DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["gt_data_root"]),
    )
    parser.add_argument("--n-samples", type=int, default=DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["n_samples"])
    parser.add_argument(
        "--std", type=float, default=DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["std"],
        help="Standard deviation of the Gaussian latent samples (default: 1.0)",
    )
    parser.add_argument("--resolution", type=int, default=DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["resolution"])
    parser.add_argument("--num-sample-points", type=int, default=DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["num_sample_points"])
    parser.add_argument("--batch-size", type=int, default=DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["batch_size"])
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION["output_file"]),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-attempts-multiplier", type=int, default=20,
        help="Max retry attempts = n_samples * this value (default: 20)",
    )
    parser.add_argument(
        "--no-png", dest="save_png", action="store_false", default=True,
        help="Disable PNG rendering of generated shapes",
    )

    args = parser.parse_args()

    evaluate_gaussian_baseline(
        vq_checkpoint=args.vq_checkpoint,
        gt_data_root=args.gt_data_root,
        n_samples=args.n_samples,
        std=args.std,
        resolution=args.resolution,
        num_sample_points=args.num_sample_points,
        batch_size=args.batch_size,
        output_file=args.output_file,
        save_png=args.save_png,
        device=args.device,
        max_attempts_multiplier=args.max_attempts_multiplier,
    )


if __name__ == "__main__":
    main()
