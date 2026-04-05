"""Generative evaluation of the AR Transformer prior.

Computes the standard 3D generative evaluation metrics used in the literature
(Achlioptas et al., "Learning Representations and Generative Models for 3D
Point Clouds", ICML 2018) comparing a set of *generated* shapes to a
*reference* set of ground-truth shapes:

    - **MMD-CD**  : Minimum Matching Distance using Chamfer Distance.
                    For each reference shape r, find the nearest generated
                    shape g*.  MMD = mean_{r} CD(r, g*).
                    Lower is better; measures generation fidelity.

    - **MMD-EMD** : Same as MMD-CD but with Earth Mover's Distance.

    - **COV-CD**  : Coverage.  Fraction of reference shapes that are the
                    nearest neighbour of at least one generated shape.
                    Higher is better; measures generation diversity.

    - **COV-EMD** : Same as COV-CD but with Earth Mover's Distance.

Usage
-----
# Evaluate with defaults from config.py:
python -m src.deepsdf_vq.evaluate_prior

# Custom paths / sample count:
python -m src.deepsdf_vq.evaluate_prior \\
    --prior-checkpoint out/deepsdf_vq_prior/prior_latest.pth \\
    --vq-checkpoint    out/deepsdf_vq/deepsdf_vq_latest.pth  \\
    --gt-data-root     data/shapenet_v2_subset               \\
    --n-samples        50                                     \\
    --temperature      1.0                                    \\
    --output-file      out/deepsdf_vq_prior/prior_evaluation_results.json
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
    load_ground_truth_mesh,
    sample_points_from_mesh,
    sdf_to_mesh,
)
from src.deepsdf_vq.evaluate import load_vq_checkpoint, _resolve_checkpoint_path
from src.deepsdf_vq.prior import ARTransformerPrior
from src.deepsdf_vq.quantizer import GroupedVectorQuantizer
from config import DEEPSDF_VQ_MODEL, DEEPSDF_VQ_PRIOR_EVALUATION, SHAPENET_CATEGORIES


# ---------------------------------------------------------------------------
# PNG rendering helper
# ---------------------------------------------------------------------------

def _save_mesh_png(mesh, index: int, output_dir: Path) -> None:
    """Render a trimesh.Trimesh from three orthographic views and save as PNG."""
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    lim = float(np.abs(verts).max()) if len(verts) else 1.0

    fig = plt.figure(figsize=(15, 5))
    view_angles = [(30, 45), (30, 135), (90, 0)]
    view_labels = ["Front", "Side", "Top"]

    for col, (elev, azim) in enumerate(view_angles, start=1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces, color="steelblue", alpha=0.7,
            edgecolor="none", linewidth=0,
        )
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(view_labels[col - 1], fontsize=10)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(f"Generated shape {index:04d}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / f"generated_{index:04d}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PNG: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prior(checkpoint_path: str, device: str) -> ARTransformerPrior:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    prior = ARTransformerPrior(
        num_codebooks=cfg["num_codebooks"],
        codebook_size=cfg["codebook_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_multiplier=cfg["ffn_multiplier"],
        dropout=cfg["dropout"],
    ).to(device)
    prior.load_state_dict(ckpt["prior_state"])
    prior.eval()
    return prior


def _load_quantizer(vq_checkpoint_path: str, meta: Dict, device: str) -> GroupedVectorQuantizer:
    resolved = _resolve_checkpoint_path(vq_checkpoint_path)
    raw_ckpt = torch.load(resolved, map_location=device, weights_only=False)
    num_codebooks = int(meta.get("num_codebooks", DEEPSDF_VQ_MODEL["num_codebooks"]))
    codebook_size = int(meta.get("codebook_size", DEEPSDF_VQ_MODEL["codebook_size"]))
    code_dim = int(meta.get("code_dim", DEEPSDF_VQ_MODEL["code_dim"]))
    quantizer = GroupedVectorQuantizer(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        code_dim=code_dim,
        init_scale=DEEPSDF_VQ_MODEL["codebook_init_scale"],
    ).to(device)
    quantizer.load_state_dict(raw_ckpt["quantizer_state"])
    quantizer.eval()
    return quantizer


def _collect_gt_mesh_paths(gt_data_root: str) -> List[Path]:
    """Return paths to all model_normalized.obj files under gt_data_root."""
    root = Path(gt_data_root)
    paths = []
    for category_id in SHAPENET_CATEGORIES:
        for obj_path in sorted((root / category_id).glob("*/models/model_normalized.obj")):
            paths.append(obj_path)
    return paths


def _sample_generated_pointclouds(
    prior: ARTransformerPrior,
    quantizer: GroupedVectorQuantizer,
    decoder,
    n_samples: int,
    temperature: float,
    resolution: int,
    num_sample_points: int,
    batch_size: int,
    device: str,
    png_output_dir: Optional[Path] = None,
) -> List[Optional[np.ndarray]]:
    """Generate n_samples shapes, save PNGs if png_output_dir is set, and return surface point clouds."""
    print(f"Sampling {n_samples} shapes from prior (temperature={temperature})...")
    sampled_indices = prior.sample(n_samples=n_samples, temperature=temperature, device=device)

    with torch.no_grad():
        latents = quantizer.decode_indices(sampled_indices)  # [n_samples, latent_size]

    if png_output_dir is not None:
        png_output_dir.mkdir(parents=True, exist_ok=True)

    pointclouds = []
    for i in range(n_samples):
        z = latents[i : i + 1]
        mesh = sdf_to_mesh(
            decoder, z, resolution=resolution, batch_size=batch_size, device=device
        )
        if mesh is None or len(mesh.vertices) == 0:
            print(f"  Shape {i}: mesh extraction failed, skipping")
            pointclouds.append(None)
            continue
        if png_output_dir is not None:
            _save_mesh_png(mesh, i, png_output_dir)
        pts = sample_points_from_mesh(mesh, num_sample_points)
        pointclouds.append(pts)
        print(f"  Shape {i}: {len(pts)} surface points")

    return pointclouds


def _load_reference_pointclouds(
    gt_mesh_paths: List[Path],
    num_sample_points: int,
) -> Tuple[List[np.ndarray], List[Path]]:
    """Load and surface-sample all ground-truth meshes."""
    print(f"Loading {len(gt_mesh_paths)} reference meshes...")
    pointclouds = []
    valid_paths = []
    for p in gt_mesh_paths:
        try:
            import trimesh
            mesh = trimesh.load(str(p), force="mesh", process=False)
            if mesh is None or len(mesh.vertices) == 0:
                print(f"  Skipping empty mesh: {p}")
                continue
            pts = sample_points_from_mesh(mesh, num_sample_points)
            pointclouds.append(pts)
            valid_paths.append(p)
        except Exception as e:
            print(f"  Failed to load {p}: {e}")
    print(f"Loaded {len(pointclouds)} reference shapes")
    return pointclouds, valid_paths


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_mmd_cov(
    gen_clouds: List[np.ndarray],
    ref_clouds: List[np.ndarray],
    dist_fn,
) -> Dict[str, float]:
    """Compute MMD and Coverage given a pairwise distance function.

    Parameters
    ----------
    gen_clouds : list of (N, 3) arrays — generated shapes (valid only).
    ref_clouds : list of (M, 3) arrays — reference shapes.
    dist_fn    : callable(pts_a, pts_b) -> float scalar distance.

    Returns
    -------
    dict with keys: mmd, coverage.
    """
    n_gen = len(gen_clouds)
    n_ref = len(ref_clouds)

    if n_gen == 0 or n_ref == 0:
        return {"mmd": float("nan"), "coverage": float("nan")}

    # Build full pairwise distance matrix D[i, j] = dist(ref_i, gen_j).
    print(f"  Computing {n_ref} x {n_gen} pairwise distances...")
    D = np.full((n_ref, n_gen), fill_value=np.inf)
    for i, ref_pts in enumerate(ref_clouds):
        for j, gen_pts in enumerate(gen_clouds):
            try:
                d, _ = dist_fn(ref_pts, gen_pts)  # chamfer_distance returns (mean, median)
                D[i, j] = d
            except Exception as e:
                print(f"    dist({i},{j}) failed: {e}")

    # MMD: for each reference shape, find the distance to its nearest generated shape.
    mmd = float(np.mean(np.min(D, axis=1)))

    # Coverage: fraction of reference shapes that are the nearest neighbour of ≥1 generated shape.
    nearest_ref_for_gen = np.argmin(D, axis=0)  # [n_gen] — which ref each gen matches best
    matched_refs = set(nearest_ref_for_gen.tolist())
    coverage = float(len(matched_refs) / n_ref)

    return {"mmd": mmd, "coverage": coverage}


def compute_mmd_cov_emd(
    gen_clouds: List[np.ndarray],
    ref_clouds: List[np.ndarray],
) -> Dict[str, float]:
    """Same as compute_mmd_cov but using EMD (scalar, no median return)."""
    n_gen = len(gen_clouds)
    n_ref = len(ref_clouds)

    if n_gen == 0 or n_ref == 0:
        return {"mmd": float("nan"), "coverage": float("nan")}

    print(f"  Computing {n_ref} x {n_gen} pairwise EMD distances...")
    D = np.full((n_ref, n_gen), fill_value=np.inf)
    for i, ref_pts in enumerate(ref_clouds):
        for j, gen_pts in enumerate(gen_clouds):
            try:
                D[i, j] = earth_movers_distance(ref_pts, gen_pts)
            except Exception as e:
                print(f"    EMD({i},{j}) failed: {e}")

    mmd = float(np.mean(np.min(D, axis=1)))
    nearest_ref_for_gen = np.argmin(D, axis=0)
    matched_refs = set(nearest_ref_for_gen.tolist())
    coverage = float(len(matched_refs) / n_ref)

    return {"mmd": mmd, "coverage": coverage}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_prior(
    prior_checkpoint: str = str(DEEPSDF_VQ_PRIOR_EVALUATION["prior_checkpoint"]),
    vq_checkpoint: str = str(DEEPSDF_VQ_PRIOR_EVALUATION["vq_checkpoint"]),
    gt_data_root: str = str(DEEPSDF_VQ_PRIOR_EVALUATION["gt_data_root"]),
    n_samples: int = DEEPSDF_VQ_PRIOR_EVALUATION["n_samples"],
    temperature: float = DEEPSDF_VQ_PRIOR_EVALUATION["temperature"],
    resolution: int = DEEPSDF_VQ_PRIOR_EVALUATION["resolution"],
    num_sample_points: int = DEEPSDF_VQ_PRIOR_EVALUATION["num_sample_points"],
    batch_size: int = DEEPSDF_VQ_PRIOR_EVALUATION["batch_size"],
    output_file: Optional[str] = str(DEEPSDF_VQ_PRIOR_EVALUATION["output_file"]),
    save_png: bool = True,
    device: Optional[str] = None,
) -> Dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Prior checkpoint : {prior_checkpoint}")
    print(f"VQ checkpoint    : {vq_checkpoint}")
    print(f"GT data root     : {gt_data_root}")
    print(f"n_samples        : {n_samples}")
    print(f"temperature      : {temperature}")

    t0 = time.time()

    # ---- Load models -------------------------------------------------------
    decoder, _, meta = load_vq_checkpoint(vq_checkpoint, device=device)
    quantizer = _load_quantizer(vq_checkpoint, meta, device)
    prior = _load_prior(prior_checkpoint, device)

    # ---- Generate shapes ---------------------------------------------------
    png_dir = Path(output_file).parent / "generated_shapes" if (save_png and output_file) else None
    gen_clouds = _sample_generated_pointclouds(
        prior=prior,
        quantizer=quantizer,
        decoder=decoder,
        n_samples=n_samples,
        temperature=temperature,
        resolution=resolution,
        num_sample_points=num_sample_points,
        batch_size=batch_size,
        device=device,
        png_output_dir=png_dir,
    )
    valid_gen = [pc for pc in gen_clouds if pc is not None]
    print(f"\n{len(valid_gen)}/{n_samples} generated shapes produced valid meshes")

    # ---- Load reference shapes ---------------------------------------------
    gt_paths = _collect_gt_mesh_paths(gt_data_root)
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
        "temperature": temperature,
        "elapsed_seconds": elapsed,
    }

    print("\n===== Prior Evaluation Results =====")
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
        description="Evaluate AR prior via MMD-CD, COV-CD, MMD-EMD, COV-EMD"
    )
    parser.add_argument(
        "--prior-checkpoint",
        type=str,
        default=str(DEEPSDF_VQ_PRIOR_EVALUATION["prior_checkpoint"]),
    )
    parser.add_argument(
        "--vq-checkpoint",
        type=str,
        default=str(DEEPSDF_VQ_PRIOR_EVALUATION["vq_checkpoint"]),
    )
    parser.add_argument(
        "--gt-data-root",
        type=str,
        default=str(DEEPSDF_VQ_PRIOR_EVALUATION["gt_data_root"]),
    )
    parser.add_argument("--n-samples", type=int, default=DEEPSDF_VQ_PRIOR_EVALUATION["n_samples"])
    parser.add_argument("--temperature", type=float, default=DEEPSDF_VQ_PRIOR_EVALUATION["temperature"])
    parser.add_argument("--resolution", type=int, default=DEEPSDF_VQ_PRIOR_EVALUATION["resolution"])
    parser.add_argument("--num-sample-points", type=int, default=DEEPSDF_VQ_PRIOR_EVALUATION["num_sample_points"])
    parser.add_argument("--batch-size", type=int, default=DEEPSDF_VQ_PRIOR_EVALUATION["batch_size"])
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEEPSDF_VQ_PRIOR_EVALUATION["output_file"]),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-png", dest="save_png", action="store_false", default=True,
        help="Disable PNG rendering of generated shapes",
    )

    args = parser.parse_args()

    evaluate_prior(
        prior_checkpoint=args.prior_checkpoint,
        vq_checkpoint=args.vq_checkpoint,
        gt_data_root=args.gt_data_root,
        n_samples=args.n_samples,
        temperature=args.temperature,
        resolution=args.resolution,
        num_sample_points=args.num_sample_points,
        batch_size=args.batch_size,
        output_file=args.output_file,
        save_png=args.save_png,
        device=args.device,
    )


if __name__ == "__main__":
    main()
