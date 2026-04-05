"""Training script for the AR Transformer prior over VQ codebook sequences.

The auto-decoder and VQ codebooks are strictly frozen.  Only the prior
parameters are updated, using a standard cross-entropy next-token loss.

Usage
-----
# Train with defaults (loads latest VQ checkpoint):
python -m src.deepsdf_vq.train_prior

# Override checkpoint / output directory:
python -m src.deepsdf_vq.train_prior \
    --vq-checkpoint out/deepsdf_vq/deepsdf_vq_latest.pth \
    --save-dir out/deepsdf_vq_prior \
    --epochs 2000
"""

import argparse
import glob
import json
import re
import time
from pathlib import Path

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import load_vq_checkpoint
from .prior import ARTransformerPrior
from src.deepsdf.logging_utils import setup_training_logger
from src.deepsdf.train import _build_lr_schedules
from config import DEEPSDF_VQ_MODEL, DEEPSDF_VQ_PRIOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_indices(vq_checkpoint_path: str, device: str) -> torch.Tensor:
    """Load the VQ checkpoint and return the shape code-index table [N, G]."""
    ckpt_path = Path(vq_checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "shape_code_indices" not in ckpt:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain 'shape_code_indices'. "
            "Re-run VQ-DeepSDF training to generate them."
        )
    indices = ckpt["shape_code_indices"]
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)
    return indices.to(device=device, dtype=torch.long)


def _resolve_vq_checkpoint(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    candidates = sorted(path.parent.glob("deepsdf_vq_epoch_*.pth"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No VQ checkpoint found at or near: {path_str}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_ar_prior(
    vq_checkpoint: str = str(DEEPSDF_VQ_PRIOR["vq_checkpoint"]),
    save_dir: str = str(DEEPSDF_VQ_PRIOR["save_dir"]),
    d_model: int = DEEPSDF_VQ_PRIOR["d_model"],
    n_heads: int = DEEPSDF_VQ_PRIOR["n_heads"],
    n_layers: int = DEEPSDF_VQ_PRIOR["n_layers"],
    ffn_multiplier: int = DEEPSDF_VQ_PRIOR["ffn_multiplier"],
    dropout: float = DEEPSDF_VQ_PRIOR["dropout"],
    lr: float = DEEPSDF_VQ_PRIOR["lr"],
    epochs: int = DEEPSDF_VQ_PRIOR["epochs"],
    batch_size: int = DEEPSDF_VQ_PRIOR["batch_size"],
    random_seed: int = DEEPSDF_VQ_PRIOR["random_seed"],
    grad_clip_norm: float = DEEPSDF_VQ_PRIOR["grad_clip_norm"],
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    logger, log_file = setup_training_logger(save_path)

    # ------------------------------------------------------------------
    # Load frozen VQ checkpoint and extract code-index sequences.
    # ------------------------------------------------------------------
    resolved_vq = _resolve_vq_checkpoint(vq_checkpoint)
    logger.info("Loading VQ checkpoint: %s", resolved_vq)

    # Load meta to discover architecture dims.
    meta_path = resolved_vq.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    num_codebooks = int(meta.get("num_codebooks", DEEPSDF_VQ_MODEL["num_codebooks"]))
    codebook_size = int(meta.get("codebook_size", DEEPSDF_VQ_MODEL["codebook_size"]))

    all_indices = _extract_indices(str(resolved_vq), device)  # [N, G]
    num_shapes, G = all_indices.shape
    assert G == num_codebooks, (
        f"Index table has G={G} columns but config says num_codebooks={num_codebooks}"
    )
    logger.info(
        "Loaded %d shape sequences | num_codebooks=%d | codebook_size=%d",
        num_shapes, num_codebooks, codebook_size,
    )

    # ------------------------------------------------------------------
    # Build AR prior (the only trainable component).
    # ------------------------------------------------------------------
    prior = ARTransformerPrior(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_multiplier=ffn_multiplier,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in prior.parameters() if p.requires_grad)
    logger.info("Prior parameters: %d", num_params)

    # ------------------------------------------------------------------
    # Optimiser and LR schedule.
    # ------------------------------------------------------------------
    schedule_specs = DEEPSDF_VQ_PRIOR["lr_schedules"]
    lr_schedules = _build_lr_schedules(lr, schedule_specs)

    optimizer = torch.optim.Adam(prior.parameters(), lr=lr_schedules[0].get_learning_rate(0))

    # ------------------------------------------------------------------
    # Resume from existing prior checkpoint if available.
    # ------------------------------------------------------------------
    loss_log: list = []
    lr_log: list = []
    timing_log: list = []
    start_epoch = 1

    pattern = str(save_path / "prior_epoch_*.pth")
    existing = glob.glob(pattern)
    if existing:
        nums = []
        for f in existing:
            m = re.search(r"prior_epoch_(\d+)\.pth", f)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            latest_n = max(nums)
            latest_ckpt = save_path / f"prior_epoch_{latest_n}.pth"
            logger.info("Resuming from prior checkpoint: %s", latest_ckpt)
            try:
                ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
                prior.load_state_dict(ckpt["prior_state"])
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt["epoch"] + 1
                loss_log = ckpt.get("loss_log", [])
                lr_log = ckpt.get("lr_log", [])
                timing_log = ckpt.get("timing_log", [])
                logger.info("Resumed prior training from epoch %d", start_epoch)
            except Exception as exc:
                logger.warning("Failed to load prior checkpoint: %s. Starting fresh.", exc)
                start_epoch = 1

    log_frequency = DEEPSDF_VQ_PRIOR["log_frequency"]
    snapshot_frequency = DEEPSDF_VQ_PRIOR["snapshot_frequency"]

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    logger.info("Starting AR prior training")
    logger.info("Device: %s", device)
    logger.info("Save dir: %s", save_path)
    logger.info("VQ checkpoint: %s", resolved_vq)

    for epoch in range(start_epoch, epochs + 1):
        prior.train()
        t0 = time.time()

        # Update LR.
        new_lr = lr_schedules[0].get_learning_rate(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        perm = torch.randperm(num_shapes, device=device)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_shapes, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch = all_indices[batch_idx]          # [B, G]

            logits = prior(batch)                   # [B, G, K]
            # Target at position g is the true index at position g (the "next token").
            loss = F.cross_entropy(
                logits.reshape(-1, codebook_size),  # [B*G, K]
                batch.reshape(-1),                  # [B*G]
            )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(prior.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

            if (num_batches % log_frequency) == 0:
                logger.info(
                    "Epoch %d | batch %d | loss=%.6f",
                    epoch, num_batches, epoch_loss / num_batches,
                )

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, num_batches)
        logger.info(
            "Epoch %d completed | duration=%.1fs | loss=%.6f | lr=%.2e",
            epoch, epoch_time, avg_loss, new_lr,
        )

        loss_log.append({"loss": avg_loss})
        lr_log.append(new_lr)
        timing_log.append(epoch_time)

        ckpt = {
            "prior_state": prior.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss_log": loss_log,
            "lr_log": lr_log,
            "timing_log": timing_log,
            "config": {
                "num_codebooks": num_codebooks,
                "codebook_size": codebook_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "ffn_multiplier": ffn_multiplier,
                "dropout": dropout,
            },
        }

        if epoch % snapshot_frequency == 0:
            torch.save(ckpt, save_path / f"prior_epoch_{epoch}.pth")
            logger.info("Saved snapshot: prior_epoch_%d.pth", epoch)

        torch.save(ckpt, save_path / "prior_latest.pth")

    logger.info("AR prior training complete.")


# ---------------------------------------------------------------------------
# Generation helper (kept here for easy CLI use)
# ---------------------------------------------------------------------------

def generate_novel_shapes(
    prior_checkpoint: str,
    vq_checkpoint: str,
    n_samples: int = 4,
    temperature: float = 1.0,
    resolution: int = 128,
    batch_size: int = 32768,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> list:
    """Sample novel shapes using the trained AR prior + frozen VQ decoder.

    Returns a list of trimesh.Trimesh objects (one per sample).
    """
    import trimesh
    from src.deepsdf.evaluate import sdf_to_mesh

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prior.
    ckpt_p = torch.load(prior_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt_p["config"]
    prior = ARTransformerPrior(
        num_codebooks=cfg["num_codebooks"],
        codebook_size=cfg["codebook_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_multiplier=cfg["ffn_multiplier"],
        dropout=cfg["dropout"],
    ).to(device)
    prior.load_state_dict(ckpt_p["prior_state"])
    prior.eval()

    # Load frozen VQ decoder + quantizer.
    decoder, _, meta = load_vq_checkpoint(vq_checkpoint, device=device)

    from .evaluate import _resolve_checkpoint_path
    from .quantizer import GroupedVectorQuantizer

    resolved = _resolve_checkpoint_path(vq_checkpoint)
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

    # Sample index sequences from the prior.
    sampled_indices = prior.sample(n_samples=n_samples, temperature=temperature, device=device)
    # Decode to continuous latent vectors.
    with torch.no_grad():
        latents = quantizer.decode_indices(sampled_indices)  # [n_samples, latent_size]

    # Reconstruct SDF meshes via marching cubes.
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    meshes = []
    for i in range(n_samples):
        z = latents[i : i + 1]  # [1, latent_size]
        try:
            mesh = sdf_to_mesh(decoder, z, resolution=resolution, batch_size=batch_size, device=device)
            meshes.append(mesh)
            if output_dir is not None:
                out_path = Path(output_dir) / f"generated_{i:04d}.obj"
                mesh.export(str(out_path))
        except Exception as exc:
            print(f"[generate] shape {i} failed: {exc}")
            meshes.append(None)

    return meshes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train AR Transformer prior over VQ codebook sequences")
    parser.add_argument(
        "--vq-checkpoint",
        type=str,
        default=str(DEEPSDF_VQ_PRIOR["vq_checkpoint"]),
        help="Path to the frozen VQ-DeepSDF checkpoint (.pth)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(DEEPSDF_VQ_PRIOR["save_dir"]),
        help="Directory to save prior checkpoints and logs",
    )
    parser.add_argument("--d-model", type=int, default=DEEPSDF_VQ_PRIOR["d_model"])
    parser.add_argument("--n-heads", type=int, default=DEEPSDF_VQ_PRIOR["n_heads"])
    parser.add_argument("--n-layers", type=int, default=DEEPSDF_VQ_PRIOR["n_layers"])
    parser.add_argument("--ffn-multiplier", type=int, default=DEEPSDF_VQ_PRIOR["ffn_multiplier"])
    parser.add_argument("--dropout", type=float, default=DEEPSDF_VQ_PRIOR["dropout"])
    parser.add_argument("--lr", type=float, default=DEEPSDF_VQ_PRIOR["lr"])
    parser.add_argument("--epochs", type=int, default=DEEPSDF_VQ_PRIOR["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEEPSDF_VQ_PRIOR["batch_size"])
    parser.add_argument("--seed", type=int, default=DEEPSDF_VQ_PRIOR["random_seed"])
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    train_ar_prior(
        vq_checkpoint=args.vq_checkpoint,
        save_dir=args.save_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
