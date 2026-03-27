import argparse
from collections import defaultdict
from pathlib import Path
import glob
import json
import math
import re
import time

import numpy as np
import torch

from .model import DeepSDFDecoder
from .quantizer import GroupedVectorQuantizer
from src.deepsdf.dataset import DeepSDFDataset
from src.deepsdf.logging_utils import setup_training_logger
from src.deepsdf.train import _build_lr_schedules, _sample_shape_points
from config import DEEPSDF_VQ_MODEL, DEEPSDF_VQ_TRAINING


def _validate_vq_dims(latent_size: int, num_codebooks: int, code_dim: int) -> None:
    if latent_size != num_codebooks * code_dim:
        raise ValueError(
            "VQ latent dimension mismatch: "
            f"latent_size={latent_size}, "
            f"num_codebooks={num_codebooks}, code_dim={code_dim}, "
            "expected latent_size == num_codebooks * code_dim"
        )


def _compute_shape_indices(shape_latents: torch.nn.Embedding, quantizer: GroupedVectorQuantizer) -> torch.Tensor:
    with torch.no_grad():
        return quantizer.encode_indices(shape_latents.weight.data)


def train_vq_autodecoder(
    data_root,
    latent_size=DEEPSDF_VQ_TRAINING["latent_size"],
    hidden_size=DEEPSDF_VQ_TRAINING["hidden_size"],
    lr=DEEPSDF_VQ_TRAINING["lr"],
    random_seed=DEEPSDF_VQ_TRAINING["random_seed"],
    epochs=DEEPSDF_VQ_TRAINING["epochs"],
    batch_points=DEEPSDF_VQ_TRAINING["batch_points"],
    device=None,
    save_dir=DEEPSDF_VQ_TRAINING["save_dir"],
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    dataset = DeepSDFDataset(data_root)
    num_shapes = len(dataset)
    if num_shapes == 0:
        raise RuntimeError(f"No shapes found under {data_root}")

    num_codebooks = DEEPSDF_VQ_MODEL["num_codebooks"]
    codebook_size = DEEPSDF_VQ_MODEL["codebook_size"]
    code_dim = DEEPSDF_VQ_MODEL["code_dim"]
    _validate_vq_dims(latent_size, num_codebooks, code_dim)

    decoder = DeepSDFDecoder(latent_size=latent_size, hidden_size=hidden_size).to(device)
    quantizer = GroupedVectorQuantizer(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        code_dim=code_dim,
        init_scale=DEEPSDF_VQ_MODEL["codebook_init_scale"],
    ).to(device)

    shape_latent_bound = DEEPSDF_VQ_TRAINING["shape_latent_bound"]
    shape_latents = torch.nn.Embedding(
        num_shapes,
        latent_size,
        max_norm=shape_latent_bound,
    ).to(device)
    torch.nn.init.normal_(
        shape_latents.weight,
        mean=0.0,
        std=DEEPSDF_VQ_TRAINING["shape_latent_init_stddev"] / math.sqrt(latent_size),
    )

    schedule_specs = DEEPSDF_VQ_TRAINING["lr_schedules"]
    lr_schedules = _build_lr_schedules(lr, schedule_specs)

    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": lr_schedules[0].get_learning_rate(0)},
            {"params": shape_latents.parameters(), "lr": lr_schedules[1].get_learning_rate(0)},
            {"params": quantizer.parameters(), "lr": lr_schedules[1].get_learning_rate(0)},
        ]
    )
    l1_loss = torch.nn.L1Loss(reduction="sum")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    logger, log_file = setup_training_logger(save_path)

    selected_shape_ids = dataset.get_shape_ids()
    selected_manifest = {
        "data_root": str(Path(data_root)),
        "num_shapes": num_shapes,
        "random_seed": random_seed,
        "shapes": selected_shape_ids,
    }
    selected_manifest_path = save_path / "selected_samples.json"
    with open(selected_manifest_path, "w") as f:
        json.dump(selected_manifest, f, indent=2)

    samples_per_scene = batch_points or DEEPSDF_VQ_TRAINING["samples_per_scene"]
    scenes_per_batch = DEEPSDF_VQ_TRAINING["scenes_per_batch"]
    batch_split = DEEPSDF_VQ_TRAINING["batch_split"]
    clamp_dist = DEEPSDF_VQ_TRAINING["clamp_dist"]
    grad_clip_norm = DEEPSDF_VQ_TRAINING["grad_clip_norm"]
    log_frequency = DEEPSDF_VQ_TRAINING["log_frequency"]
    snapshot_frequency = DEEPSDF_VQ_TRAINING["snapshot_frequency"]
    additional_snapshots = set(DEEPSDF_VQ_TRAINING["additional_snapshots"])

    commitment_weight = DEEPSDF_VQ_TRAINING["commitment_weight"]
    codebook_weight = DEEPSDF_VQ_TRAINING["codebook_weight"]
    entropy_weight = DEEPSDF_VQ_TRAINING["entropy_weight"]

    min_t = -clamp_dist
    max_t = clamp_dist

    loss_log = []
    lr_log = []
    timing_log = []
    start_epoch = 1

    checkpoint_pattern = save_path / "deepsdf_vq_epoch_*.pth"
    existing_checkpoints = glob.glob(str(checkpoint_pattern))

    if existing_checkpoints:
        epoch_numbers = []
        for ckpt_file in existing_checkpoints:
            match = re.search(r"deepsdf_vq_epoch_(\d+)\.pth", ckpt_file)
            if match:
                epoch_numbers.append(int(match.group(1)))

        if epoch_numbers:
            latest_epoch = max(epoch_numbers)
            latest_ckpt = save_path / f"deepsdf_vq_epoch_{latest_epoch}.pth"
            logger.info("Found existing checkpoint: %s", latest_ckpt)

            try:
                ckpt = torch.load(latest_ckpt, map_location=device)
                decoder.load_state_dict(ckpt["decoder_state"])
                quantizer.load_state_dict(ckpt["quantizer_state"])
                shape_latents.weight.data = torch.from_numpy(ckpt["shape_latents"]).to(device)
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt["epoch"] + 1
                loss_log = ckpt.get("loss_log", [])
                lr_log = ckpt.get("lr_log", [])
                timing_log = ckpt.get("timing_log", [])
                logger.info("Resumed training from epoch %d", start_epoch)
            except Exception as exc:
                logger.warning(
                    "Failed to load checkpoint %s: %s. Starting fresh.",
                    latest_ckpt,
                    exc,
                )
                start_epoch = 1

    grouped_ids = defaultdict(list)
    for shape in selected_shape_ids:
        grouped_ids[shape["category_id"]].append(shape["model_id"])

    logger.info("Starting VQ-DeepSDF training")
    logger.info("Device: %s", device)
    logger.info("Data root: %s", Path(data_root))
    logger.info("Artifacts directory: %s", save_path)
    logger.info("Training log file: %s", log_file)

    for category_id in sorted(grouped_ids):
        logger.info("Selected IDs | category=%s count=%d", category_id, len(grouped_ids[category_id]))

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        perm = np.random.permutation(num_shapes)

        recon_running = 0.0
        vq_running = 0.0
        entropy_running = 0.0
        perplexity_running = 0.0
        total_running = 0.0
        count = 0

        for param_group, schedule in zip(optimizer.param_groups[:2], lr_schedules):
            param_group["lr"] = schedule.get_learning_rate(epoch)
        optimizer.param_groups[2]["lr"] = lr_schedules[1].get_learning_rate(epoch)

        for i in range(0, num_shapes, scenes_per_batch):
            batch_indices = perm[i : i + scenes_per_batch]
            batch_samples = []
            batch_shape_indices = []

            for idx in batch_indices:
                data = dataset[idx]["data"]
                if data.shape[0] == 0:
                    continue
                sample = _sample_shape_points(data, samples_per_scene)
                if sample.shape[0] == 0:
                    continue
                batch_samples.append(sample)
                batch_shape_indices.append(np.full((sample.shape[0],), idx, dtype=np.int64))

            if not batch_samples:
                continue

            sdf_data = np.vstack(batch_samples)
            indices = np.concatenate(batch_shape_indices)

            num_sdf_samples = sdf_data.shape[0]
            xyz = torch.from_numpy(sdf_data[:, 0:3]).to(device)
            sdf_gt = torch.from_numpy(sdf_data[:, 3:4]).to(device)
            if clamp_dist is not None:
                sdf_gt = torch.clamp(sdf_gt, min_t, max_t)

            indices = torch.from_numpy(indices).to(device)

            xyz_chunks = torch.chunk(xyz, batch_split)
            sdf_chunks = torch.chunk(sdf_gt, batch_split)
            idx_chunks = torch.chunk(indices, batch_split)

            batch_total_loss = 0.0
            batch_recon_loss = 0.0
            batch_vq_loss = 0.0
            batch_entropy = 0.0
            batch_perplexity = 0.0
            optimizer.zero_grad()

            for chunk_idx in range(batch_split):
                chunk_indices = idx_chunks[chunk_idx]
                unique_indices, inverse = torch.unique(chunk_indices, return_inverse=True)

                z_e_unique = shape_latents(unique_indices)
                vq_out = quantizer(z_e_unique)
                z_q_chunk = vq_out.quantized_flat[inverse]

                pred_sdf = decoder(z_q_chunk, xyz_chunks[chunk_idx])
                if clamp_dist is not None:
                    pred_sdf = torch.clamp(pred_sdf, min_t, max_t)

                recon_loss = l1_loss(pred_sdf, sdf_chunks[chunk_idx]) / num_sdf_samples
                vq_loss = (
                    codebook_weight * vq_out.codebook_loss
                    + commitment_weight * vq_out.commitment_loss
                    - entropy_weight * vq_out.entropy
                )
                chunk_loss = recon_loss + vq_loss

                chunk_loss.backward()

                batch_total_loss += float(chunk_loss.item())
                batch_recon_loss += float(recon_loss.item())
                batch_vq_loss += float(vq_loss.item())
                batch_entropy += float(vq_out.entropy.item())
                batch_perplexity += float(vq_out.perplexity.item())

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(shape_latents.parameters(), grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(quantizer.parameters(), grad_clip_norm)

            optimizer.step()

            recon_running += batch_recon_loss
            vq_running += batch_vq_loss
            entropy_running += batch_entropy
            perplexity_running += batch_perplexity
            total_running += batch_total_loss
            count += 1

            if (count % log_frequency) == 0:
                logger.info(
                    "Epoch %d | batch %d | total=%.6f recon=%.6f vq=%.6f perplexity=%.2f",
                    epoch,
                    count,
                    total_running / count,
                    recon_running / count,
                    vq_running / count,
                    perplexity_running / count,
                )

        epoch_time = time.time() - start
        avg_total = total_running / max(1, count)
        avg_recon = recon_running / max(1, count)
        avg_vq = vq_running / max(1, count)
        avg_entropy = entropy_running / max(1, count)
        avg_perplexity = perplexity_running / max(1, count)

        logger.info(
            "Epoch %d completed | duration=%.1fs | total=%.6f recon=%.6f vq=%.6f entropy=%.3f perplexity=%.2f",
            epoch,
            epoch_time,
            avg_total,
            avg_recon,
            avg_vq,
            avg_entropy,
            avg_perplexity,
        )

        loss_log.append(
            {
                "total": avg_total,
                "recon": avg_recon,
                "vq": avg_vq,
                "entropy": avg_entropy,
                "perplexity": avg_perplexity,
            }
        )
        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
        timing_log.append(epoch_time)

        shape_code_indices = _compute_shape_indices(shape_latents, quantizer)

        ckpt = {
            "decoder_state": decoder.state_dict(),
            "quantizer_state": quantizer.state_dict(),
            "shape_latents": shape_latents.weight.data.detach().cpu().numpy(),
            "shape_code_indices": shape_code_indices.detach().cpu().numpy(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss_log": loss_log,
            "lr_log": lr_log,
            "timing_log": timing_log,
        }

        if epoch % snapshot_frequency == 0 or epoch in additional_snapshots:
            torch.save(ckpt, save_path / f"deepsdf_vq_epoch_{epoch}.pth")
            logger.info("Saved checkpoint: %s", save_path / f"deepsdf_vq_epoch_{epoch}.pth")

        torch.save(ckpt, save_path / "deepsdf_vq_latest.pth")

    meta = {
        "model_type": "vq_deepsdf",
        "num_shapes": num_shapes,
        "latent_size": latent_size,
        "hidden_size": hidden_size,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "code_dim": code_dim,
        "random_seed": random_seed,
    }
    with open(save_path / "meta.json", "w") as f:
        json.dump(meta, f)


def main():
    parser = argparse.ArgumentParser(description="Train VQ-DeepSDF auto-decoder on preprocessed SDF data")
    parser.add_argument(
        "data_root",
        nargs="?",
        default=str(DEEPSDF_VQ_TRAINING["data_root"]),
        help="Root directory with preprocessed SDF files",
    )
    parser.add_argument("--latent-size", type=int, default=DEEPSDF_VQ_TRAINING["latent_size"])
    parser.add_argument("--hidden-size", type=int, default=DEEPSDF_VQ_TRAINING["hidden_size"])
    parser.add_argument("--lr", type=float, default=DEEPSDF_VQ_TRAINING["lr"])
    parser.add_argument("--seed", type=int, default=DEEPSDF_VQ_TRAINING["random_seed"])
    parser.add_argument("--epochs", type=int, default=DEEPSDF_VQ_TRAINING["epochs"])
    parser.add_argument("--batch-points", type=int, default=DEEPSDF_VQ_TRAINING["batch_points"])
    parser.add_argument("--save-dir", type=str, default=str(DEEPSDF_VQ_TRAINING["save_dir"]))

    args = parser.parse_args()

    train_vq_autodecoder(
        args.data_root,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        random_seed=args.seed,
        epochs=args.epochs,
        batch_points=args.batch_points,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
