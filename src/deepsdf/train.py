import argparse
from collections import defaultdict
from pathlib import Path
import json
import math
import time

import numpy as np
import torch
from .dataset import DeepSDFDataset
from .logging_utils import setup_training_logger
from .model import DeepSDFDecoder
from config import DEEPSDF_TRAINING


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        raise NotImplementedError


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def _build_lr_schedules(lr, schedule_specs):
    if not schedule_specs:
        return [ConstantLearningRateSchedule(lr), ConstantLearningRateSchedule(lr)]

    schedules = []
    for spec in schedule_specs:
        schedule_type = spec.get("type", "constant").lower()
        if schedule_type == "step":
            schedules.append(
                StepLearningRateSchedule(
                    spec["initial"],
                    spec["interval"],
                    spec["factor"],
                )
            )
        elif schedule_type == "warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    spec["initial"],
                    spec["final"],
                    spec["length"],
                )
            )
        elif schedule_type == "constant":
            schedules.append(ConstantLearningRateSchedule(spec["value"]))
        else:
            raise ValueError(f"Unknown learning rate schedule type: {schedule_type}")

    if len(schedules) == 1:
        schedules = schedules * 2
    return schedules


def _sample_shape_points(data, samples_per_scene):
    if data.shape[0] == 0:
        return data
    if data.shape[0] > samples_per_scene:
        choice = np.random.choice(data.shape[0], samples_per_scene, replace=False)
        return data[choice]
    choice = np.random.choice(data.shape[0], samples_per_scene, replace=True)
    return data[choice]


def train_autodecoder(
    data_root,
    latent_size=DEEPSDF_TRAINING["latent_size"],
    hidden_size=DEEPSDF_TRAINING["hidden_size"],
    lr=DEEPSDF_TRAINING["lr"],
    random_seed=DEEPSDF_TRAINING["random_seed"],
    objects_per_category=DEEPSDF_TRAINING["objects_per_category"],
    epochs=DEEPSDF_TRAINING["epochs"],
    batch_points=DEEPSDF_TRAINING["batch_points"],
    device=None,
    save_dir=DEEPSDF_TRAINING["save_dir"],
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    dataset = DeepSDFDataset(
        data_root,
        objects_per_category=objects_per_category,
        random_seed=random_seed,
    )
    num_shapes = len(dataset)
    if num_shapes == 0:
        raise RuntimeError(f"No shapes found under {data_root}")

    decoder = DeepSDFDecoder(latent_size=latent_size, hidden_size=hidden_size).to(device)

    # Create latent code embedding (auto-decoder)
    code_bound = DEEPSDF_TRAINING["code_bound"]
    latents = torch.nn.Embedding(num_shapes, latent_size, max_norm=code_bound).to(device)
    torch.nn.init.normal_(
        latents.weight,
        mean=0.0,
        std=DEEPSDF_TRAINING["code_init_stddev"] / math.sqrt(latent_size),
    )

    schedule_specs = DEEPSDF_TRAINING["lr_schedules"]
    lr_schedules = _build_lr_schedules(lr, schedule_specs)

    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": lr_schedules[0].get_learning_rate(0)},
            {"params": latents.parameters(), "lr": lr_schedules[1].get_learning_rate(0)},
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
        "objects_per_category": objects_per_category,
        "random_seed": random_seed,
        "shapes": selected_shape_ids,
    }
    selected_manifest_path = save_path / "selected_samples.json"
    with open(selected_manifest_path, "w") as f:
        json.dump(selected_manifest, f, indent=2)

    selected_ids_txt_path = save_path / "selected_sample_ids.txt"
    with open(selected_ids_txt_path, "w") as f:
        for shape in selected_shape_ids:
            f.write(f"{shape['category_id']}/{shape['model_id']}\n")

    samples_per_scene = batch_points or DEEPSDF_TRAINING["samples_per_scene"]
    scenes_per_batch = DEEPSDF_TRAINING["scenes_per_batch"]
    batch_split = DEEPSDF_TRAINING["batch_split"]
    clamp_dist = DEEPSDF_TRAINING["clamp_dist"]
    do_code_regularization = DEEPSDF_TRAINING["code_regularization"]
    code_reg_lambda = DEEPSDF_TRAINING["code_regularization_lambda"]
    grad_clip_norm = DEEPSDF_TRAINING["grad_clip_norm"]
    log_frequency = DEEPSDF_TRAINING["log_frequency"]
    snapshot_frequency = DEEPSDF_TRAINING["snapshot_frequency"]
    additional_snapshots = set(DEEPSDF_TRAINING["additional_snapshots"])

    min_t = -clamp_dist
    max_t = clamp_dist

    loss_log = []
    lr_log = []
    timing_log = []

    logger.info("Starting DeepSDF training")
    logger.info("Device: %s", device)
    logger.info("Data root: %s", Path(data_root))
    logger.info("Artifacts directory: %s", save_path)
    logger.info("Saved selected samples manifest: %s", selected_manifest_path)
    logger.info("Saved selected sample IDs list: %s", selected_ids_txt_path)
    logger.info(
        "Hyperparameters | epochs=%d latent_size=%d hidden_size=%d lr=%.6f batch_points=%d objects_per_category=%s seed=%d",
        epochs,
        latent_size,
        hidden_size,
        lr,
        batch_points,
        objects_per_category,
        random_seed,
    )

    grouped_ids = defaultdict(list)
    for shape in selected_shape_ids:
        grouped_ids[shape["category_id"]].append(shape["model_id"])
    for category_id in sorted(grouped_ids):
        logger.info("Selected IDs | category=%s count=%d", category_id, len(grouped_ids[category_id]))
        logger.info("Selected IDs | %s", ", ".join(grouped_ids[category_id]))

    for epoch in range(1, epochs + 1):
        start = time.time()
        perm = np.random.permutation(num_shapes)
        total_loss = 0.0
        count = 0

        for param_group, schedule in zip(optimizer.param_groups, lr_schedules):
            param_group["lr"] = schedule.get_learning_rate(epoch)

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
                batch_shape_indices.append(
                    np.full((sample.shape[0],), idx, dtype=np.int64)
                )

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

            batch_loss = 0.0
            optimizer.zero_grad()

            for chunk_idx in range(batch_split):
                batch_vecs = latents(idx_chunks[chunk_idx])
                pred_sdf = decoder(batch_vecs, xyz_chunks[chunk_idx])
                if clamp_dist is not None:
                    pred_sdf = torch.clamp(pred_sdf, min_t, max_t)

                chunk_loss = l1_loss(pred_sdf, sdf_chunks[chunk_idx]) / num_sdf_samples

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1.0, epoch / 100.0) * l2_size_loss
                    ) / num_sdf_samples
                    chunk_loss = chunk_loss + reg_loss

                chunk_loss.backward()
                batch_loss += float(chunk_loss.item())

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip_norm)

            optimizer.step()

            total_loss += batch_loss
            count += 1

            if (count % log_frequency) == 0:
                logger.info(
                    "Epoch %d | batch %d | running_loss=%.6f",
                    epoch,
                    count,
                    total_loss / count,
                )

        epoch_time = time.time() - start
        avg_loss = total_loss / max(1, count)
        logger.info(
            "Epoch %d completed | duration=%.1fs | avg_loss=%.6f",
            epoch,
            epoch_time,
            avg_loss,
        )

        loss_log.append(avg_loss)
        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
        timing_log.append(epoch_time)

        # Save checkpoint
        ckpt = {
            "decoder_state": decoder.state_dict(),
            "latents": latents.weight.data.cpu().numpy(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "avg_loss": avg_loss,
            "loss_log": loss_log,
            "lr_log": lr_log,
            "timing_log": timing_log,
        }

        if epoch % snapshot_frequency == 0 or epoch in additional_snapshots:
            torch.save(ckpt, save_path / f"deepsdf_epoch_{epoch}.pth")
            logger.info("Saved checkpoint: %s", save_path / f"deepsdf_epoch_{epoch}.pth")

        torch.save(ckpt, save_path / "deepsdf_latest.pth")
        logger.info("Updated latest checkpoint: %s", save_path / "deepsdf_latest.pth")

    # Save final metadata
    meta = {
        "num_shapes": num_shapes,
        "latent_size": latent_size,
        "hidden_size": hidden_size,
        "objects_per_category": objects_per_category,
        "random_seed": random_seed,
    }
    with open(save_path / "meta.json", "w") as f:
        json.dump(meta, f)

    logger.info("Saved metadata: %s", save_path / "meta.json")
    logger.info("Training finished. Checkpoints saved to %s", save_path)
    logger.info("Training log file: %s", log_file)


def main():
    parser = argparse.ArgumentParser(description="Train DeepSDF auto-decoder on preprocessed SDF data")
    parser.add_argument(
        "data_root",
        nargs="?",
        default=str(DEEPSDF_TRAINING["data_root"]),
        help="Root directory with preprocessed SDF (per-shape sdf.npz files)",
    )
    parser.add_argument("--latent-size", type=int, default=DEEPSDF_TRAINING["latent_size"])
    parser.add_argument("--hidden-size", type=int, default=DEEPSDF_TRAINING["hidden_size"])
    parser.add_argument("--lr", type=float, default=DEEPSDF_TRAINING["lr"])
    parser.add_argument("--seed", type=int, default=DEEPSDF_TRAINING["random_seed"])
    parser.add_argument(
        "--objects-per-category",
        type=int,
        default=DEEPSDF_TRAINING["objects_per_category"],
        help="Max number of objects sampled per category (uses all when unset)",
    )
    parser.add_argument("--epochs", type=int, default=DEEPSDF_TRAINING["epochs"])
    parser.add_argument("--batch-points", type=int, default=DEEPSDF_TRAINING["batch_points"])
    parser.add_argument("--save-dir", type=str, default=str(DEEPSDF_TRAINING["save_dir"]))

    args = parser.parse_args()

    train_autodecoder(
        args.data_root,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        random_seed=args.seed,
        objects_per_category=args.objects_per_category,
        epochs=args.epochs,
        batch_points=args.batch_points,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
