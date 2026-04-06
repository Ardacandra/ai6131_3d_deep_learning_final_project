import argparse
from pathlib import Path
import json
from typing import Dict, Tuple

import numpy as np
import torch

from .model import DeepSDFDecoder
from .quantizer import GroupedVectorQuantizer
from src.deepsdf.evaluate import evaluate_dataset
from config import DEEPSDF_VQ_EVALUATION, DEEPSDF_VQ_MODEL, DEEPSDF_VQ_TRAINING


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.exists():
        return path

    candidates = sorted(path.parent.glob("deepsdf_vq_epoch_*.pth"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_vq_checkpoint(checkpoint_path: str, device: str = "cpu") -> Tuple[DeepSDFDecoder, torch.Tensor, Dict]:
    """Load VQ-DeepSDF checkpoint and return decoder plus quantized latent table."""
    resolved = _resolve_checkpoint_path(checkpoint_path)
    ckpt = torch.load(resolved, map_location=device, weights_only=False)

    meta_path = resolved.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    latent_size = int(meta.get("latent_size", DEEPSDF_VQ_TRAINING["latent_size"]))
    hidden_size = int(meta.get("hidden_size", DEEPSDF_VQ_TRAINING["hidden_size"]))
    num_codebooks = int(meta.get("num_codebooks", DEEPSDF_VQ_MODEL["num_codebooks"]))
    codebook_size = int(meta.get("codebook_size", DEEPSDF_VQ_MODEL["codebook_size"]))
    code_dim = int(meta.get("code_dim", DEEPSDF_VQ_MODEL["code_dim"]))

    decoder = DeepSDFDecoder(latent_size=latent_size, hidden_size=hidden_size).to(device)
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()

    quantizer = GroupedVectorQuantizer(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        code_dim=code_dim,
        init_scale=DEEPSDF_VQ_MODEL["codebook_init_scale"],
        ema_decay=DEEPSDF_VQ_MODEL["ema_decay"],
        dead_code_threshold=DEEPSDF_VQ_MODEL["dead_code_threshold"],
    ).to(device)
    quantizer.load_state_dict(ckpt["quantizer_state"], strict=False)
    quantizer.eval()

    if "shape_code_indices" in ckpt:
        indices = ckpt["shape_code_indices"]
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices)
        indices = indices.to(device=device, dtype=torch.long)
        latents = quantizer.decode_indices(indices)
    else:
        shape_latents = ckpt["shape_latents"]
        if isinstance(shape_latents, np.ndarray):
            shape_latents = torch.from_numpy(shape_latents)
        shape_latents = shape_latents.to(device=device, dtype=torch.float32)
        latents = quantizer(shape_latents).quantized_flat

    return decoder, latents, meta


def evaluate_vq_deepsdf(
    checkpoint_path: str,
    data_root: str,
    gt_data_root: str,
    resolution: int,
    num_sample_points: int,
    batch_size: int,
    max_shapes=None,
    output_file=None,
    percentile: float = DEEPSDF_VQ_EVALUATION["percentile"],
):
    """Run baseline DeepSDF evaluation on VQ-quantized latent codes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, latents, meta = load_vq_checkpoint(checkpoint_path, device=device)

    # Write a temporary checkpoint that evaluate_dataset can load via load_checkpoint.
    temp_ckpt_path = Path(checkpoint_path).parent / "_tmp_vq_eval_bridge.pth"
    bridge = {
        "decoder_state": decoder.state_dict(),
        "latents": latents.detach().cpu().numpy(),
    }
    torch.save(bridge, temp_ckpt_path)

    temp_meta_path = temp_ckpt_path.parent / "meta.json"
    original_meta = None
    if temp_meta_path.exists():
        with open(temp_meta_path, "r") as f:
            original_meta = f.read()

    with open(temp_meta_path, "w") as f:
        json.dump(
            {
                "latent_size": int(meta.get("latent_size", DEEPSDF_VQ_TRAINING["latent_size"])),
                "hidden_size": int(meta.get("hidden_size", DEEPSDF_VQ_TRAINING["hidden_size"])),
            },
            f,
        )

    try:
        return evaluate_dataset(
            checkpoint_path=str(temp_ckpt_path),
            data_root=data_root,
            gt_data_root=gt_data_root,
            device=device,
            resolution=resolution,
            num_sample_points=num_sample_points,
            max_shapes=max_shapes,
            output_path=output_file,
        )
    finally:
        if temp_ckpt_path.exists():
            temp_ckpt_path.unlink()
        if original_meta is None:
            temp_meta_path.unlink(missing_ok=True)
        else:
            with open(temp_meta_path, "w") as f:
                f.write(original_meta)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQ-DeepSDF model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEEPSDF_VQ_EVALUATION["checkpoint"]),
        help="Path to VQ-DeepSDF checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEEPSDF_VQ_EVALUATION["data_root"]),
        help="Path to SDF test data",
    )
    parser.add_argument(
        "--gt-data-root",
        type=str,
        default=str(DEEPSDF_VQ_EVALUATION["gt_data_root"]),
        help="Path to ground truth mesh data",
    )
    parser.add_argument("--resolution", type=int, default=DEEPSDF_VQ_EVALUATION["resolution"])
    parser.add_argument("--num-sample-points", type=int, default=DEEPSDF_VQ_EVALUATION["num_sample_points"])
    parser.add_argument("--batch-size", type=int, default=DEEPSDF_VQ_EVALUATION["batch_size"])
    parser.add_argument("--max-shapes", type=int, default=DEEPSDF_VQ_EVALUATION["max_shapes"])
    parser.add_argument("--output-file", type=str, default=str(DEEPSDF_VQ_EVALUATION["output_file"]))
    parser.add_argument("--percentile", type=float, default=DEEPSDF_VQ_EVALUATION["percentile"])

    args = parser.parse_args()

    evaluate_vq_deepsdf(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        gt_data_root=args.gt_data_root,
        resolution=args.resolution,
        num_sample_points=args.num_sample_points,
        batch_size=args.batch_size,
        max_shapes=args.max_shapes,
        output_file=args.output_file,
        percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
