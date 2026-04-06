from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    quantized_flat: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    entropy: torch.Tensor
    perplexity: torch.Tensor


class GroupedVectorQuantizer(nn.Module):
    """Grouped VQ bottleneck with EMA codebook updates and dead-code reset.

    The codebook is updated via exponential moving averages rather than
    back-propagation (VQ-VAE-2 style).  This is more stable than gradient-only
    updates for small datasets and prevents codebook collapse.

    Dead codes (those whose EMA cluster size falls below `dead_code_threshold`)
    are reseeded with random encoder outputs from the current batch so that the
    full codebook capacity stays in use.
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        code_dim: int,
        init_scale: float = 0.1,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook is a buffer — updated via EMA, not gradients.
        codes = torch.empty(num_codebooks, codebook_size, code_dim)
        nn.init.uniform_(codes, -init_scale, init_scale)
        self.register_buffer("codebook", codes)

        # EMA accumulators: cluster sizes and sum of assigned encoder outputs.
        self.register_buffer(
            "ema_cluster_size",
            torch.ones(num_codebooks, codebook_size),
        )
        self.register_buffer(
            "ema_embed_sum",
            codes.clone(),
        )

    @property
    def latent_size(self) -> int:
        return self.num_codebooks * self.code_dim

    def _reshape_groups(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"Expected [batch, latent_size], got {tuple(z.shape)}")
        if z.size(1) != self.latent_size:
            raise ValueError(
                f"Latent size mismatch: expected {self.latent_size}, got {z.size(1)}"
            )
        return z.view(z.size(0), self.num_codebooks, self.code_dim)

    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Return nearest code indices with shape [batch, num_codebooks]."""
        z_groups = self._reshape_groups(z)
        distances = (
            (z_groups.unsqueeze(2) - self.codebook.unsqueeze(0)).pow(2).sum(dim=-1)
        )
        return torch.argmin(distances, dim=-1)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices [batch, num_codebooks] to flat latent vectors [batch, latent_size]."""
        if indices.dim() != 2:
            raise ValueError(
                f"Expected index tensor [batch, num_codebooks], got {tuple(indices.shape)}"
            )
        if indices.size(1) != self.num_codebooks:
            raise ValueError(
                "Index tensor has wrong number of codebooks: "
                f"expected {self.num_codebooks}, got {indices.size(1)}"
            )

        group_idx = torch.arange(self.num_codebooks, device=indices.device).unsqueeze(0).expand(indices.size(0), -1)
        quantized = self.codebook[group_idx, indices]  # [B, G, D]
        return quantized.reshape(indices.size(0), self.latent_size)

    def _ema_update(self, z_groups: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook entries in-place via EMA and reset dead codes.

        Args:
            z_groups: encoder outputs [B, num_codebooks, code_dim]
            indices:  nearest-code assignments [B, num_codebooks]
        """
        gamma = self.ema_decay
        z_det = z_groups.detach()   # [B, G, D] — no gradient through EMA path
        B = z_det.size(0)

        # one_hot: [B, G, K]
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).to(dtype=z_det.dtype)

        # New per-group counts [G, K] and embed sums [G, K, D].
        new_counts = one_hot.sum(dim=0)
        new_sum = torch.einsum("bgk,bgd->gkd", one_hot, z_det)

        # EMA updates (in-place).
        self.ema_cluster_size.mul_(gamma).add_(new_counts, alpha=1.0 - gamma)
        self.ema_embed_sum.mul_(gamma).add_(new_sum, alpha=1.0 - gamma)

        # Laplace-smoothed normalisation to avoid division by near-zero cluster sizes.
        n = self.ema_cluster_size.sum(dim=-1, keepdim=True)           # [G, 1]
        smoothed = (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        self.codebook.copy_(self.ema_embed_sum / smoothed.unsqueeze(-1))

        # Dead-code reset: reinitialise unused codes to random encoder outputs
        # so dormant codebook entries can be reclaimed.
        dead = self.ema_cluster_size < self.dead_code_threshold       # [G, K]
        if dead.any():
            for g in range(self.num_codebooks):
                dead_g = dead[g]
                num_dead = int(dead_g.sum().item())
                if num_dead == 0:
                    continue
                n_avail = min(B, num_dead)
                perm = torch.randperm(B, device=z_det.device)[:n_avail]
                replacements = z_det[perm, g, :]                     # [n_avail, D]
                if n_avail < num_dead:
                    reps = (num_dead + n_avail - 1) // n_avail
                    replacements = replacements.repeat(reps, 1)[:num_dead]
                self.codebook[g, dead_g, :] = replacements
                self.ema_embed_sum[g, dead_g, :] = replacements
                self.ema_cluster_size[g, dead_g] = self.dead_code_threshold

    def forward(self, z: torch.Tensor) -> VQOutput:
        z_groups = self._reshape_groups(z)
        distances = (
            (z_groups.unsqueeze(2) - self.codebook.unsqueeze(0)).pow(2).sum(dim=-1)
        )
        indices = torch.argmin(distances, dim=-1)

        group_idx = torch.arange(self.num_codebooks, device=indices.device).unsqueeze(0).expand(indices.size(0), -1)
        z_q_groups = self.codebook[group_idx, indices]  # [B, G, D]

        # EMA codebook update during training — no gradient flows through codebook.
        if self.training:
            self._ema_update(z_groups, indices)

        # Only the commitment loss is needed; the codebook is updated via EMA.
        commitment_loss = F.mse_loss(z_groups, z_q_groups.detach())
        codebook_loss = torch.zeros((), device=z.device)  # EMA replaces gradient update; kept for API compatibility

        z_st_groups = z_groups + (z_q_groups - z_groups).detach()

        usage = F.one_hot(indices, num_classes=self.codebook_size).float().mean(dim=(0, 1))
        entropy = -(usage * torch.log(usage + 1e-10)).sum()
        perplexity = torch.exp(entropy)

        return VQOutput(
            quantized_flat=z_st_groups.reshape(z.size(0), self.latent_size),
            indices=indices,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            entropy=entropy,
            perplexity=perplexity,
        )
