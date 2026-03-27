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
    """Grouped VQ bottleneck that maps latent groups to nearest codebook entries."""

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        code_dim: int,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.code_dim = code_dim

        self.codebook = nn.Parameter(
            torch.empty(num_codebooks, codebook_size, code_dim)
        )
        nn.init.uniform_(self.codebook, -init_scale, init_scale)

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

        gather_index = indices.unsqueeze(-1).expand(-1, -1, self.code_dim)
        quantized = torch.gather(self.codebook, dim=1, index=gather_index)
        return quantized.reshape(indices.size(0), self.latent_size)

    def forward(self, z: torch.Tensor) -> VQOutput:
        z_groups = self._reshape_groups(z)
        distances = (
            (z_groups.unsqueeze(2) - self.codebook.unsqueeze(0)).pow(2).sum(dim=-1)
        )
        indices = torch.argmin(distances, dim=-1)

        gather_index = indices.unsqueeze(-1).expand(-1, -1, self.code_dim)
        z_q_groups = torch.gather(self.codebook, dim=1, index=gather_index)

        commitment_loss = F.mse_loss(z_groups, z_q_groups.detach())
        codebook_loss = F.mse_loss(z_q_groups, z_groups.detach())

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
