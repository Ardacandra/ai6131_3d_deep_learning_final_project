import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEEPSDF_MODEL


class DeepSDFDecoder(nn.Module):
    """DeepSDF decoder modeled after the reference architecture."""

    def __init__(
        self,
        latent_size=DEEPSDF_MODEL["latent_size"],
        hidden_size=DEEPSDF_MODEL["hidden_size"],
        num_layers=DEEPSDF_MODEL["num_layers"],
        dims=DEEPSDF_MODEL["dims"],
        dropout=DEEPSDF_MODEL["dropout"],
        dropout_prob=DEEPSDF_MODEL["dropout_prob"],
        norm_layers=DEEPSDF_MODEL["norm_layers"],
        latent_in=DEEPSDF_MODEL["latent_in"],
        weight_norm=DEEPSDF_MODEL["weight_norm"],
        xyz_in_all=DEEPSDF_MODEL["xyz_in_all"],
        use_tanh=DEEPSDF_MODEL["use_tanh"],
        latent_dropout=DEEPSDF_MODEL["latent_dropout"],
    ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if dims is None:
            dims = [hidden_size] * num_layers

        dims = [latent_size + 3] + list(dims) + [1]

        self.dims = dims
        self.num_layers = len(dims)
        self.norm_layers = tuple(norm_layers)
        self.latent_in = tuple(latent_in)
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (not weight_norm) and self.norm_layers and layer in self.norm_layers:
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    def forward(self, z, xyz):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if xyz.dim() == 1:
            xyz = xyz.unsqueeze(0)

        input_tensor = torch.cat([z, xyz], dim=-1)
        xyz_tensor = input_tensor[:, -3:]

        if input_tensor.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input_tensor[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz_tensor], 1)
        else:
            x = input_tensor

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input_tensor], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz_tensor], 1)
            x = lin(x)
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (not self.weight_norm) and self.norm_layers and layer in self.norm_layers:
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
