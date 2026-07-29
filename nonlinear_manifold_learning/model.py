from collections.abc import Sequence

import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        enc_layer_sizes: list[int] = [input_dim, *hidden_dims, latent_dim]
        dec_layer_sizes: list[int] = [latent_dim, *reversed(hidden_dims), input_dim]
        self.encoder = self._build_mlp(enc_layer_sizes)
        self.decoder = self._build_mlp(dec_layer_sizes)

    def _build_mlp(
        self, layer_sizes: Sequence[int], activation: type[nn.Module] = nn.ReLU
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for d_in, d_out in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(in_features=d_in, out_features=d_out))
            layers.append(activation())
        return nn.Sequential(*layers[:-1])
