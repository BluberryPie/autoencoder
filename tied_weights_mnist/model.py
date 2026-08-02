import math
from collections.abc import Sequence

import torch
import torch.nn as nn


def build_alternating(
    linear_layers: Sequence[nn.Module],
    activation: type[nn.Module],
    final_activation: type[nn.Module] | None,
):
    layers: list[nn.Module] = []
    for linear_layer in linear_layers:
        layers.append(linear_layer)
        layers.append(activation())
    layers = layers[:-1]
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class TiedLinear(nn.Module):
    def __init__(self, source: nn.Module):
        """`source`s stores the reference to the corresponding linear layer of the encoder"""
        super().__init__()
        self.source = source
        self.bias = self._init_bias()

    def _init_bias(self) -> torch.Tensor:
        bound = 1 / math.sqrt(self.source.out_features)
        dist = torch.distributions.Uniform(-bound, bound)
        bias = dist.sample(sample_shape=(self.source.in_features,))
        return nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            input=x, weight=self.source.weight.t(), bias=self.bias
        )


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module],
    ):
        super().__init__()
        layer_dims = [input_dim, *hidden_dims, latent_dim]
        self.linear_layers = [
            nn.Linear(in_features=d_in, out_features=d_out)
            for d_in, d_out in zip(layer_dims, layer_dims[1:])
        ]
        self.layers = build_alternating(
            linear_layers=self.linear_layers,
            activation=activation,
            final_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(
        self,
        activation: type[nn.Module],
        sources: Sequence[nn.Module],
    ):
        super().__init__()
        linear_layers = [TiedLinear(source=source) for source in sources]
        self.layers = build_alternating(
            linear_layers=linear_layers,
            activation=activation,
            final_activation=nn.Sigmoid,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TiedAutoEncoder(nn.Module):
    """AutoEncoder with tied weights"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, activation)
        self.decoder = Decoder(
            activation, sources=list(reversed(self.encoder.linear_layers))
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


class AutoEncoder(nn.Module):
    """Plain Autoencoder (Baseline model for comparison)"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        enc_layer_sizes: list[int] = [input_dim, *hidden_dims, latent_dim]
        dec_layer_sizes: list[int] = [latent_dim, *reversed(hidden_dims), input_dim]
        enc_linear_layers = [
            nn.Linear(in_features=d_in, out_features=d_out)
            for d_in, d_out in zip(enc_layer_sizes, enc_layer_sizes[1:])
        ]
        dec_linear_layers = [
            nn.Linear(in_features=d_in, out_features=d_out)
            for d_in, d_out in zip(dec_layer_sizes, dec_layer_sizes[1:])
        ]
        self.encoder = build_alternating(
            linear_layers=enc_linear_layers,
            activation=activation,
            final_activation=None,
        )
        self.decoder = build_alternating(
            linear_layers=dec_linear_layers,
            activation=activation,
            final_activation=nn.Sigmoid,
        )

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(input)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        latent = self.encode(input)
        reconstructed = self.decode(latent)
        return reconstructed
