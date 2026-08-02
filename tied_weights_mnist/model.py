import math

import torch
import torch.nn as nn


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
        return nn.functional.linear(input=x, weight=self.source.weight.t(), bias=self.bias)
