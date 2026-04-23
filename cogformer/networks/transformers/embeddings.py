import math

import torch
import torch.nn as nn
import numpy as np


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal embedding for a scalar input.

    Maps (..., 1) → (..., dim) using sin/cos at geometrically spaced frequencies.
    No learned parameters; negligible compute overhead.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalEmbedding dim must be even"
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=x.device, dtype=x.dtype)
            / max(half - 1, 1)
        )
        args = x * freqs  # (..., half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., dim)


class FourierEmbedding(nn.Module):
    """Fourier projection embedding with normally distributed frequencies.

    Based on Tancik et al. (2020), Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains.

    Maps (..., 1) → (..., embed_dim) via sin/cos projections with scaled random frequencies.
    Frequencies are trainable by default, allowing the model to learn which frequency
    range is most informative.

    Args:
        embed_dim   : output dimensionality (must be even; embed_dim // 2 frequencies)
        scale       : std of the initial frequency distribution; controls the range of
                      frequencies at initialisation. Default 1.0 is appropriate for
                      flow-matching time t ∈ [0, 1].
        trainable   : if True (default), frequencies are learned during training.
                      If False, they are fixed after initialisation.
    """
    def __init__(self, embed_dim: int, scale: float = 1.0, trainable: bool = True):
        super().__init__()
        assert embed_dim % 2 == 0, "FourierEmbedding embed_dim must be even"
        self.embed_dim = embed_dim
        self.scale = scale

        frequencies = torch.randn(embed_dim // 2) * scale
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        proj = x * self.frequencies * 2 * math.pi   # (..., embed_dim // 2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., embed_dim)





