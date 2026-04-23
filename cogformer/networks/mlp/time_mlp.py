import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeMLP(nn.Module):
    """Time-conditioned output MLP using FiLM (Feature-wise Linear Modulation).

    Replaces a plain linear output projection with a small MLP where the hidden
    state is modulated by the time embedding before the final scalar projection.
    This allows the velocity field to condition more expressively on where in the
    flow trajectory (t) the model is at each step.

    FiLM reference: Perez et al. (2018), FiLM: Visual Reasoning with a General
    Conditioning Layer.

    Args:
        input_dim         : dimensionality of the incoming hidden state (proj_dim)
        time_embedding_dim: dimensionality of the time embedding
        hidden_dim        : width of the intermediate layer; defaults to input_dim
    """

    def __init__(self, input_dim: int, time_embedding_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        # FiLM: maps time embedding → per-feature scale (γ) and shift (β)
        self.film_projection = nn.Linear(time_embedding_dim, 2 * input_dim)

        # MLP: modulated features → scalar output
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        # FiLM modulation: scale and shift from time embedding
        gamma, beta = self.film_projection(time_embedding).chunk(2, dim=-1)  # each (..., input_dim)
        x = (1 + gamma) * x + beta

        # MLP to scalar
        return self.output(F.mish(self.hidden(x)))
