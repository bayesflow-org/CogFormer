import torch
import torch.nn as nn
from .mab import MAB


class PMA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seed_dim: int,
        num_heads: int = 4,
        num_seeds: int = 1,
        ln: bool = False,
        dropout: float = 0.0
):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, seed_dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(query_dim=seed_dim, key_dim=input_dim, num_heads=num_heads, ln=ln, dropout=dropout)

    def forward(self, x):
        return self.mab(self.S.repeat(x.size(0), 1, 1), x)