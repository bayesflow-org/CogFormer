import torch
import torch.nn as nn
from .multihead_attention_block import MultiheadAttentionBlock


class PoolingByMultiheadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seed_dim: int,
        num_heads: int = 4,
        num_seeds: int = 1,
        layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.seed = nn.Parameter(torch.Tensor(1, num_seeds, seed_dim))
        nn.init.xavier_uniform_(self.seed)
        self.mab = MultiheadAttentionBlock(
            query_dim=seed_dim, 
            key_dim=input_dim, 
            num_heads=num_heads, 
            layer_norm=layer_norm, 
            dropout=dropout
        )

    def forward(self, x):
        return self.mab(self.S.repeat(x.size(0), 1, 1), x)