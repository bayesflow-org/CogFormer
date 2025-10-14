import torch.nn as nn
from .mab import MAB


class SAB(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, ln: bool = False, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(query_dim=input_dim, num_heads=num_heads, ln=ln, dropout=dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        return self.mab(x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
