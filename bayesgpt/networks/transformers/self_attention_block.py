import torch.nn as nn
from .multihead_attention_block import MultiheadAttentionBlock


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        layer_norm: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mab = MultiheadAttentionBlock(
            query_dim=input_dim, 
            num_heads=num_heads, 
            layer_norm=layer_norm,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        return self.mab(x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
