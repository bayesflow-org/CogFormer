import torch.nn as nn

from .self_attention_block import SelfAttentionBlock
from .pooling_by_multihead_attention import PoolingByMultiheadAttention


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        seed_dim: int = 128,
        num_layers: int = 3,
        num_seeds: int = 1,
        num_heads: int = 4,
        layer_norm: bool = False,
        dropout: float = 0.0,
        layer_dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.input_embedding = nn.Linear(input_dim, embed_dim)

        self.layers = nn.ModuleList(
            [SelfAttentionBlock(
                input_dim=embed_dim,
                num_heads=num_heads,
                layer_norm=layer_norm,
                dropout=dropout
            ) for _ in range(num_layers)]
        )
        self.post_dropout = nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()

        self.pma = PoolingByMultiheadAttention(
            input_dim=embed_dim,
            seed_dim=seed_dim,
            num_seeds=num_seeds,
            num_heads=num_heads,
            layer_norm=layer_norm,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None):
        """
        X: (B, T, C)
        attn_mask: optional (T, T) or (B*num_heads, T, T)
        key_padding_mask: optional (B, T) with True for PAD tokens
        """

        out = self.input_embedding(x)

        for sab in self.layers:
            out = sab(out, attn_mask=attn_mask)
            out = self.post_dropout(out)

        pooled = self.pma(out)
        return pooled