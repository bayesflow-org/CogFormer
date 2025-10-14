import torch
import torch.nn as nn
from .sab import SAB


class Decoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        ln: bool = False,
        dropout: float = 0.0,
        layer_dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.input_proj = nn.Linear(input_dim, proj_dim)

        self.layers = nn.ModuleList(
            [SAB(input_dim=proj_dim, num_heads=num_heads, ln=ln, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.post_dropout = nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()

        self.output_proj = nn.Linear(proj_dim, 2)


    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        X: (B, T, C)
        attn_mask: optional (T, T) or (B*num_heads, T, T)
        key_padding_mask: optional (B, T) with True for PAD tokens
        """

        out = self.input_proj(x)

        for sab in self.layers:
            out = sab(out, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            out = self.post_dropout(out)

        mu, log_var = self.output_proj(out).chunk(2, dim=-1)
        return mu, log_var