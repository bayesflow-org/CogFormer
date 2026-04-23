import torch.nn as nn
from .attention_layers import (
    cross_attention_layers,
    self_attention_layers,
    mixed_attention_layers,
    custom_attention_layers
)
from .self_attention_block import SelfAttentionBlock


class Decoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        seed_dim: int = 128,
        proj_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        layer_norm: bool = False,
        dropout: float = 0.05,
        layer_design: str = None,
        layer_kwargs: dict = None,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.input_proj = nn.Linear(input_dim, proj_dim)

        if layer_design is not None:
            match layer_design:
                case "self_attention":
                    self.layers = self_attention_layers(
                        query_dim=proj_dim,
                        key_dim=seed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        layer_norm=layer_norm,
                        dropout=dropout,
                        **layer_kwargs
                    )
                case "mixed_attention":
                    self.layers = mixed_attention_layers(
                        query_dim=proj_dim,
                        key_dim=seed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        layer_norm=layer_norm,
                        dropout=dropout,
                        **layer_kwargs
                    )
                case "custom_attention":
                    self.layers = custom_attention_layers(
                        query_dim=proj_dim,
                        key_dim=seed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        layer_norm=layer_norm,
                        dropout=dropout,
                        **layer_kwargs
                    )
        else:
            self.layers = mixed_attention_layers(
                query_dim=proj_dim,
                key_dim=seed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                layer_norm=layer_norm,
                dropout=dropout,
                **layer_kwargs
            )

        self.output_proj = nn.Linear(proj_dim, 2)
        self.num_heads = num_heads


    def forward(self, query, key, query_mask=None):
        """
        X: (B, T, C)
        attn_mask: optional (T, T) or (B*num_heads, T, T)
        key_padding_mask: optional (B, T) with True for PAD tokens
        """
        out = self.input_proj(query)

        if query_mask is not None:
            batch_size, query_dim = query_mask.shape
            key_dim = key.shape[1]

            query_block = query_mask == 0

            #  (B*H, Tq, Tk) - repeat batch mask across heads
            cross_attn_mask_bt = query_block.unsqueeze(-1).expand(batch_size, query_dim, key_dim)
            cross_attn_mask = cross_attn_mask_bt.repeat_interleave(self.num_heads, dim=0)

            self_attn_mask_bt = query_block.unsqueeze(-1).expand(batch_size, query_dim, query_dim)
            self_attn_mask = self_attn_mask_bt.repeat_interleave(self.num_heads, dim=0)

        else:
            cross_attn_mask = None
            self_attn_mask = None

        for layer in self.layers:
            if isinstance(layer, SelfAttentionBlock):
                out = layer(query=out, attn_mask=self_attn_mask)
            else:
                out = layer(query=out, key=key, attn_mask=cross_attn_mask)

            if query_mask is not None:
                out = out * query_mask[..., None]

        mu, log_var = self.output_proj(out).chunk(2, dim=-1)

        if query_mask is not None:
            mu = mu * query_mask[..., None]
            log_var = log_var * query_mask[..., None]

        return mu, log_var
