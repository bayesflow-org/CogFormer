import torch
import torch.nn as nn

from .attention_layers import (
    cross_attention_layers,
    self_attention_layers,
    mixed_attention_layers,
    custom_attention_layers
)
from .self_attention_block import SelfAttentionBlock
from cogformer.utils.tensor_utils import broadcast_right


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

        self.input_proj = nn.Linear(input_dim + 1 + 1, proj_dim)

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

        self.output_proj = nn.Linear(proj_dim, 1)
        self.num_heads = num_heads

    def velocity(self, theta_t, query, key, t, query_mask=None):

        query_block = None
        # Ensure parameter mask is batched
        if query_mask is not None:
            # batch_size, query_dim = query_mask.shape
            # key_dim = key.shape[1]

            query_block = query_mask <= 0

        #     #  (B*H, Tq, Tk) - repeat batch mask across heads
        #     cross_attn_mask_bt = query_block.unsqueeze(-1).expand(batch_size, query_dim, key_dim)
        #     cross_attn_mask = cross_attn_mask_bt.repeat_interleave(self.num_heads, dim=0)
        #
        #     self_attn_mask_bt = query_block.unsqueeze(-1).expand(batch_size, query_dim, query_dim)
        #     self_attn_mask = self_attn_mask_bt.repeat_interleave(self.num_heads, dim=0)
        # else:
        #     cross_attn_mask = None
        #     self_attn_mask = None

        # Study the effect of incorporating encoder outputs here
        out = torch.cat([theta_t, query, t], dim=-1)

        out = self.input_proj(out)

        # Run input through cross-attention layers
        for layer in self.layers:
            if isinstance(layer, SelfAttentionBlock):
                out = layer(query=out, attn_mask=None)
            else:
                out = layer(query=out, key=key, attn_mask=None)

            if query_mask is not None:
                out = out * query_mask[..., None]

        return self.output_proj(out)

    def forward(self, theta, query, key, query_mask=None):
        """Returns predicted and target velocity (potentially masked)."""

        # Generate time
        t = torch.rand(theta.shape[0], device=theta.device)
        t = broadcast_right(t, theta.shape)

        # Generate z = theta0
        z = torch.randn_like(theta)

        # Interpolation
        theta_t = t * theta + (1 - t) * z

        # Predict velocity v at theta_t
        pred_velocity = self.velocity(theta_t, query, key, t, query_mask)
        target_velocity = theta - z

        if query_mask is not None:
            pred_velocity = pred_velocity * query_mask[..., None]
            target_velocity = target_velocity * query_mask[..., None]

        return pred_velocity, target_velocity
