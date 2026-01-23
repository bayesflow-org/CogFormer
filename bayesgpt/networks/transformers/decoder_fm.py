import torch
import torch.nn as nn

from .multihead_attention_block import MultiheadAttentionBlock
from bayesgpt.utils.tensor_utils import broadcast_right


class Decoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        seed_dim: int = 128,
        proj_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        layer_norm: bool = False,
        dropout: float = 0.0,
        layer_dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.input_proj = nn.Linear(input_dim + 1 + 1, proj_dim)

        layers = []
        for i in range(num_layers):

            l = MultiheadAttentionBlock(
                query_dim=proj_dim,
                key_dim=seed_dim,
                num_heads=num_heads,
                layer_norm=layer_norm,
                dropout=dropout
            )

            layers.append(l)

        self.layers = nn.ModuleList(layers)
        self.post_dropout = nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()

        self.output_proj = nn.Linear(proj_dim, 1)
        self.num_heads = num_heads


    def velocity(self, theta_t, query, key, t, query_mask=None):
    
        # Ensure parameter mask is batched
        if query_mask is not None:
            batch_size, query_dim = query_mask.shape
            key_dim = key.shape[1]

            query_block = query_mask == 0

            #  (B*H, Tq, Tk) - repeat batch mask across heads
            attn_mask_bt = query_block.unsqueeze(-1).expand(batch_size, query_dim, key_dim)
            attn_mask = attn_mask_bt.repeat_interleave(self.num_heads, dim=0)
        else:
            attn_mask = None

        out = torch.cat([theta_t, query, t], dim=-1)

        out = self.input_proj(out)

        # Run input through cross-attention layers
        for mab in self.layers:
            
            out = mab(query=out, key=key, attn_mask=attn_mask)
            out = self.post_dropout(out)

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
