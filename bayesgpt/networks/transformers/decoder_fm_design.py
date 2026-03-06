import math

import torch
import torch.nn as nn

from .attention_layers import (
    self_attention_layers,
    mixed_attention_layers,
    custom_attention_layers
)
from .self_attention_block import SelfAttentionBlock
from bayesgpt.utils.tensor_utils import broadcast_right


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal embedding for a scalar input.

    Maps (..., 1) → (..., dim) using sin/cos at geometrically spaced frequencies.
    No learned parameters; negligible compute overhead.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalEmbedding dim must be even"
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device, dtype=x.dtype)
            / max(half - 1, 1)
        )
        args = x * freqs  # (..., half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., dim)


class Decoder(nn.Module):

    def __init__(
            self,
            input_dim: int = 2,
            seed_dim: int = 128,
            proj_dim: int = 64,
            num_layers: int = 3,
            num_heads: int = 4,
            layer_norm: bool = False,
            dropout: float = 0.05,
            layer_design: str = None,
            layer_kwargs: dict = None,
            time_embedding_dim: int = 32,
            pos_embedding_dim: int = 32,
            model_embed_dim: int = 0,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        if layer_kwargs is None:
            layer_kwargs = {}

        self.time_embedding = SinusoidalEmbedding(time_embedding_dim)
        self.pos_embedding = SinusoidalEmbedding(pos_embedding_dim)

        # input: theta_t (1) + param_embedding (pos_embedding_dim) + reg_embedding (pos_embedding_dim)
        #        + t_embedding (time_embedding_dim) [+ model_embedding (model_embed_dim)]
        self.input_proj = nn.Linear(1 + 2 * pos_embedding_dim + time_embedding_dim + model_embed_dim, proj_dim)

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

    def velocity(self, theta_t, query, key, t, query_mask=None, model_embedding=None):

        # key_padding_mask for self-attention: True = inactive token, should be ignored
        key_padding_mask = (query_mask <= 0) if query_mask is not None else None

        # Sinusoidal embeddings for positional indices and time
        param_embedding = self.pos_embedding(query[..., :1])         # (B, T, pos_embedding_dim)
        regressor_embedding = self.pos_embedding(query[..., 1:])     # (B, T, pos_embedding_dim)
        time_embedding = self.time_embedding(t)                      # (B, T, time_embedding_dim)

        components = [theta_t, param_embedding, regressor_embedding, time_embedding]
        if model_embedding is not None:
            components.append(model_embedding)                       # (B, T, model_embed_dim)
        out = torch.cat(components, dim=-1)
        out = self.input_proj(out)

        # Run through attention layers
        for layer in self.layers:
            if isinstance(layer, SelfAttentionBlock):
                out = layer(query=out, key_padding_mask=key_padding_mask)
            else:
                out = layer(query=out, key=key, attn_mask=None)

            if query_mask is not None:
                out = out * query_mask[..., None]

        return self.output_proj(out)

    def forward(self, theta, query, key, query_mask=None, model_embedding=None):
        """Returns predicted and target velocity (potentially masked)."""

        # Generate time
        t = torch.rand(theta.shape[0], device=theta.device)
        t = broadcast_right(t, theta.shape)

        # Generate z = theta0
        z = torch.randn_like(theta)

        # Interpolation
        theta_t = t * theta + (1 - t) * z

        # Predict velocity v at theta_t
        pred_velocity = self.velocity(theta_t, query, key, t, query_mask, model_embedding)
        target_velocity = theta - z

        if query_mask is not None:
            pred_velocity = pred_velocity * query_mask[..., None]
            target_velocity = target_velocity * query_mask[..., None]

        return pred_velocity, target_velocity
