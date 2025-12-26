import torch.nn as nn
from .multihead_attention_block import MultiheadAttentionBlock


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

        self.input_proj = nn.Linear(input_dim, proj_dim)

        self.layers = nn.ModuleList(
            [MultiheadAttentionBlock(
                query_dim=proj_dim,
                key_dim=seed_dim,
                num_heads=num_heads,
                layer_norm=layer_norm,
                dropout=dropout
            ) for _ in range(num_layers)]
        )
        self.post_dropout = nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()

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

            q_block = query_mask == 0

            #  (B*H, Tq, Tk) - repeat batch mask across heads
            attn_mask_bt = q_block.unsqueeze(-1).expand(batch_size, query_dim, key_dim)
            attn_mask = attn_mask_bt.repeat_interleave(self.num_heads, dim=0)
        else:
            attn_mask = None


        for mab in self.layers:
            out = mab(query=out, key=key, attn_mask=attn_mask)
            out = self.post_dropout(out)

            if query_mask is not None:
                out = out * query_mask[..., None]

        mu, log_var = self.output_proj(out).chunk(2, dim=-1)

        if query_mask is not None:
            mu = mu * query_mask[..., None]
            log_var = log_var * query_mask[..., None]

        return mu, log_var