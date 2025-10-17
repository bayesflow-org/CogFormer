import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead Attention Block using torch.nn.MultiheadAttention.

    Args:
        query_dim:  feature dim of Q input  (B, T_q, dim_Q)
        key_dim:  feature dim of K input  (B, T_k, dim_K)
        num_heads: number of attention heads
        layer_norm: whether to use LayerNorm before/after the FFN residual
    """
    def __init__(
        self,
        query_dim: int,
        key_dim: int = None,
        num_heads: int = 4,
        layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        if key_dim is None:
            key_dim = query_dim

        # Built-in MHA; batch_first=True to accept (B, T, C)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=key_dim,
            vdim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm_0 = nn.LayerNorm(query_dim) if layer_norm else None
        self.layer_norm_1 = nn.LayerNorm(query_dim) if layer_norm else None

        self.fc_o = nn.Linear(query_dim, query_dim)

    def forward(self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Q: (B, T_q, dim_Q)
        K: (B, T_k, dim_K) — used for both keys and values
        attn_mask: optional (T_q, T_k) or (B*num_heads, T_q, T_k)
        key_padding_mask: optional (B, T_k), True for PAD positions
        """

        attn_out, _ = self.attn(
            query=query,
            key=key,
            value=key,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # 3) First residual (q + attention), optional LayerNorm
        out = query + attn_out
        if self.layer_norm_0 is not None:
            out = self.layer_norm_0(out)

        # 4) Simple FFN + residual (match original: ReLU then linear)
        out = out + F.gelu(self.fc_o(out))
        if self.layer_norm_1 is not None:
            out = self.layer_norm_1(out)

        return out
