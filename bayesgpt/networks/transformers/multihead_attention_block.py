import torch
import torch.nn as nn


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead Attention Block using torch.nn.MultiheadAttention.

    Args:
        query_dim:  feature dim of Q input  (B, T_q, dim_Q)
        key_dim:  feature dim of K input  (B, T_k, dim_K)
        num_heads: number of attention heads
        layer_norm: whether to use pre-LayerNorm (pre-LN transformer style)
        ffn_dim: inner dim of the 2-layer FFN; defaults to 4 * query_dim
    """
    def __init__(
        self,
        query_dim: int,
        key_dim: int = None,
        num_heads: int = 4,
        layer_norm: bool = False,
        dropout: float = 0.0,
        output_dim: int = None,
        ffn_dim: int = None,
    ):
        super().__init__()

        if key_dim is None:
            key_dim = query_dim

        if output_dim is None:
            output_dim = query_dim

        if ffn_dim is None:
            ffn_dim = 4 * query_dim

        # Built-in MHA; batch_first=True to accept (B, T, C)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=key_dim,
            vdim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-LN norms (applied before each sublayer)
        self.layer_norm_0 = nn.LayerNorm(query_dim) if layer_norm else None
        self.layer_norm_1 = nn.LayerNorm(query_dim) if layer_norm else None

        # 2-layer FFN with 4x expansion
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, output_dim),
        )

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

        # Pre-LN attention sublayer
        normed = self.layer_norm_0(query) if self.layer_norm_0 is not None else query
        attn_out, _ = self.attn(
            query=normed,
            key=key,
            value=key,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = query + attn_out

        # Pre-LN FFN sublayer
        normed = self.layer_norm_1(out) if self.layer_norm_1 is not None else out
        out = out + self.ffn(normed)

        return out
