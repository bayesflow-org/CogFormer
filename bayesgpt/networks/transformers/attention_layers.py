from torch import nn
from .multihead_attention_block import MultiheadAttentionBlock
from .self_attention_block import SelfAttentionBlock


def cross_attention_layers(
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
):

    layers = []
    for i in range(num_layers):
        l = MultiheadAttentionBlock(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        layers.append(l)
    return nn.ModuleList(layers)


def self_attention_layers(
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
    skip_first: bool = False,
    skip_last: bool = False
):
    """
    Customizable self-attention layers with the option
    to use mab on the first and/or the last layer.
    """
    layers = []

    if skip_first:
        l = MultiheadAttentionBlock(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
        )
    else:
        l = SelfAttentionBlock(
            input_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm
        )
    layers.append(l)

    for i in range(1, num_layers - 1):
        l = SelfAttentionBlock(
            input_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        layers.append(l)

    if skip_last:
        l = MultiheadAttentionBlock(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
        )
    else:
        l = SelfAttentionBlock(
            input_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm
        )
    layers.append(l)

    return nn.ModuleList(layers)

def mixed_attention_layers(
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
    mab_first: bool = True,
):
    """
    Customized attention layers with the simultaneous use of
    self-attention (sab) and cross-attention blocks (mab).

    By default, the blocks are arranged as an interlaced
    undulation with mab first:

    mab --> sab --> mab --> sab --> mab --> ...

    The blocks can be arranged with sab comes first
    with mab_first == False.
    """
    cross_attention_block = MultiheadAttentionBlock(
        query_dim=query_dim,
        key_dim=key_dim,
        num_heads=num_heads,
        dropout=dropout,
        layer_norm=layer_norm
    )
    self_attention_block = SelfAttentionBlock(
        input_dim=key_dim,
        num_heads=num_heads,
        dropout=dropout,
        layer_norm=layer_norm
    )

    layers = []
    for i in range(num_layers):
        if i % 2 == 0:
            l = cross_attention_block if mab_first else self_attention_block
        else:
            l = self_attention_block if mab_first else cross_attention_block
        layers.append(l)

    return nn.ModuleList(layers)

def patterned_attention_layers(
    pattern: tuple = (1, 0, 0),
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
):
    """
    Customizable patterned attention layers with user-specified pattern
    indicating whether a block is mab (1) or sab (0).
    """
    if len(pattern) > num_layers:
        print("Pattern truncated to match with number of attention layers")

    design = [pattern[i % num_layers] for i in range(num_layers)]
    layers = []
    for i in range(num_layers):
        if design[i] == 1:
            l = MultiheadAttentionBlock(
                query_dim=query_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm=layer_norm
            )
        else:
            l = SelfAttentionBlock(
                input_dim=key_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm=layer_norm
            )
        layers.append(l)

    return nn.ModuleList(layers)
