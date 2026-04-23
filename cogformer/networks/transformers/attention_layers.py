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
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
    query_dim: int = 128,
    skip_first: bool = False,
    skip_last: bool = False,
):
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
            input_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
        )
    layers.append(l)

    for _ in range(1, num_layers - 1):
        layers.append(
            SelfAttentionBlock(
                input_dim=query_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm=layer_norm,
            )
        )

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
            input_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
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
    layers = []
    for i in range(num_layers):
        use_mab = (i % 2 == 0) if mab_first else (i % 2 == 1)
        if use_mab:
            layers.append(
                MultiheadAttentionBlock(
                    query_dim=query_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
            )
        else:
            layers.append(
                SelfAttentionBlock(
                    input_dim=query_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
            )
    return nn.ModuleList(layers)


def custom_attention_layers(
    pattern: tuple = (1, 0, 0),
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
):
    design = [pattern[i % len(pattern)] for i in range(num_layers)]
    layers = []
    for i in range(num_layers):
        if design[i] == 1:
            layers.append(
                MultiheadAttentionBlock(
                    query_dim=query_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
            )
        else:
            layers.append(
                SelfAttentionBlock(
                    input_dim=query_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
            )
    return nn.ModuleList(layers)

def custom_attention_layers(
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

    design = [pattern[i % len(pattern)] for i in range(num_layers)]
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
