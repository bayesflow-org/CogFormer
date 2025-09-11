import keras
from keras.layers import Dense, GlobalAveragePooling1D

from collections.abc import Sequence

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable
from bayesflow.networks.transformers.mab import MultiHeadAttentionBlock
from bayesflow.networks.summary_network import SummaryNetwork

from ..encoders import PositionEncoder


@serializable("bayesflow.networks")
class Simformer(SummaryNetwork):
    """Transformer-based summary network for Simformer, using MultiHeadAttentionBlock.

    This network processes high-dimensional simulator outputs to produce compact summary
    representations for downstream diffusion-based inference, using a transformer architecture
    with multi-head attention blocks.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input simulator data.
    output_dim : int
        Dimensionality of the output summary representation.
    sequence_length : int, optional
        Expected sequence length of input data, by default 100.
    num_layers : int, optional
        Number of transformer encoder layers, by default 4.
    num_heads : int, optional
        Number of attention heads in multi-head attention, by default 8.
    hidden_dim : int, optional
        Hidden dimensionality of the transformer, by default 256.
    dropout : float, optional
        Dropout rate for regularization, by default 0.1.
    mlp_depth : int, optional
        Number of layers in the MLP within MultiHeadAttentionBlock, by default 2.
    mlp_width : int, optional
        Width of each MLP layer in MultiHeadAttentionBlock, by default 128.

    Attributes
    ----------
    input_dim : int
        Dimensionality of the input data.
    output_dim : int
        Dimensionality of the output summary.
    hidden_dim : int
        Hidden dimensionality of the transformer.
    input_projector : keras.layers.Dense
        Input projection layer for simulator data.
    position_encoder : PositionalEncoding
        Positional encoding layer for sequence context.
    transformer_layers : list
        List of MultiHeadAttentionBlock layers for transformer encoding.
    pooling : keras.layers.GlobalAveragePooling1D
        Adaptive pooling layer for variable-length sequences.
    output_projector : keras.layers.Dense
        Output projection layer for summary representation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim = 256,
        sequence_length = 100,
        num_layers = 2,
        num_heads = 8,
        dropout = 0.1,
        mlp_depth=2,
        mlp_width=128
    ):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input projector
        self.input_projector = Dense(hidden_dim, name="input_projection")

        # Position encoder
        self.position_encoder = PositionEncoder(
            hidden_dim = hidden_dim,
            dropout = dropout,
            max_length = sequence_length
        )

        # Sequential transformer layers
        self.transformer_layers = [
            MultiHeadAttentionBlock(
                embed_dim = hidden_dim,
                num_heads = num_heads,
                dropout = dropout,
                mlp_depth = mlp_depth,
                mlp_width = mlp_width,
                mlp_activation="gelu",
                kernel_initializer="glorot_uniform",
                layer_norm = True
            )
            for i in range(num_layers)
        ]

        # Adaptive pooling
        self.pooling = GlobalAveragePooling1D()

        # Output projector
        self.output_projector = Dense(output_dim, name="output_projection")


    def build(self, input_shape):
        """Builds the model and validates input dimension.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor, expected to be (batch_size, seq_len, input_dim).
        """
        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim {self.input_dim}, but got {input_shape[-1]}"
            )
        super().build(input_shape)

#
    def call(self, x, mask=None, training=False):
        """Forward pass for processing simulator outputs.

        Parameters
        ----------
        x : keras.KerasTensor
            Input tensor of shape (batch_size, seq_len, input_dim).
        mask : keras.KerasTensor, optional
            Attention mask for variable-length sequences, by default None.
        training : bool, optional
            Whether the model is in training mode, by default False.

        Returns
        -------
        keras.KerasTensor
            Summary representation of shape (batch_size, output_dim).
        """
        # Project simulator data
        x = self.input_projector(x)

        # Add positional encoding
        x = self.position_encoder(x)

        # Transformer encoding with MultiHeadAttentionBlock
        for mab in self.transformer_layers:
            x = mab(x, x, training=training, attention_mask=mask)

        # Adaptive pooling
        x = self.pooling(x)

        # Project to summary
        x = self.output_projector(x)

        return x
