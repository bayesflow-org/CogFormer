import keras
from keras import ops
from keras.layers import Dropout

from bayesflow.utils.serialization import serializable


@serializable("bayesflow.networks")
class PositionalEncoder(keras.Layer):

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        max_length: int = 5000,
        **kwargs
    ):
        """
        Parameters
        ----------
        hidden_dim : int
            The dimensionality of the model (hidden dimension).
        dropout : float, optional
            Dropout rate for regularization, by default 0.1.
        max_len : int, optional
            Maximum sequence length for positional encodings, by default 5000.
        **kwargs
            Additional keyword arguments for the Keras Layer.

        Attributes
        ----------
        dropout : keras.layers.Dropout
            keras.layers.Dropout layer for regularization.
        pe : keras.backend.Variable
            Positional encoding tensor, shape (max_len, 1, d_model).
        """
        super(PositionalEncoder, self).__init__(**kwargs)
        self.dropout = Dropout(dropout)

        # Compute positional encodings using Keras ops
        position = ops.expand_dims(ops.arange(max_length, dtype="float32"), 1)
        div_term = ops.exp(
            ops.arange(0, hidden_dim, 2, dtype="float32") * (-ops.log(10000.0) / hidden_dim)
        )
        pe = ops.zeros([max_length, 1, hidden_dim], dtype="float32")
        pe = pe + ops.concatenate([ops.sin(position * div_term), ops.cos(position * div_term)], axis=-1)
        self.pe = self.add_weight(
            name="positional_encoding", shape=(max_length, 1, hidden_dim), initializer=lambda x: pe, trainable=False
        )


    def call(self, x):
        """Applies positional encoding to the input tensor.

        Parameters
        ----------
        x : keras.KerasTensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        keras.KerasTensor
            Output tensor with positional encodings added, same shape as input.
        """
        x = x + self.pe[: ops.shape(x)[1], :, :]
        return self.dropout(x)
