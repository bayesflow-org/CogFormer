import keras
from keras import ops, layers, Model
from typing import Optional, Tuple


class Simformer(Model):
    """
    A Keras 3 implementation of a Simformer-like architecture for
    simulation-based inference.

    This architecture is implemented in Keras, and handles joint diffusion
    on parameters (theta) and data (x) with flexible conditioning.

    Assumptions:
    -   sim_data is a np.ndarray of shape (batch_size, data_dim).
        If a Mapping, preprocess to flat array.
    -   M_C = 1 for conditioned variables (fixed/no noise),
        0 for variables to sample (add noise).
    -   infer_mask from data is inverted to M_C for params (M_C = 1 - infer_mask).
    -   x is typically conditioned (M_C_x = 1 for posterior), but flexible.
    -   inference_conditions are embedded as an extra token.

    Parameters
    ----------
    num_params : int
        Number of parameters in the global schema.
    data_dim : int
        Dimensionality of the simulation data (x).
    condition_dim : int
        Dimensionality of additional inference conditions (e.g., full_conditions.shape[-1]).
    embed_dim : int, optional
        Embedding dimension for id, value, and condition.
    num_layers : int, optional
        Number of transformer layers.
    num_heads : int, optional
        Number of attention heads.
    num_diffusion_steps : int, optional
        Number of diffusion steps for training and sampling.
    """

    def __init__(
        self,
        num_params: int,
        data_dim: int,
        condition_dim: int,
        embed_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 4,
        num_diffusion_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_params = num_params
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.embed_dim = embed_dim
        self.token_dim = 3 * embed_dim  # cat(id, value, cond)
        self.num_vars = num_params + 1  # params + x
        self.num_extra = 1 if condition_dim > 0 else 0
        self.total_tokens = self.num_vars + self.num_extra
        self.num_diffusion_steps = num_diffusion_steps

        # Embeddings
        self.id_embedding = layers.Embedding(
            self.total_tokens, embed_dim, name="id_embedding"
        )
        self.condition_embedding = layers.Embedding(
            2, embed_dim, name="condition_embedding"
        )
        self.value_linear_layers = [
            layers.Dense(embed_dim, name=f"value_linear_layer_{i}")
            for i in range(num_params)
        ] + [
            layers.Dense(
                embed_dim, name="value_linear_layer_x", input_shape=(data_dim,)
            )
        ]
        if condition_dim > 0:
            self.conditional_linear_layer = layers.Dense(
                embed_dim, name="conditional_linear_layer", input_shape=(condition_dim,)
            )
        else:
            self.conditional_linear_layer = None

        # Time embedding for diffusion
        self.time_embedding = keras.Sequential(
            [
                layers.Dense(embed_dim, input_shape=(1,), name="time_dense1"),
                layers.Activation("silu", name="time_silu"),
                layers.Dense(embed_dim, name="time_dense2"),
            ],
            name="time_embedding",
        )

        # Transformer
        transformer_layers = [
            layers.TransformerEncoderLayer(
                d_model=self.token_dim,
                num_heads=num_heads,
                dff=self.token_dim * 4,
                dropout=0.1,
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.transformer = keras.Sequential(transformer_layers, name="transformer")

        # Score projectors
        self.score_linear_layers = [
            layers.Dense(1, name=f"score_linear_{i}") for i in range(num_params)
        ] + [layers.Dense(data_dim, name="score_linear_x")]

        # Diffusion schedule (variance preserving)
        betas = ops.linspace(1e-4, 0.02, num_diffusion_steps)
        self.alphas = 1.0 - betas
        self.alpha_bars = ops.cumprod(self.alphas, axis=0)

        # Define model inputs and outputs
        z_shape = (None, num_params + data_dim)
        t_shape = (None,)
        M_C_shape = (None, self.num_vars)
        cond_shape = (None, condition_dim) if condition_dim > 0 else None
        inputs = [
            keras.Input(shape=z_shape, name="z"),
            keras.Input(shape=t_shape, name="t"),
            keras.Input(shape=M_C_shape, name="M_C"),
            keras.Input(shape=cond_shape, name="cond") if cond_shape else None,
        ]
        inputs = [i for i in inputs if i is not None]
        outputs = self.call(inputs, training=False)
        super().__init__(inputs=inputs, outputs=outputs)

    def call(
        self,
        inputs: Tuple[
            keras.KerasTensor,
            keras.KerasTensor,
            keras.KerasTensor,
            Optional[keras.KerasTensor],
        ],
        training: bool = False,
    ) -> keras.KerasTensor:
        """
        Predicts the score for the noised joint z = cat(theta, x.flatten(-1)).

        Parameters
        ----------
        inputs : Tuple of (z, t, M_C, cond):
            - z: Tensor (batch_size, num_params + data_dim), noised joint vector.
            - t: Tensor (batch_size,), diffusion timesteps.
            - M_C: Tensor (batch_size, num_vars), condition mask (1 = conditioned, 0 = to sample).
            - cond: Tensor (batch_size, condition_dim), optional inference conditions.

        Returns
        -------
        KerasTensor (batch_size, num_params + data_dim)
            Predicted score.
        """
        z, t, M_C, condition = inputs
        batch_size = ops.shape(z)[0]
        tokens = []
        pos = 0

        for i in range(self.num_vars):
            if i < self.num_params:
                dim_v = 1
            else:
                dim_v = self.data_dim
            value = z[:, pos : pos + dim_v]
            value_embedding = self.value_linear_layers[i](value)
            id_embedding = self.id_embedding(ops.full((batch_size,), i, dtype="int32"))
            condition_embedding = self.condition_embedding(ops.cast(M_C[:, i], "int32"))
            token = ops.concatenate(
                [id_embedding, value_embedding, condition_embedding], axis=-1
            )
            tokens.append(token)
            pos += dim_v

        # Add condition token if present
        if self.conditional_linear_layer is not None and condition is not None:
            conditional_value_embedding = self.conditional_linear_layer(condition)
            conditional_id_embedding = self.id_embedding(
                ops.full((batch_size,), self.num_vars, dtype="int32")
            )
            conditional_condition_embedding = self.condition_embedding(
                ops.ones((batch_size,), dtype="int32")
            )
            cond_token = ops.concatenate(
                [
                    conditional_id_embedding,
                    conditional_value_embedding,
                    conditional_condition_embedding,
                ],
                axis=-1,
            )
            tokens.append(cond_token)

        tokens = ops.stack(tokens, axis=1)  # (batch_size, total_tokens, token_dim)

        # Add time embedding (broadcast to all tokens)
        time_embedding = self.time_embedding(
            ops.expand_dims(t, -1)
        )  # (batch_size, embed_dim)
        tokens = tokens + ops.expand_dims(time_embedding, 1)  # Broadcast

        out = self.transformer(
            tokens, training=training
        )  # (batch_size, total_tokens, token_dim)

        # Extract scores for variables (ignore extra cond token)
        scores = []
        for i in range(self.num_vars):
            out_i = out[:, i]
            score_i = self.score_linear_layers[i](out_i)
            scores.append(score_i)
        score = ops.concatenate(scores, axis=-1)
        return score
