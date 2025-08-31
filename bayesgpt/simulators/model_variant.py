import numpy as np
from typing import Union, Optional
from collections.abc import Mapping

from .model import Model
from .tokenizer import Tokenizer


class ModelVariant:
    """
    Encapsulates a model variant with free and fixed parameters for a single simulation.

    Handles tokenization and simulation for num_samples trials, integrating with
    NestedModelFamily, which manages batching.

    Parameters
    ----------
    name : str
        Name or identifier for this variant.
    model : type[Model]
        A callable class implementing `.simulate(params: dict, num_samples: int, context: Optional[np.ndarray])`.
    tokenizer : Tokenizer
        Manages parameter sampling, fixing, and masking.
    num_samples : int
        Number of samples (trials) per simulation.
    """

    def __init__(
        self,
        name: str,
        model: type[Model],
        tokenizer: Tokenizer,
        num_samples: int
    ):
        self.name = name
        self.model: Model = model()  # Instantiate the model
        self.tokenizer = tokenizer
        self.parameter_names = list(tokenizer.parameter_names)  # All parameter names
        self.num_samples = num_samples  # Trials per simulation

    def sample(
        self,
        context: Optional[np.ndarray] = None
    ) -> dict[str, Union[np.ndarray, Mapping[str, np.ndarray], str]]:
        """
        Simulate data for a single model run with num_samples trials.

        Parameters
        ----------
        context : np.ndarray, optional
            Context array to condition parameter sampling, shape (context_shape,).

        Returns
        -------
        dict[str, Union[np.ndarray, Mapping[str, np.ndarray], str]]
            Dictionary with keys:
            - 'sim_data': Simulation outputs, shape (num_samples, ...) or mapping with arrays of shape (num_samples, ...).
            - 'full_params': Sampled and fixed parameters, shape (num_parameters,).
            - 'inference_conditions': Concatenated inference conditions, shape (num_parameters,).
            - 'variant_name': Variant identifier (str).
        """

        # Sample parameters for a single simulation
        sampled_parameters = self.tokenizer.sample(context=context)
        params_dict = self.tokenizer.combine(sampled_parameters)  # Combine into model-compatible dictionary

        # Run simulation with num_samples trials
        sim_data = self.model.simulate(params_dict, num_samples=self.num_samples, context=context)

        # Build full parameter vector with fixed and sampled values
        base_values = self.tokenizer.get_base_values()  # Get fixed/default values
        full_params = base_values.copy()
        for name in self.parameter_names:
            sl = self.tokenizer.parameter_slices[name]
            if self.tokenizer.mask[sl][0] == 1.0:  # Free parameter
                full_params[sl] = sampled_parameters.get(name, np.random.randn(sl.stop - sl.start)).astype(np.float32)

        # Get inference conditions for this simulation
        inference_conditions = self.tokenizer.build_inference_conditions(
            context=context,
            include_variant=False,
            include_context=False
        )

        return {
            "sim_data": sim_data,
            "full_params": full_params,
            "inference_conditions": inference_conditions["full_conditions"],
            "variant_name": self.name
        }

    def get_mask(self) -> np.ndarray:
        """
        Return the tri-state mask for parameter roles.

        Returns
        -------
        np.ndarray
            Tri-state mask, shape (num_parameters,), where -1.0 is inactive, 0.0 is fixed,
            and 1.0 is free.
        """
        return self.tokenizer.mask  # Return tokenizer's mask

    def get_base_values(self) -> np.ndarray:
        """
        Return the vector of fixed/default parameter values.

        Returns
        -------
        np.ndarray
            Conditioning vector, shape (num_parameters,), with fixed/default values.
        """
        return self.tokenizer.base_values  # Return tokenizer's base values

    def build_inference_conditions(
        self,
        variant_encoder: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Build inference conditions with optional variant/context encoders.

        Parameters
        ----------
        variant_encoder : np.ndarray, optional
            One-hot encoded variant identifier, shape (num_variants,).
        context : np.ndarray, optional
            Context variables, shape (context_shape,).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys:
            - 'mask': Parameter mask, shape (num_parameters,).
            - 'base_values': Fixed/default values, shape (num_parameters,).
            - 'variant': Variant encoder (if included), shape (num_variants,).
            - 'context': Context encoder (if included), shape (context_shape,).
            - 'full_conditions': Concatenated conditions, shape (D,).

        Raises
        ------
        ValueError
            If context or variant encoder shapes are invalid.
        """
        return self.tokenizer.build_inference_conditions(
            one_hot_variant=variant_encoder,
            context=context,
            include_variant=variant_encoder is not None,
            include_context=context is not None
        )
