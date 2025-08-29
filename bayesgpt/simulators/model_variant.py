import numpy as np
from typing import Union, Optional
from collections.abc import Mapping

from .model import Model
from ..utils.tokenizer import Tokenizer


class ModelVariant:
    """
    Encapsulates a model variant with free and fixed parameters,
    providing simulation and inference interface with parameter masking.

    Parameters
    ----------
    name : str
        Name or identifier for this variant.
    model : type
        A callable class implementing a `.simulate(params: dict, batch_size: int, num_samples: int)` method.
    tokenizer : Tokenizer
        Configuration object that manages sampling, value fixing, and masking of parameters.
    num_samples : int
        Number of samples to generate.
    """

    def __init__(
        self,
        name: str,
        model: type[Model],
        tokenizer: Tokenizer,
        num_samples: int
    ):
        self.name = name
        self.model: Model = model()
        self.tokenizer = tokenizer
        self.parameter_names = list(tokenizer.parameter_names)
        self.num_samples = num_samples

    def sample(
        self,
        batch_size: int,
        context: Optional[np.ndarray] = None
    ) -> dict[str, Union[np.ndarray, Mapping[str, np.ndarray]]]:
        """
        Simulate data and return full parameter vectors.

        Parameters
        ----------
        batch_size : int
            Number of simulations to run.
        context : np.ndarray, optional
            Context array to condition parameter sampling.

        Returns
        -------
        dict with keys:
        - "sim_data" : np.ndarray or Mapping[str, np.ndarray]
            Simulation outputs (array if model returns array, mapping if model returns mapping).
        - "full_params" : np.ndarray of shape (batch_size, num_parameters)
            Sampled set of parameters.
        - "inference_conditions" : np.ndarray
            Concatenated inference conditions (mask, base values, etc.).
        """
        # Sample and collect params
        sampled_parameters = self.tokenizer.sample(batch_size, context)
        params_dict = self.tokenizer.combine(sampled_parameters, batch_size)

        # Run simulator
        sim_data = self.model.simulate(params_dict, batch_size, self.num_samples)

        # Build full set of parameters
        base_values = self.tokenizer.get_base_values(batch_size=1)
        full_params = np.tile(base_values, (batch_size, 1)).astype(np.float32)

        for name in self.parameter_names:
            sl = self.tokenizer.parameter_slices[name]
            if self.tokenizer.mask[sl][0] == 1.0:  # Free parameters
                full_params[:, sl] = sampled_parameters[name].astype(np.float32)

        inference_conditions = self.tokenizer.build_inference_conditions(
            batch_size=batch_size,
            include_variant=False,
            include_context=False
        )

        return {
            "sim_data": sim_data,
            "full_params": full_params,
            "inference_conditions": inference_conditions["full_conditions"],
        }

    def get_mask(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated tri-state mask for parameter roles.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Tri-state mask where -1.0 indicates inactive, 0.0 fixed, and 1.0 free parameters.
        """
        return self.tokenizer.get_mask(batch_size)

    def get_base_values(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated conditioning vector with fixed/default values.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Conditioning vector with fixed/default values.
        """
        return self.tokenizer.get_base_values(batch_size)

    def build_inference_conditions(
        self,
        batch_size: int,
        *,
        one_hot_variant: np.ndarray = None,
        context: np.ndarray = None,
    ) -> dict[str, np.ndarray]:
        """
        Build inference conditions including optional variant/context data.

        Parameters
        ----------
        batch_size : int
            Number of rows to produce.
        one_hot_variant : np.ndarray, optional
            One-hot encoded variant identifier.
        context : np.ndarray, optional
            Context variables for conditioning.

        Returns
        -------
        dict with keys:
        - mask : np.ndarray of shape (batch_size, D)
        - base_values : np.ndarray of shape (batch_size, D)
        - variant : np.ndarray, optional
        - context : np.ndarray, optional
        - full_conditions : np.ndarray of shape (batch_size, 2*D + variant/context dims)
        """
        return self.tokenizer.build_inference_conditions(
            batch_size=batch_size,
            one_hot_variant=one_hot_variant,
            context=context,
            include_variant=one_hot_variant is not None,
            include_context=context is not None,
        )
