import numpy as np
from collections.abc import Callable


class Tokenizer:
    """
        Manages parameter sampling and simulation input construction
        for a model variant using a shared global parameter schema.

        Each parameter in the global schema can be:
        - Sampled (free): if defined in `parameter_samplers`
        - Fixed: if defined in `fixed_values`
        - Defaulted: if unspecified (filled with `fallback_value`)

        Parameters
        ----------
        parameter_names : list of str
            Ordered list of all parameter names in the global schema.

        free_parameters : dict of {str: Callable[[int], np.ndarray]}
            Mapping of parameter names to sampling functions for free parameters.

        fixed_parameters : dict of {str: float}
            Mapping of parameter names to fixed values.

        fallback_value : float, optional
            Value used for parameters that are neither free nor fixed.
            Defaults to 0.0.

        Raises
        ------
        ValueError
            If any parameter is defined as both free and fixed.
        """

    def __init__(
        self,
        parameter_names: list[str],
        free_parameters: dict[str, Callable[[int], np.ndarray]],
        fixed_parameters: dict[str, float],
        fallback_value: float = 0.0
    ):
        self.parameter_names = list(parameter_names)
        self.free_parameters = free_parameters
        self.fixed_parameters = fixed_parameters
        self.fallback_value = fallback_value

        # Sanity check: no parameters should be both fixed and free
        overlap = set(free_parameters) & set(fixed_parameters)
        if overlap:
            raise ValueError(f"The following parameters cannot be both fixed and free: {overlap}")

        for name, sampler in self.free_parameters.items():
            if not callable(sampler):
                raise ValueError(f"The parameter '{name}' must be a callable")

        # Binary infer masks for the parameters (1.0 for free parameters, 0.0 otherwise)
        self.infer_mask = np.array(
            [1.0 if name in free_parameters else 0.0 for name in self.parameter_names],
            dtype=np.float32
        )

        self.active_mask = np.array(
            [1.0 if (n in self.free_parameters or n in self.fixed_parameters) else 0.0
             for n in self.parameter_names],
            dtype=np.float32,
        )

        # Add conditioning so that fixed parameters get default values
        # if otherwise undefined.
        self.base_values = np.array([
            [fixed_parameters.get(name, fallback_value) for name in self.parameter_names]
        ], dtype=np.float32)

        self.num_parameters = len(self.parameter_names)


    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Samples a batch of values for the free parameters.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate for each parameter.

        Returns
        -------
        dict of {str: np.ndarray}
            A dictionary mapping free parameter names to sampled values.
            Each array has shape (batch_size,).
        """
        return {
            name: sampler(batch_size)
            for name, sampler in self.free_parameters.items()
        }


    def combine(
        self,
        sample_parameters: dict[str, np.ndarray],
        index: int
    ) -> dict[str, float]:
        """
        Combines sampled parameters along with the fixed and default parameters
        into a full dictionary for a single simulation instance.

        Parameters
        ----------
        sample_parameters : dict of {str: np.ndarray}
            A batch of sampled free parameters. Each array should have shape (batch_size,).

        index : int
            The index of the sample to extract from the batch.

        Returns
        -------
        dict of {str: float}
            A dictionary containing the full parameter set for a single simulation.
        """

        full_parameters = {}

        for i, name in enumerate(self.parameter_names):
            if self.active_mask[i] == 0.0:
                continue
            if self.infer_mask[i] == 1.0:
                full_parameters[name] = float(sample_parameters[name][index])
            elif name in self.fixed_parameters:
                full_parameters[name] = self.fixed_parameters[name]
            else:
                full_parameters[name] = self.fallback_value

        return full_parameters


    def build_inference_conditions(
            self,
            batch_size: int,
            *,
            include_variant: bool = True,
            include_context: bool = True,
            variant_encoder: np.ndarray = None,
            context_encoder: np.ndarray = None
    ):
        """
        Builds an inference condition using a shared global parameter schema.

        Parameters
        ----------
        batch_size      : int
            The number of samples to generate for each parameter.
        include_variant : bool, optional, default: True
            Whether to include variant one-hot encoder inference condition.
        include_context : bool, optional, default: True
            Whether to include context encoder inference condition.
        variant_encoder : np.ndarray, optional, default: None
            The one-hot encoded variant encoder.
        context_encoder : np.ndarray, optional, default: None
            The context variables.

        Returns
        -------
        A dictionary consisting of all components of the inference conditions:
        - infer_mask: binary infer_mask for free and fixed parameters
        - fixed_defaults: fixed default values for the fixed parameters
        - variant: one-hot encoded variant encoder
        - context: context variables
        - full_embeddings: all of the above concatenated for training purposes.
        """

        # Batch infer_mask and conditions
        batched_infer_mask  = np.tile(self.infer_mask, (batch_size, 1)).astype(np.float32)
        batched_active_mask = np.tile(self.active_mask, (batch_size, 1)).astype(np.float32)
        batched_base_values = np.tile(self.base_values, (batch_size, 1)).astype(np.float32)

        # Make a list of inference conditions to be concatenated as embeddings
        full_conditions = [batched_infer_mask, batched_active_mask, batched_base_values]
        inference_conditions = {
            "infer_mask": batched_infer_mask,
            "active_mask": batched_active_mask,
            "base_values": batched_base_values,
        }

        if include_variant:
            if variant_encoder is not None:
                inference_conditions["variant"] = variant_encoder.astype(np.float32)
                full_conditions.append(variant_encoder)

        if include_context:
            if context_encoder is not None:
                inference_conditions["context"] = context_encoder.astype(np.float32)
                full_conditions.append(context_encoder)

        inference_conditions["full_conditions"] = np.concatenate(full_conditions, axis=1)
        return inference_conditions


    def get_infer_mask(self) -> np.ndarray:
        """
        Returns the infer_mask array for the free parameters.

        Returns
        -------
        np.ndarray of shape (num_parameters,)
            Binary infer_mask indicating which parameters are free (and therefore learnable).
        """
        return self.infer_mask


    def get_active_mask(self) -> np.ndarray:
        """
        Returns the infer_mask array for the free parameters.

        Returns
        -------
        np.ndarray of shape (num_parameters,)
            Binary infer_mask indicating which parameters are free (and therefore learnable).
        """
        return self.active_mask


    def get_base_values(self) -> np.ndarray:
        """
        Returns the conditioning vector for the free parameters.

        Returns
        -------
        np.ndarray of shape (num_parameters,)
            Conditioning vector for the fixed and default values for all parameters.
            This vector aligns with the global parameter schema.
        """
        return self.base_values
