import numpy as np
from typing import Union, Optional
from collections.abc import Mapping

from .model import Model
from .context_manager import ContextManager


class NestedModelFamily:
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
    context_manager : context_manager
        Manages parameter sampling, fixing, and masking.
    num_samples : int
        Number of samples (trials) per simulation.
    """

    def __init__(
        self,
        name: str,
        model: type[Model],
        context_manager: ContextManager,
        num_samples: int
    ):
        self.name = name
        self.model: Model = model()  # Instantiate the model
        self.context_manager = context_manager
        self.parameter_names = list(context_manager.parameter_names)  # All parameter names
        self.num_samples = num_samples  # Number of samples per simulation

    def sample(
            self,
            num_samples: Optional[int] = None,
            context: Optional[np.ndarray] = None
    ) -> dict[str, Union[np.ndarray, Mapping[str, np.ndarray], str]]:
        """
        Simulate data for a single model run with num_samples trials.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples per simulation.
        context : np.ndarray, optional
            Context array to condition parameter sampling, shape (num_samples,).

        Returns
        -------
        dict[str, Union[np.ndarray, Mapping[str, np.ndarray], str]]
            Dictionary with keys:
            - 'sim_data': Simulation outputs, shape (num_samples, ...) or mapping with arrays of shape (num_samples, ...).
            - 'full_params': Sampled and fixed parameters, shape (num_parameters,).
            - 'inference_conditions': Concatenated inference conditions, shape (num_parameters,).
            - 'variant_name': Variant identifier (str).
            - 'sampled_parameters': Dictionary of sampled free parameters, mapping names to arrays of shape (num_samples, dims[p]).
        """
        num_samples = self.num_samples if num_samples is None else num_samples

        # Sample parameters for a single simulation
        sampled_parameters = self.context_manager.sample(context=context, num_samples=num_samples)
        params_dict = self.context_manager.combine(sampled_parameters)  # Combine into model-compatible dictionary

        # Ensure parameters are correctly shaped for SuperDDM
        for key, value in params_dict.items():
            param_dim = self.context_manager.get_parameter_dims(key)
            if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
                # Broadcast scalar or single-value parameters to (num_samples,)
                params_dict[key] = np.full(num_samples, np.asarray(value).item(), dtype=np.float32)
            elif isinstance(value, np.ndarray):
                # Ensure multidimensional parameters maintain their shape (e.g., v_components, v_schedule)
                if value.ndim == 1 and value.shape[0] == num_samples:
                    params_dict[key] = value.astype(np.float32)
                elif value.ndim == 2 and value.shape[0] == num_samples:
                    params_dict[key] = value.astype(np.float32)
                else:
                    # Reshape to (num_samples, param_dim) if needed
                    try:
                        params_dict[key] = value.reshape(num_samples, param_dim).astype(np.float32)
                    except ValueError:
                        raise ValueError(f"Parameter {key} has incompatible shape {value.shape} for num_samples={num_samples} and dim={param_dim}")

        # Run simulation with num_samples trials
        sim_data = self.model.simulate(params_dict, num_samples=num_samples, context=context)

        # Build full parameter vector with fixed and sampled values
        base_values = self.context_manager.base_values()  # Get fixed/default values
        full_params = base_values.copy()
        for name in self.parameter_names:
            sl = self.context_manager.parameter_slices[name]
            if self.context_manager.mask[sl][0] == 1.0:  # Free parameter
                param_val = sampled_parameters.get(name, np.random.randn(num_samples, sl.stop - sl.start))
                # Take mean across samples for multidimensional parameters to fit into full_params
                full_params[sl] = np.mean(param_val, axis=0).astype(np.float32) if param_val.ndim > 1 else param_val[0].astype(np.float32)
            # Fixed parameters are already in base_values

        # Get inference conditions for this simulation
        inference_conditions = self.context_manager.build_inference_conditions(
            context=context,
            include_variant=False,
            include_context=False
        )

        return {
            "sim_data": sim_data,
            "full_params": full_params,
            "inference_conditions": inference_conditions["full_conditions"],
            "variant_name": self.name,
            "sampled_parameters": sampled_parameters
        }

    @property
    def mask(self) -> np.ndarray:
        """
        Return the tri-state mask for parameter roles.

        Returns
        -------
        np.ndarray
            Tri-state mask, shape (num_parameters,), where -1.0 is inactive, 0.0 is fixed,
            and 1.0 is free.
        """
        return self.context_manager.mask

    @property
    def base_values(self) -> np.ndarray:
        """
        Return the vector of fixed/default parameter values.

        Returns
        -------
        np.ndarray
            Conditioning vector, shape (num_parameters,), with fixed/default values.
        """
        return self.context_manager.base_values

    def build_inference_conditions(
        self,
        one_hot_variant: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Build inference conditions with optional variant/context encoders.

        Parameters
        ----------
        one_hot_variant : np.ndarray, optional
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
        return self.context_manager.build_inference_conditions(
            one_hot_variant=one_hot_variant,
            context=context,
            include_variant=one_hot_variant is not None,
            include_context=context is not None
        )
