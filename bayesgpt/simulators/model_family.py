import numpy as np
from typing import Dict, Optional, Union
from .model import Model
from .context_manager import ContextManager

class NestedModelFamily:
    def __init__(self, name: str, model: type[Model], context_manager: ContextManager, num_samples: int):
        # Initialize with model name, model class, context manager, and number of samples
        self.name = name
        self.model = model(context_manager)  # Instantiate model with context manager
        self.context_manager = context_manager
        self.parameter_names = context_manager.parameter_names
        self.num_samples = num_samples

    def sample(
        self,
        num_samples: Optional[int] = None,
        params: Optional[Dict[str, Union[np.ndarray, float]]] = None
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray], str]]:
        # Simulate data for a single model run
        num_samples = self.num_samples if num_samples is None else num_samples
        params = params or {}

        # Convert float params to np.ndarray for generate_regressors
        params_array = {k: np.array([v], dtype=np.float32) if isinstance(v, (int, float)) else v for k, v in params.items()}

        # Sample free parameters
        sampled_parameters = self.context_manager.sample(num_samples)

        # Generate regressed parameters for provided params (non-fixed only)
        regressors, regressed_params = self.context_manager.generate_regressors(params_array, num_samples)

        # Use float values for fixed parameters
        fixed_parameters = {k: float(v) if isinstance(v, (int, float)) else v[0] for k, v in params.items()}
        params_dict = self.context_manager.combine(sampled_parameters, fixed_parameters)
        params_dict.update(regressed_params)  # Override with regressed parameters

        # Run simulation
        sim_data = self.model.simulate(params_dict, num_samples=num_samples)

        # Build full parameter vector for inference
        full_params = np.zeros(self.context_manager.param_vector_size, dtype=np.float32)
        for name in self.parameter_names:
            sl = self.context_manager.param_index_slices[name]
            param_val = params_dict[name]
            full_params[sl] = np.mean(param_val, axis=0) if param_val.ndim > 1 else param_val[0]

        return {
            "sim_data": sim_data,
            "full_params": full_params,
            "sampled_parameters": sampled_parameters,
            "variant_name": self.name
        }