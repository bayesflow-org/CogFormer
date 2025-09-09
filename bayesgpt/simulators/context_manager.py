import numpy as np
from typing import Dict, Tuple, Set


class ContextManager:
    def __init__(self, parameter_names: list[str], fixed_parameters: Set[str] = None):
        self.parameter_names = list(parameter_names)
        self.fixed_parameters = set(fixed_parameters or [])
        if not all(p in self.parameter_names for p in self.fixed_parameters):
            raise ValueError("All fixed_parameters must be in parameter_names")
        self.param_dims = {}
        self.param_index_slices = None
        self.param_vector_size = None
        self.mask = None  # Delay mask creation until dims are inferred

    def _build_parameter_slices(self) -> Tuple[Dict[str, slice], int]:
        param_index_slices = {}
        param_vector_size = 0
        for name in self.parameter_names:
            dim = self.param_dims.get(name, 1)  # Default to 1 if not inferred
            param_index_slices[name] = slice(param_vector_size, param_vector_size + dim)
            param_vector_size += dim
        return param_index_slices, param_vector_size

    def _build_mask(self) -> np.ndarray:
        mask = np.ones(self.param_vector_size, dtype=np.float32)
        for name in self.fixed_parameters:
            sl = self.param_index_slices[name]
            mask[sl] = 0.0
        return mask

    def generate_regressors(self, params: Dict[str, np.ndarray], num_samples: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        regressors = {}
        regressed_parameters = {}
        for param_name in self.parameter_names:
            if param_name not in params or param_name in self.fixed_parameters:
                continue
            param_vector = params[param_name]
            dim = self.param_dims.get(param_name, 1)
            if param_vector.shape[0] == 1:  # Scalar case
                regressed_parameters[param_name] = np.full(num_samples, param_vector[0], dtype=np.float32)
                regressors[param_name] = np.ones((num_samples, 1), dtype=np.float32)
            else:  # Regression case
                if param_vector.shape[-1] != dim:
                    raise ValueError(f"Expected {param_name} dim {dim}, got {param_vector.shape[0]}")
                design_mat = np.c_[np.ones((num_samples, 1)), np.random.rand(num_samples, dim - 1)]
                regressed_parameters[param_name] = design_mat @ param_vector
                regressors[param_name] = design_mat.astype(np.float32)
        return regressors, regressed_parameters

    def sample(self, num_samples: int) -> Dict[str, np.ndarray]:
        out = {}
        for name in self.parameter_names:
            if name in self.fixed_parameters:
                continue
            dim = self.param_dims.get(name, 1)  # Use inferred or default dim
            out[name] = np.random.randn(num_samples, dim).astype(np.float32)
        return out

    def combine(self, sampled: Dict[str, np.ndarray], fixed_parameters: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        out = {}
        fixed = fixed_parameters or {}
        num_samples = next(iter(sampled.values())).shape[0] if sampled else 1
        for name in self.parameter_names:
            dim = self.param_dims.get(name, 1)
            if name in self.fixed_parameters:
                if name not in fixed:
                    raise ValueError(f"Fixed parameter {name} requires a value in fixed_parameters")
                out[name] = np.full(num_samples, fixed[name], dtype=np.float32)
            elif name in fixed:
                out[name] = np.full(num_samples, fixed[name], dtype=np.float32)
            else:
                out[name] = sampled.get(name, np.random.randn(num_samples, dim).astype(np.float32))
        return out

    def get_parameter_dims(self, name: str) -> int:
        return self.param_dims.get(name, 1)
