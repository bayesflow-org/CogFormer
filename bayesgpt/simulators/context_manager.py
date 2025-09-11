import numpy as np
from typing import Dict, Tuple, Set
from collections.abc import Callable


class ContextManager:
    def __init__(
        self,
        parameter_names: list[str],
        fixed_parameters: list[str] = None
    ):
        self.parameter_names = parameter_names
        # self.fixed_parameters = fixed_parameters
        # if not all(p in self.parameter_names for p in self.fixed_parameters):
        #     raise ValueError("All fixed_parameters must be in parameter_names")
        self.mask = None  # Delay mask creation until dims are inferred

    def build_mask(self, fixed_parameters: list[str] = None) -> dict[str, float]:
        """
        Mask away the fixed parameters as 0.0
        """

        mask = {}
        for name in self.parameter_names:
            mask[name] = 0.0 if name in fixed_parameters else 1.0
        self.mask = mask
        return mask

    def apply_mask(self, sampled_params: np.ndarray | dict[str, float], mask: dict[str, float]) -> dict[str, float]:

        masked_params = {}

        if isinstance(sampled_params, dict):
            for key, value in sampled_params.items():
                masked_params[key] = (sampled_params[key] * self.mask[key]).astype(np.float32)
        elif isinstance(sampled_params, np.ndarray):
            for i in range(sampled_params.shape[0]):
                masked_params[i] = sampled_params[i] * self.mask[i]

        return masked_params

    # def sample(self) -> Dict[str, np.ndarray | float]:
    #     """Draw once per simulation"""
    #     out: Dict[str, np.ndarray | float] = {}
    #     for name in self.parameter_names:
    #         if name in self.fixed_parameters:
    #             continue
    #         dim = self.param_dims.get(name, 1)
    #         if dim == 1:
    #             out[name] = np.float32(np.random.randn())  # scalar, per-sim
    #         else:
    #             out[name] = np.random.randn(dim).astype(np.float32)  # (dim,), per-sim
    #     return out

    # def flex_combine(self, sampled: Dict[str, np.ndarray], overrides: Dict[str, float | np.ndarray] | None = None
    #                      ) -> Dict[str, np.ndarray]:
    #     """Treat keys in `overrides` as fixed for THIS call; others come from `sampled` (or fresh sample)."""
    #     out, overrides = {}, (overrides or {})
    #     num_samples = next(iter(sampled.values())).shape[0] if sampled else 1
    #
    #     for name in self.parameter_names:
    #         dim = self.param_dims.get(name, 1)
    #
    #         if name in overrides:
    #             val = np.asarray(overrides[name], dtype=np.float32)
    #
    #             # scalar -> broadcast to (N,) or (N, dim)
    #             if val.ndim == 0:
    #                 out[name] = (np.full(num_samples, float(val), np.float32)
    #                              if dim == 1 else np.full((num_samples, dim), float(val), np.float32))
    #                 continue
    #
    #             # (dim,) -> tile across trials
    #             if dim > 1 and val.ndim == 1 and val.shape[0] == dim:
    #                 out[name] = np.tile(val[None, :], (num_samples, 1)).astype(np.float32)
    #                 continue
    #
    #             # (N,) or (N,dim) exact match
    #             if (val.ndim == 1 and val.shape[0] == num_samples) or \
    #                     (val.ndim == 2 and val.shape == (num_samples, dim)):
    #                 out[name] = val.astype(np.float32, copy=False)
    #                 continue
    #
    #             raise ValueError(f"{name}: shape {val.shape} incompatible with num_samples={num_samples}, dim={dim}")
    #
    #         # not overridden -> use sampled if present, else sample fresh
    #         if name in sampled:
    #             out[name] = sampled[name]
    #         else:
    #             out[name] = (np.random.randn(num_samples, dim).astype(np.float32)
    #                          if dim > 1 else np.random.randn(num_samples).astype(np.float32))
    #     return out
    #
    # def combine(
    #         self,
    #         sampled: Dict[str, np.ndarray | float],
    #         fixed_parameters: Dict[str, float] | None = None,
    #         num_samples: int = 1,
    # ) -> Dict[str, np.ndarray]:
    #     out: Dict[str, np.ndarray] = {}
    #     fixed = fixed_parameters or {}
    #
    #     for name in self.parameter_names:
    #         dim = self.param_dims.get(name, 1)
    #
    #         if name in self.fixed_parameters:
    #             # fixed: use user value if provided, else neutral default (or 0.0)
    #             val = fixed.get(name, 0.0)
    #             val = np.asarray(val, dtype=np.float32)
    #         elif name in fixed:
    #             # user tried to override a free param (disallow if you want strictness)
    #             val = np.asarray(fixed[name], dtype=np.float32)
    #         else:
    #             # free: take per-simulation draw from sampled
    #             val = np.asarray(sampled.get(name, 0.0), dtype=np.float32)
    #
    #         # Broadcast rules
    #         if val.ndim == 0:
    #             out[name] = (np.full(num_samples, float(val), np.float32) if dim == 1
    #                          else np.full((num_samples, dim), float(val), np.float32))
    #         elif val.ndim == 1:
    #             if dim == 1:
    #                 # (1,) or (num_samples,) -> broadcast if length 1, else accept if matches
    #                 if val.size == 1:
    #                     out[name] = np.full(num_samples, float(val[0]), np.float32)
    #                 elif val.size == num_samples:
    #                     out[name] = val.astype(np.float32, copy=False)
    #                 else:
    #                     raise ValueError(f"{name}: expected length 1 or {num_samples}, got {val.shape}")
    #             else:
    #                 # (dim,) -> tile across trials
    #                 if val.size == dim:
    #                     out[name] = np.tile(val[None, :], (num_samples, 1)).astype(np.float32)
    #                 else:
    #                     raise ValueError(f"{name}: expected shape ({dim},), got {val.shape}")
    #         elif val.ndim == 2:
    #             # (num_samples, dim) exact
    #             if val.shape == (num_samples, dim):
    #                 out[name] = val.astype(np.float32, copy=False)
    #             else:
    #                 raise ValueError(f"{name}: expected shape ({num_samples},{dim}), got {val.shape}")
    #         else:
    #             raise ValueError(f"{name}: unsupported ndim={val.ndim}")
    #     return out
    #
    # def get_parameter_dims(self, name: str) -> int:
    #     return self.param_dims.get(name, 1)
    #
    # def normalize_params(self, params: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray | float]:
    #     """Model-agnostic: collapse constants to scalars; squeeze (N,1)->(N,)."""
    #     out = {}
    #     for k, v in params.items():
    #         arr = np.asarray(v)
    #         if arr.ndim == 0:
    #             out[k] = float(arr)
    #         elif arr.ndim == 1:
    #             # constant vector → scalar
    #             out[k] = float(arr[0]) if arr.size and np.all(arr == arr[0]) else arr
    #         elif arr.ndim == 2 and arr.shape[1] == 1:
    #             v1 = arr[:, 0]
    #             out[k] = float(v1[0]) if v1.size and np.all(v1 == v1[0]) else v1
    #         else:
    #             out[k] = arr
    #     return out
    #
    # def validate(self, params: Dict[str, np.ndarray | float], num_samples: int) -> None:
    #     """Light sanity checks (dims only); keep model-agnostic."""
    #     for k in self.parameter_names:
    #         if k not in params:
    #             raise ValueError(f"Missing parameter: {k}")
    #         v = params[k]
    #         if isinstance(v, np.ndarray):
    #             if v.ndim == 1 and v.shape[0] not in (1, num_samples):
    #                 raise ValueError(f"{k}: expected length {num_samples} (or scalar), got {v.shape}")
    #             if v.ndim > 1 and v.shape[0] != num_samples:
    #                 raise ValueError(f"{k}: leading dim must be {num_samples}, got {v.shape}")
