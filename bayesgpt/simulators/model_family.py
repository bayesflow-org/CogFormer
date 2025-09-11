import numpy as np
from typing import Dict, Optional, Union
from collections.abc import Callable, Iterable
from .model import Model
from .context_manager import ContextManager
from utils.simulator_utils import generate_regressors

class NestedModelFamily:
    def __init__(
        self,
        name: str,
        model: type[Model],
        context_manager: ContextManager,
        prior_fun: Callable,
        num_samples: int = 10
    ):
        # Initialize with model name, model class, context manager, and number of samples
        self.name = name
        self.model = model()  # Instantiate model with context manager
        self.context_manager = context_manager
        self.parameter_names = context_manager.parameter_names
        self.num_samples = num_samples
        self.prior_fun = prior_fun

    # def sample_old(
    #     self,
    #     num_samples: Optional[int] = None,
    #     params: Optional[Dict[str, Union[np.ndarray, float]]] = None,
    #     context: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
    # ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray], str]]:
    #     # Simulate data for a single model run
    #     num_samples = self.num_samples if num_samples is None else num_samples
    #     params = params or {}
    #
    #     if self.context_manager.param_index_slices is None:
    #         self.context_manager.build_layout()
    #
    #
    #     # Convert float params to np.ndarray for generate_regressors
    #     params_array = {k: np.array([v], dtype=np.float32) if isinstance(v, (int, float)) else v for k, v in params.items()}
    #
    #     # Sample free parameters
    #     sampled_parameters = self.context_manager.sample(num_samples)
    #
    #     # Generate regressed parameters for provided params (non-fixed only)
    #     regressors, regressed_params = generate_regressors(
    #         params_array,
    #         num_samples,
    #         self.context_manager.param_dims,
    #         self.context_manager.fixed_parameters
    #     )
    #
    #     # Use float values for fixed parameters
    #     fixed_parameters = {k: float(v) if isinstance(v, (int, float)) else v[0] for k, v in params.items()}
    #     params_dict = self.context_manager.combine(sampled_parameters, fixed_parameters, num_samples=num_samples)
    #     params_dict.update(regressed_params)  # Override with regressed parameters
    #
    #     params_dict = self.context_manager.normalize_params(params_dict)
    #     self.context_manager.validate(params_dict, num_samples)
    #
    #     params_dict = self.model.prepare_params(params_dict, num_samples)
    #
    #     # Run simulation
    #     sim_data = self.model.simulate(params_dict, num_samples=num_samples, context=context)
    #
    #     # Build full parameter vector for inference
    #     full_params = np.zeros(self.context_manager.param_vector_size, dtype=np.float32)
    #
    #
    #     for name in self.parameter_names:
    #         sl = self.context_manager.param_index_slices[name]
    #         val = np.asarray(params_dict[name])
    #         if val.ndim == 0:
    #             full_params[sl] = float(val)
    #         elif val.ndim == 1:
    #             full_params[sl] = np.array([val.mean()], dtype=np.float32)
    #         else:
    #             full_params[sl] = np.mean(val, axis=0, dtype=np.float32)
    #
    #     return {
    #         "sim_data": sim_data,
    #         "full_params": full_params,
    #         "sampled_parameters": sampled_parameters,
    #         "variant_name": self.name
    #     }

    def sample(
            self,
            fixed_configs: Iterable[Iterable[str]],
            num_samples: Optional[int] = None,
            context: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ):
        """
        For each config (list/set of parameter names to fix), build a mask, apply it to a
        single prior draw, map to model parameters, and simulate once. Returns a list.
        """
        num_samples = self.num_samples if num_samples is None else num_samples
        results = []

        for idx, fixed_params in enumerate(fixed_configs):
            # Build mask for this config (aligned with parameter_names)
            masks = self.context_manager.build_mask(list(fixed_params))  # shape (P,)

            # Draw prior once and apply mask (masked entries -> 0.0)
            priors = self._draw_prior_vec()  # shape (P,)
            priors_dict = self._prior_vec_to_dict(priors)

            masked_priors = self.context_manager.apply_mask(priors_dict, masks)  # shape (P,)

            # Convert masked vector to dict expected by the model (ordered by parameter_names)
            # params_dict = {name: float(masked_priors[i]) for i, name in enumerate(self.parameter_names)}

            # 4) Model path: prepare → simulate
            params_for_model = self.model.prepare_params(masked_priors, num_samples)
            sim_data = self.model.simulate(params_for_model, num_samples=num_samples, context=context)

            # 5) Package results for this config
            results.append(
                {
                    "variant_name": f"{self.name}|cfg{idx + 1}",
                    "fixed_parameters": list(fixed_params),
                    "mask": masks,
                    "prior_draw": priors.astype(np.float32, copy=False),
                    "full_params": masked_priors,
                    "sim_data": sim_data,
                }
            )

        return results

    def _draw_prior_vec(self) -> np.ndarray:
        """Draw once from the jitted prior; ensure shape matches parameter_names."""
        arr = np.asarray(self.prior_fun(), dtype=np.float32).ravel()
        if arr.size != len(self.parameter_names):
            raise ValueError(
                f"Prior length {arr.size} != #params {len(self.parameter_names)} ({len(self.parameter_names)})."
            )
        return arr

    def _draw_prior_dict(self) -> dict[str, float]:
        """Call prior_fn, coerce to plain dict, validate keys, and reorder by parameter_names."""
        raw = self.prior_fun()
        try:
            raw = dict(raw)  # supports numba typed.Dict
        except Exception as e:
            raise TypeError("prior_fun must return a dict-like {param: value}.") from e
        if set(raw.keys()) != set(self.parameter_names):
            raise ValueError("Prior dict keys must match parameter_names exactly.")
        # Reorder and cast to float
        return {name: float(raw[name]) for name in self.parameter_names}

    def _dict_to_vector(self, d: dict[str, float]) -> np.ndarray:
        return np.array([float(d[name]) for name in self.parameter_names], dtype=np.float32)

    def _mask_to_vector(self, mask_any) -> np.ndarray:
        # Accepts dict {name:0/1} or 1-D array/list
        if isinstance(mask_any, dict):
            return np.array([float(mask_any[n]) for n in self.parameter_names], dtype=np.float32)
        m = np.asarray(mask_any, dtype=np.float32).ravel()
        if m.size != len(self.parameter_names):
            raise ValueError(f"Mask length {m.size} != #params {len(self.parameter_names)}.")
        return (m > 0.5).astype(np.float32)

    def _prior_vec_to_dict(self, prior_vec: np.ndarray) -> dict[str, float]:
        """
        Convert a 1-D prior vector (canonical order) into a {param: value} dict.

        Parameters
        ----------
        prior_vec : array_like, shape (P,)
            Vector returned by the prior sampler, aligned with `self.parameter_names`.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to scalar float value.

        Raises
        ------
        ValueError
            If the vector length does not match the number of parameters
            or contains non-finite values.
        """
        arr = np.asarray(prior_vec, dtype=np.float32).ravel()
        if arr.size != len(self.parameter_names):
            raise ValueError(
                f"Prior length {arr.size} != #params {len(self.parameter_names)}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Prior vector contains non-finite values (NaN/Inf).")
        return {name: np.float32(arr[i]) for i, name in enumerate(self.parameter_names)}