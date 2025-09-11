import numpy as np
from typing import Dict, Optional, Union
from collections.abc import Callable, Iterable

from .model import Model
from .context_manager import ContextManager


class NestedModelFamily:
    def __init__(
        self,
        name: str,
        model: type[Model],
        context_manager: ContextManager,
        prior_fun: Callable,
        num_samples: int = 10
    ):
        self.name = name
        self.model = model()
        self.context_manager = context_manager
        self.parameter_names = context_manager.parameter_names
        self.num_samples = num_samples
        self.prior_fun = prior_fun

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

            # Model path: prepare → simulate
            model_params = self.model.prepare_params(masked_priors, num_samples)
            print(model_params)
            sim_data = self.model.simulate(model_params, num_samples=num_samples, context=context)

            # Package results for this config
            results.append(
                {
                    "variant_name": f"{self.name}|cfg{idx + 1}",
                    "fixed_parameters": list(fixed_params),
                    "mask": masks,
                    "prior_draw": priors.astype(np.float32, copy=False),
                    "full_params": masked_priors,
                    "sim_data": sim_data,
                    "context": context
                }
            )

        return results

    def _draw_prior_vec(self) -> np.ndarray:
        """
        Draw once from the jitted prior; ensure shape matches parameter_names.
        """
        arr = np.asarray(self.prior_fun(), dtype=np.float32).ravel()
        if arr.size != len(self.parameter_names):
            raise ValueError(
                f"Prior length {arr.size} != #params {len(self.parameter_names)} ({len(self.parameter_names)})."
            )
        return arr

    def _prior_vec_to_dict(self, prior_vec: np.ndarray) -> dict[str, float]:
        """
        Convert a 1-D prior vector (canonical order) into a {param: value} dict.
        """
        arr = np.asarray(prior_vec, dtype=np.float32).ravel()
        if arr.size != len(self.parameter_names):
            raise ValueError(
                f"Prior length {arr.size} != #params {len(self.parameter_names)}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Prior vector contains non-finite values (NaN/Inf).")
        return {name: np.float32(arr[i]) for i, name in enumerate(self.parameter_names)}
