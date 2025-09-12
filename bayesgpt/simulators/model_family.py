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
        num_samples: int = 10,
        intrinsic_params: list[str] | None = None,
    ):
        self.name = name
        self.model = model()
        self.context_manager = context_manager
        self.parameter_names = context_manager.parameter_names
        self.num_samples = num_samples
        self.prior_fun = prior_fun
        self.intrinsic_params = intrinsic_params


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

    def sample_with_design(
            self,
            design_config: dict[str, list[str]],
            intrinsic_names: list[str],
            priors: dict[str, dict[str, callable]],
            *,
            num_samples: int | None = None,
            context: dict[str, np.ndarray] | None = None,
    ):
        """
        Full regression path (model-agnostic):
          design_matrix      : (num_samples, num_design_factors)
          parameter_mask     : (num_design_factors, num_intrinsic_params)
          parameter_matrix   : (num_design_factors, num_intrinsic_params)
          intrinsic_values   : (num_samples, num_intrinsic_params) = design_matrix @ parameter_matrix
        """
        num_samples = self.num_samples if num_samples is None else int(num_samples)

        # 1) Construct matrices
        design_matrix = self.context_manager.build_design_matrix(
            design_config=design_config, num_samples=num_samples, context=context
        )
        parameter_mask = self.context_manager.build_parameter_mask(
            design_config=design_config, intrinsic_names=intrinsic_names
        )
        parameter_matrix = self.context_manager.sample_parameter_matrix(
            parameter_mask=parameter_mask, priors=priors, intrinsic_names=intrinsic_names
        )

        # 2) Compose per-trial intrinsic values
        intrinsic_values_matrix = (design_matrix @ parameter_matrix).astype(np.float32)  # (N×M)

        # 3) Package params for the model (still model-agnostic)
        params = {
            name: intrinsic_values_matrix[:, j].astype(np.float32, copy=False)
            for j, name in enumerate(intrinsic_names)
        }
        params["_intercepts"] = {name: float(parameter_matrix[0, j]) for j, name in enumerate(intrinsic_names)}

        # 4) Model call
        model_params = self.model.prepare_params(params, num_samples=num_samples)
        sim_data = self.model.simulate(model_params, num_samples=num_samples, context=None)

        return {
            "variant_name": f"{self.name}|full_regression",
            "design_config": design_config,
            "design_matrix": design_matrix,
            "parameter_mask": parameter_mask,
            "parameter_matrix": parameter_matrix,
            "intrinsic_names": intrinsic_names,
            "params": model_params,
            "sim_data": sim_data,
        }

    def _draw_prior_vec(self) -> np.ndarray:
        """
        Draw once from the num_intrinsic_paramsitted prior; ensure shape matches parameter_names.
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
