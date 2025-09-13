import numpy as np
from typing import Callable


class ContextManager:
    def __init__(
        self,
        parameter_names: list[str] = None,
    ):
        self.parameter_names = parameter_names
        self.mask = None  # Delay mask creation until dims are inferred


    def build_mask(self, fixed_parameters: list[str] = None, free_parameters: list[str] = None) -> dict[str, float]:
        """
        Mask away the fixed parameters as 0.0
        """
        mask = {}
        if free_parameters is not None and fixed_parameters is not None:
            raise ValueError("Only specify one of fixed_parameters or free_parameters")
        elif free_parameters is not None:
            for name in self.parameter_names:
                mask[name] = 1.0 if name in free_parameters else 0.0
        elif fixed_parameters is not None:
            for name in self.parameter_names:
                mask[name] = 0.0 if name in fixed_parameters else 1.0

        self.mask = mask
        return mask

    def build_random_mask(
            self,
            free_parameters: list[str] | set[str] | None = None,
            p_free: float = 0.5,
            rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """
        Randomly marks parameters as free(=1.0) or fixed(=0.0), while forcing 'free_parameters' to 1.0.

        Parameters
        ----------
        free_parameters : list[str] | set[str] | None
            Parameters that must remain free (mask=1.0).
        p_free : float
            Probability that a non-always-free parameter is free. Must be in [0, 1].
        rng : np.random.Generator | None
            Optional numpy Generator for deterministic masks.

        Returns
        -------
        dict[str, float]
            {param_name: 0.0 or 1.0}
        """
        if rng is None:
            rng = np.random.default_rng()
        free_parameters = set(free_parameters or [])
        mask = {}
        for name in self.parameter_names:
            if name in free_parameters:
                mask[name] = 1.0
            else:
                mask[name] = float(rng.random() < p_free)
        self.mask = mask
        return mask

    def apply_mask(
        self,
        sampled_params: np.ndarray | dict[str, float],
        mask: np.ndarray | dict[str, float],
        fixed_values: dict[str, float] | None = None,
    ) -> dict[str, float]:

        fixed_values = fixed_values or {}

        # Accessors normalized across dict/ndarray inputs
        def _mask_for(i: int, name: str) -> float:
            return float(mask[name]) if isinstance(mask, dict) else float(mask[i])

        def _sampled_for(i: int, name: str) -> float:
            return float(sampled_params[name]) if isinstance(sampled_params, dict) else float(sampled_params[i])

        masked_params: dict[str, float] = {}
        for i, name in enumerate(self.parameter_names):
            m = np.float32(_mask_for(i, name))
            s = np.float32(_sampled_for(i, name))
            f = np.float32(fixed_values.get(name, 0.0))
            # sampled when free (m=1), fixed intercept when masked (m=0)
            val = s * m + f * (1.0 - m)
            masked_params[name] = np.float32(val)
        return masked_params

    def build_design_matrix(
        self,
        design_config: dict[str, list[str]],
        num_obs: int,
        *,
        context: dict[str, np.ndarray] | None = None,
        discrete_prob: float = 0.5
    ) -> np.ndarray:
        """
        Build design_matrix with shape (num_obs, num_regressors).
        Column order follows insertion order of design_config keys.
        - Key "1" is the intercept column (all ones).
        - Other keys may pull their column from `context[key]` (length=num_obs);
          if absent, a U(0,1) column is generated.
        """
        context = context or {}
        design_keys = list(design_config.keys())
        num_regressors = len(design_keys)
        design_matrix = np.empty((int(num_obs), num_regressors), dtype=np.float32)

        for design_index, key in enumerate(design_keys):
            if key == "1":
                design_matrix[:, design_index] = 1.0
            else:
                if key in context: # Manually define the non-intercept columns as contexts
                    col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                    if col.shape[0] != num_obs:
                        raise ValueError(
                            f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}"
                        )
                    design_matrix[:, design_index] = col
                else:  # Automatically determine if a non-intercept config is discrete or not
                    discrete = np.random.rand() > discrete_prob
                    design_matrix[:, design_index] = np.random.uniform(
                        0.0, 1.0, size=num_obs
                    ).astype(np.float32)
                    design_matrix[:, design_index] = design_matrix[:, design_index] if discrete else design_matrix[:, design_index].round()
        return design_matrix

    def build_parameter_mask(
        self,
        design_config: dict[str, list[str]],
        intrinsic_params: list[str],
    ) -> np.ndarray:
        """
        Build parameter_mask with shape (num_regressors, num_intrinsic_params).
        parameter_mask[design_index, param_index] == 1 iff the design key affects that intrinsic.
        """
        design_keys = list(design_config.keys())
        num_regressors = len(design_keys)
        num_intrinsic_params = len(intrinsic_params)

        parameter_mask = np.zeros((num_regressors, num_intrinsic_params), dtype=np.float32)
        for design_index, key in enumerate(design_keys):
            for intrinsic in design_config[key]:
                if intrinsic in intrinsic_params:
                    param_index = intrinsic_params.index(intrinsic)
                    parameter_mask[design_index, param_index] = 1.0
        return parameter_mask

    def build_random_parameter_mask(
            self,
            intrinsic_params: list[str],
            num_regressors: int,
            mandatory_intrinsics: list[str] | set[str] | None = None,
            intercept_only_intrinsics: list[str] | set[str] | None = None,
            p_free: float = 0.5,
            rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Random parameter_mask for full-regression designs.

        Parameters
        ----------
        intrinsic_params : list[str]
            Names of the intrinsic parameters (columns).
        num_regressors : int
            Total number of design factors (rows, including intercept).
        mandatory_intrinsics : list[str] | set[str] | None
            Intrinsics that must always have their intercept row set to 1.
        intercept_only_intrinsics : list[str] | set[str] | None
            Intrinsics that are restricted to intercept row only (never slopes).
        p_free : float
            Probability that a non-mandatory, non-intercept-only intrinsic is free (1.0).
        rng : np.random.Generator | None
            Optional numpy Generator for reproducibility.

        Returns
        -------
        np.ndarray
            Binary parameter_mask of shape (num_regressors, len(intrinsic_params)).
        """
        rng = rng or np.random.default_rng()
        mandatory_intrinsics = set(mandatory_intrinsics or [])
        intercept_only_intrinsics = set(intercept_only_intrinsics or [])

        m, n = int(num_regressors), len(intrinsic_params)
        mask = np.zeros((m, n), dtype=np.float32)

        # Intercept row
        for j, name in enumerate(intrinsic_params):
            if name in mandatory_intrinsics or name in intercept_only_intrinsics:
                mask[0, j] = 1.0
            else:
                mask[0, j] = float(rng.random() < p_free)

        # Slope rows
        for i in range(1, m):
            for j, name in enumerate(intrinsic_params):
                if name in intercept_only_intrinsics:
                    mask[i, j] = 0.0
                else:
                    mask[i, j] = float(rng.random() < p_free)

        return mask

    def mask_to_design_config(
            self,
            parameter_mask: np.ndarray,
            intrinsic_params: list[str],
    ) -> dict[str, list[str]]:
        """
        Convert a (num_regressors × num_intrinsics) binary mask into a design_config dict.
        Row 0 -> key "1" (intercept), rows 1.. -> "u_1", "u_2", ...
        """
        m, n = parameter_mask.shape
        if n != len(intrinsic_params):
            raise ValueError("parameter_mask width != len(intrinsic_params)")
        config: dict[str, list[str]] = {}
        for i in range(m):
            key = "1" if i == 0 else f"u_{i}"
            config[key] = [intrinsic_params[j] for j in range(n) if parameter_mask[i, j] == 1.0]
        return config

    def sample_parameter_matrix(
        self,
        parameter_mask: np.ndarray,
        priors: dict[str, dict[str, Callable]],
        intrinsic_params: list[str],
    ) -> np.ndarray:
        """
        Build parameter_matrix with shape (num_regressors, num_intrinsic_params) by sampling
        entry-wise wherever parameter_mask==1.
        - Row 0 (intercept) uses priors[intrinsic]['intercept']()
        - Rows >=1 (slopes)  use priors[intrinsic]['slope']()
        - Entries with mask==0 are 0.0
        """

        num_regressors, num_intrinsic_params = parameter_mask.shape
        parameter_matrix = np.zeros((num_regressors, num_intrinsic_params))

        for design_index in range(num_regressors):
            for param_index, intrinsic in enumerate(intrinsic_params):
                if parameter_mask[design_index, param_index] == 1.0:

                    if design_index == 0:
                        sampler = priors[intrinsic]["intercept"]
                    else:
                        sampler = priors[intrinsic]["slope"]

                    parameter_matrix[design_index, param_index] = sampler()
        return parameter_matrix
