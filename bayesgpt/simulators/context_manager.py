import numpy as np
from typing import Callable


class ContextManager:
    def __init__(self, parameter_names: list[str] = None):
        self.parameter_names = parameter_names or []

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
    ) -> np.ndarray:

        mandatory = set(mandatory_intrinsics or [])
        intercept_only = set(intercept_only_intrinsics or [])

        num_intrinsic_params = len(intrinsic_params)
        mask = np.zeros((num_regressors, num_intrinsic_params), dtype=np.float32)

        # Intercept row
        for j, name in enumerate(intrinsic_params):
            if name in mandatory or name in intercept_only:
                mask[0, j] = 1.0
            else:
                mask[0, j] = float(np.random.random() < p_free)

        # Slope rows
        for i in range(1, num_regressors):
            for j, name in enumerate(intrinsic_params):
                if name in intercept_only:
                    mask[i, j] = 0.0
                else:
                    mask[i, j] = float(np.random.random() < p_free)

        return mask

    def build_random_discrete_mask(
        self,
        num_regressors: int,
        discrete_prob: float = 0.5
    ) -> np.ndarray:
        """
        Bernoulli mask over regressors: 1->discrete {0,1}, 0->continuous U[0,1].
        By default, keeps the intercept (col 0) continuous.
        """
        discrete_mask = 1 * (np.random.rand(num_regressors) < discrete_prob)
        # Intercept should always be continuous
        discrete_mask[0] = 0
        return discrete_mask

    def build_design_matrix(
        self,
        design_config: dict[str, list[str]],
        num_obs: int,
        *,
        context: dict[str, np.ndarray] | None = None,
        discrete_mask: np.ndarray | None = None,
        discrete_prob: float = 0.5
    ) -> np.ndarray:
        # Provide context
        context = context or {}
        regressor_keys = list(design_config.keys())
        if "1" in regressor_keys and regressor_keys[0] != "1":
            regressor_keys = ["1"] + [k for k in regressor_keys if k != "1"]

        num_regressors = len(regressor_keys)

        # Generate discrete mask if none provided
        if discrete_mask is None:
            discrete_mask = self.build_random_discrete_mask(num_regressors=num_regressors, discrete_prob=discrete_prob)

        design_matrix = np.empty((num_obs, num_regressors), dtype=np.float32)

        for j, key in enumerate(regressor_keys):
            if key == "1":
                design_matrix[:, j] = 1.0
                continue

            if key in context:
                col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                if col.shape[0] != num_obs:
                    raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")
                design_matrix[:, j] = col
            elif discrete_mask[j] == 1:
                design_matrix[:, j] = np.random.randint(0, 2, size=num_obs).astype(np.float32)
            else:
                design_matrix[:, j] = np.random.uniform(0.0, 1.0, size=num_obs).astype(np.float32)

        return design_matrix

    def mask_to_design_config(
            self,
            parameter_mask: np.ndarray,
            intrinsic_params: list[str],
    ) -> dict[str, list[str]]:
        """
        Convert a (num_regressors × num_intrinsics) binary mask into a design_config dict.
        Row 0 -> key "1" (intercept), rows 1.. -> "u_1", "u_2", ...
        """
        num_regressors, num_intrinsic_params = parameter_mask.shape

        config: dict[str, list[str]] = {}
        for i in range(num_regressors):
            key = "1" if i == 0 else f"u_{i}"
            config[key] = []
            for j in range(num_intrinsic_params):
                if parameter_mask[i, j] == 1.0:
                    config[key].append(intrinsic_params[j])

        return config

    def sample_parameter_matrix(
        self,
        parameter_mask: np.ndarray,
        prior_fun: dict[str, Callable],
        intrinsic_params: list[str],
    ) -> np.ndarray:
        """
        Build parameter_matrix with shape (num_regressors, num_intrinsic_params) by sampling
        entry-wise wherever parameter_mask==1.
        - Row 0 (intercept) uses prior_fun[intrinsic]['intercept']()
        - Rows >=1 (slopes)  use prior_fun[intrinsic]['slope']()
        - Entries with mask==0 are 0.0
        """

        num_regressors, num_intrinsic_params = parameter_mask.shape
        parameter_matrix = np.zeros((num_regressors, num_intrinsic_params), dtype=np.float32)

        for design_index in range(num_regressors):
            for param_index, intrinsic in enumerate(intrinsic_params):
                if parameter_mask[design_index, param_index] == 1.0:

                    if design_index == 0:
                        sampler = prior_fun[intrinsic]["intercept"]
                    else:
                        sampler = prior_fun[intrinsic]["slope"]

                    parameter_matrix[design_index, param_index] = sampler()
        return parameter_matrix
