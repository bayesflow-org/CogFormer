import numpy as np
from typing import Callable


class ContextManager:
    def __init__(self, parameter_names: list[str] = None):
        self.parameter_names = parameter_names or []

    def build_parameter_mask(
        self,
        design_config: dict[str, list[str]],
        intrinsic_params: list[str],
        max_num_regressors: int = 10,
        keep_intercept: bool = False
    ) -> np.ndarray:

        if max_num_regressors is None:
            raise ValueError("max_num_regressors must be provided for padded masks.")

        has_intercept = ("1" in design_config) and keep_intercept
        non_intercept_keys = [k for k in design_config.keys() if k != "1"]

        # Order: intercept first if present, then non-intercepts
        ordered_keys = (["1"] if has_intercept else []) + non_intercept_keys
        num_intrinsic_params = len(intrinsic_params)

        mask = np.zeros((max_num_regressors, num_intrinsic_params), dtype=np.float32)
        for design_index, key in enumerate(ordered_keys):
            for intrinsic in design_config[key]:
                if intrinsic in intrinsic_params:
                    param_index = intrinsic_params.index(intrinsic)
                    mask[design_index, param_index] = 1.0

        return mask

    def build_random_parameter_mask(
            self,
            intrinsic_params: list[str],
            num_regressors: int,
            max_num_regressors: int = 10,
            mandatory_intrinsics: list[str] | set[str] | None = None,
            intercept_only_intrinsics: list[str] | set[str] | None = None,
            free_prob: float = 0.5,     # Probability of a param being free
            keep_intercept: bool = False
    ) -> np.ndarray:

        mandatory = set(mandatory_intrinsics or [])
        intercept_only = set(intercept_only_intrinsics or [])

        num_intrinsic_params = len(intrinsic_params)
        # Always include intercept row, even if num_regressors = 0
        num_rows = num_regressors + (1 if keep_intercept else 0)
        mask = np.zeros((max_num_regressors, num_intrinsic_params), dtype=np.float32)

        # Intercept row
        if keep_intercept:
            for j, name in enumerate(intrinsic_params):
                if name in mandatory or name in intercept_only:
                    mask[0, j] = 1.0
                else:
                    mask[0, j] = float(np.random.random() < free_prob)

        # Slope rows
        start = 1 if keep_intercept else 0
        for i in range(start, num_rows):
            for j, name in enumerate(intrinsic_params):
                if name in intercept_only:
                    mask[i, j] = 0.0
                else:
                    mask[i, j] = float(np.random.random() < free_prob)

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
        # No intercept, no discrete mask
        if num_regressors == 0:
            return np.array([])

        discrete_mask = 1 * (np.random.rand(num_regressors) < discrete_prob)
        print(discrete_mask)
        return discrete_mask

    def build_regressor_mask(
        self,
        num_regressors: int,
        max_num_regressors: int = 10,
        keep_intercept: bool = False
    ) -> np.ndarray:
        mask = np.zeros(max_num_regressors, dtype=np.float32)
        used = num_regressors + (1 if keep_intercept else 0)
        mask[:used] = 1.0
        return mask

    def build_design_matrix(
        self,
        design_config: dict[str, list[str]],
        num_obs: int,
        max_num_regressors: int = 10,
        context: dict[str, np.ndarray] | None = None,
        discrete_prob: float = 0.5,
        keep_intercept: bool = False,
        min_num_categories: int = 2,
        max_num_categories: int = 5
    ) -> np.ndarray:
        # Provide context
        context = context or {}
        regressor_keys = list(design_config.keys())
        has_intercept = ("1" in regressor_keys) and keep_intercept

        # Exclude intercept from regressor keys
        regressor_keys = [k for k in regressor_keys if k != "1"]
        num_regressors = len(regressor_keys)

        # Construct per-parameter column blocks
        block_width = max_num_categories - 1
        num_cols = num_regressors * block_width + (1 if has_intercept else 0)

        # Design matrix always includes intercept column if present in design_config
        design_matrix = np.zeros((num_obs, num_cols), dtype=np.float32)

        # Generate discrete mask for the non-intercept regressors if none provided
        discrete_mask = self.build_random_discrete_mask(
            num_regressors=num_regressors, discrete_prob=discrete_prob
        )

        # Fill in the matrix
        col_idx = 0
        if has_intercept:
            design_matrix[:, 0] = 1
            col_idx = 1

        for j, key in enumerate(regressor_keys):
            start = col_idx + j * block_width
            end = start + block_width

            if key in context:
                col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                if col.shape[0] != num_obs:
                    raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")
                design_matrix[:, start] = col
            elif discrete_mask[j] == 1:
                # Sample dummies
                dummies = self.sample_dummies(
                    num_obs=num_obs,
                    min_num_categories=min_num_categories,
                    max_num_categories=max_num_categories
                )

                # Infer num_categories and increment the column index
                num_categories = dummies.shape[1]
                col_idx += num_categories
            else:
                design_matrix[:, start] = np.random.uniform(0.0, 1.0, size=num_obs)
                col_idx += 1

        return design_matrix

    def mask_to_design_config(
            self,
            parameter_mask: np.ndarray,
            intrinsic_params: list[str],
            keep_intercept: bool = False,
            ignore_padding: bool = True
    ) -> dict[str, list[str]]:
        """
        Convert a (num_regressors × num_intrinsics) binary mask into a design_config dict.
        Row 0 -> key "1" (intercept), rows 1.. -> "u_1", "u_2", ...
        """
        num_rows, num_intrinsic_params = parameter_mask.shape
        config: dict[str, list[str]] = {}

        for i in range(num_rows):
            # Skip padded rows if requested
            if ignore_padding and not parameter_mask[i].any():
                continue

            if i == 0 and keep_intercept:
                key = "1"
            else:
                key = f"u_{i + 1}" if keep_intercept else f"u_{i}"

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


    def sample_dummies(
        self,
        num_obs: int,
        discrete_prob: float = 0.5,
        min_num_categories: int = 2,
        max_num_categories: int = 5,
    ):
        if np.random.rand() < discrete_prob:
            num_categories = np.random.randint(min_num_categories, max_num_categories + 1)
            if num_categories < 2:
                num_categories = 2

            p = float(np.ones(num_categories) / num_categories)
            one_hot_encoder = np.random.multinomial(1, p, size=num_obs).astype(np.float32)
            dummies = one_hot_encoder[:, :num_categories - 1]
            return dummies
