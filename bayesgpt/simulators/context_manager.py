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
        max_num_categories: int = 5,
        keep_intercept: bool = False
    ) -> np.ndarray:

        regressor_keys = list(design_config.keys())
        has_intercept = ("1" in regressor_keys) and keep_intercept
        regressor_keys = [k for k in regressor_keys if k != "1"]

        # Figure out block width and total number of columns
        num_regressors = len(regressor_keys)
        block_width = max_num_categories - 1
        num_intrinsic_params = len(intrinsic_params)
        num_cols = num_regressors * block_width + (1 if has_intercept else 0)

        mask = np.zeros((num_cols, num_intrinsic_params), dtype=np.float32)

        # Building the matrix
        # Start with intercept
        row_idx = 0
        if has_intercept:
            for name in design_config.get("1", []):
                if name in intrinsic_params:
                    j = intrinsic_params.index(name)
                    mask[row_idx, j] = 1.0
            row_idx = 1

        # Then to each regressor
        for design_index, key in enumerate(regressor_keys):
            start = row_idx + design_index * block_width
            rows = list(range(start, start + block_width))
            for intrinsic in design_config[key]:
                if intrinsic in intrinsic_params:
                    param_index = intrinsic_params.index(intrinsic)
                    for r in rows:
                        mask[r, param_index] = 1.0

        return mask

    def build_random_design_config(
        self,
        intrinsic_params: list[str],
        num_regressors: int,
        free_prob: float = 0.5,
        keep_intercept: bool = False,
        free_intrinsics: list[str] | set[str] | None = None,
        fixed_intrinsics: list[str] | set[str] | None = None,
    ) -> dict[str, list[str]]:

        mandatory = set(free_intrinsics or [])
        intercept_only = set(fixed_intrinsics or [])
        config: dict[str, list[str]] = {}

        if keep_intercept:
            names = []
            for param in intrinsic_params:
                if param in mandatory or param in intercept_only or (np.random.rand() < free_prob):
                    names.append(param)
            config["1"] = names

        for r in range(num_regressors):
            key = f"u_{r+1}" if keep_intercept else f"u_{r}"
            names = []
            for param in intrinsic_params:
                if param in intercept_only:
                    continue
                elif param in mandatory and (np.random.rand() < free_prob):
                    names.append(param)
            config[key] = names

        return config

    def build_random_parameter_mask(
            self,
            intrinsic_params: list[str],
            num_regressors: int,
            max_num_regressors: int = 10,
            max_num_categories: int = 4,
            free_intrinsics: list[str] | set[str] | None = None,
            fixed_intrinsics: list[str] | set[str] | None = None,
            free_prob: float = 0.5,     # Probability of a param being free
            keep_intercept: bool = False
    ) -> tuple:

        design_config = self.build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=num_regressors,
            free_prob=free_prob,
            keep_intercept=keep_intercept,
            free_intrinsics=free_intrinsics,    # Intercept + slope
            fixed_intrinsics=fixed_intrinsics   # Intercept only
        )

        parameter_mask = self.build_parameter_mask(
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_regressors=max_num_regressors,
            max_num_categories=max_num_categories,
            keep_intercept=keep_intercept,
        )

        return parameter_mask, design_config

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

        discrete_mask = np.array(1 * (np.random.rand(num_regressors) < discrete_prob))
        return discrete_mask

    def build_regressor_mask(
        self,
        num_regressors: int,
        max_num_categories: int = 4,
        keep_intercept: bool = False,
    ) -> np.ndarray:
        block_width = max_num_categories - 1
        num_cols = num_regressors * block_width + (1 if keep_intercept else 0)
        mask = np.ones(num_cols, dtype=np.float32)

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
        max_num_categories: int = 4
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
        design_matrix = np.zeros((num_obs, num_cols))

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
                design_matrix[:, start:(start + num_categories)] = dummies
            else:
                design_matrix[:, start] = np.random.uniform(0.0, 1.0, size=num_obs)

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
        Row 0 -> key "1" (intercept), rows 1 -> "u_1", "u_2", ...
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
        prior_fun: dict[str, Callable | dict[str, Callable]],
        intrinsic_params: list[str],
    ) -> np.ndarray:

        num_regressors, num_intrinsic_params = parameter_mask.shape
        parameter_matrix = np.zeros((num_regressors, num_intrinsic_params))

        # Heuristic: treat the very first row as intercept if any nonzero exists there.
        has_intercept = parameter_mask[0].any()

        for design_index in range(num_regressors):
            for param_index, intrinsic in enumerate(intrinsic_params):
                if parameter_mask[design_index, param_index] == 1.0:

                    if (design_index == 0) and has_intercept:
                        sampler = prior_fun[intrinsic]["intercept"]
                    else:
                        sampler = prior_fun[intrinsic]["slope"]

                    parameter_matrix[design_index, param_index] = sampler()
        return parameter_matrix


    def sample_dummies(
        self,
        num_obs: int,
        min_num_categories: int = 2,
        max_num_categories: int = 4,
    ):
        # Randomly generate a num_categories
        num_categories = np.random.randint(min_num_categories, max_num_categories + 1)

        p = float(1.0 / num_categories)
        one_hot= np.random.multinomial(1, [p] * num_categories, size=num_obs)
        dummies = one_hot[:, :num_categories - 1]
        return dummies
