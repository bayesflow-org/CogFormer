import numpy as np
from typing import Callable


class ContextManager:
    """
    Utility class for constructing design configurations, parameter masks,
    design matrices, and sampled parameter matrices for NestedModelFamily.
    """
    def __init__(self, intrinsic_params: list[str] = None):
        self.intrinsic_params = intrinsic_params

    def build_design_config(
        self,
        intrinsic_params: list[str],
        regressed_params: list[str] = None,
        num_regressors: int = 2,
        keep_intercept: bool = False,
        add_interaction: bool = False,
    ):

        config = {}

        # Intercept first
        if keep_intercept:
            names = intrinsic_params
            config["1"] = names

        # Then main effect
        main_keys = []
        for r in range(num_regressors):
            key = f"u_{r+1}" if keep_intercept else f"u_{r}"
            config[key] = regressed_params
            main_keys.append(key)

        # Optionally, interaction effect
        if add_interaction:
            for i in range(len(main_keys)):
                for j in range(i + 1, len(main_keys)):
                    a, b = main_keys[i], main_keys[j]
                    shared_set = set(config[a]) & set(config[b])
                    if shared_set:
                        shared = [p for p in intrinsic_params if p in shared_set]
                        config[f"{a}:{b}"] = shared

        return config

    def build_random_design_config(
        self,
        intrinsic_params: list[str],
        num_regressors: int,
        free_intrinsics: list[str] | set[str] | None = None,
        fixed_intrinsics: list[str] | set[str] | None = None,
        keep_intercept: bool = False,
        free_prob: float = 0.5,
        intercept_prob: float = 0.5,
        add_interaction: bool = False,
    ) -> dict[str, list[str]]:
        """
        Randomly build a design_config mapping regressors and optional intercept
        to subsets of intrinsic parameters.
        """
        # Set up mandatory and intercept only params based on free and fixed intrinsics
        config = {}

        # Intercept first
        if keep_intercept:
            names = intrinsic_params
            config["1"] = names

        # Main effect
        main_keys = []
        for r in range(num_regressors):
            key = f"u_{r+1}" if keep_intercept else f"u_{r}"
            names = []
            for param in intrinsic_params:
                if (param not in fixed_intrinsics) and (np.random.rand() < free_prob):
                    names.append(param)
            config[key] = names
            main_keys.append(key)

        # Interaction effect
        if add_interaction:
            for i in range(len(main_keys)):
                for j in range(i + 1, len(main_keys)):
                    a, b = main_keys[i], main_keys[j]
                    shared_set = set(config[a]) & set(config[b])
                    if shared_set:
                        shared = [p for p in intrinsic_params if p in shared_set]
                        config[f"{a}:{b}"] = shared

        return config

    def build_parameter_mask(
        self,
        design_config: dict[str, list[str]],
        intrinsic_params: list[str],
        max_num_categories: int = 3,
        keep_intercept: bool = False
    ) -> np.ndarray:
        """
        Convert a design_config dict into a binary parameter mask mapping
        regressors (rows) to intrinsic parameters (columns).
        """
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

    def build_intrinsic_priors(
        self,
        prior_fun: Callable | dict = None,
        free_intrinsics: list[str] | set[str] | None = None,
        fixed_intrinsics: list[str] | set[str] | None = None,
        fixed_values: dict[str, float] | None = None
    ):
        priors = {}

        for k in prior_fun.keys():
            # If the user provides a fixed value for a fixed intrinsic parameter,
            # use that value. Otherwise, default to zero.
            if k in fixed_intrinsics and k in fixed_values:
                fixed_value = fixed_values[k]
            else:
                fixed_value = 0.0
            prior = {
                k: {
                    # Assign variable to avoid late-binding issues
                    "intercept": prior_fun[k] if k in free_intrinsics else lambda v=fixed_value: v,
                    "slope": lambda key=k: np.random.normal(0.0, 1.0) if key in free_intrinsics else lambda: 0.0
                }
            }
            priors = priors | prior

        return priors

    def build_random_parameter_mask(
        self,
        intrinsic_params: list[str],
        num_regressors: int = 2,
        # max_num_regressors: int = 5,
        max_num_categories: int = 4,
        free_intrinsics: list[str] | set[str] | None = None,
        fixed_intrinsics: list[str] | set[str] | None = None,
        free_prob: float = 0.5,     # Probability of a param being free
        keep_intercept: bool = False
    ) -> tuple:
        """
        Randomly generate both a design_config and its corresponding parameter mask.

        free_intrinsics: both intercept and slopes are sampled
        fixed_intrinsics: only intercept is sampled, slopes are not
        frozen_intrinsics: neither intercept nor slope are sampled
        """
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
            # max_num_regressors=max_num_regressors,
            max_num_categories=max_num_categories,
            keep_intercept=keep_intercept,
        )

        return parameter_mask, design_config

    def build_discrete_mask(
            self,
            design_config: dict[str, list[str]],
            discrete_prob: float = 0.5,
            keep_intercept: bool = False,
            add_interaction: bool = False,
    ):
        """
        Sample a discrete mask for main regressors only and expand it to a total-length
        mask aligned with design_config key order (excluding intercept).
        """
        regressor_keys = list(design_config.keys())
        if keep_intercept and ("1" in regressor_keys):
            regressor_keys = [k for k in regressor_keys if k != "1"]
        else:
            regressor_keys = [k for k in regressor_keys if k != "1"]

        main_keys = [k for k in regressor_keys if ":" not in k]
        num_main = len(main_keys)

        main_discrete_mask = self.build_random_discrete_mask(
            num_regressors=num_main,
            discrete_prob=discrete_prob,
        ).astype(np.float32, copy=False)

        discrete_mask = np.full((len(regressor_keys),), -1.0, dtype=np.float32)

        ptr = 0
        for i, k in enumerate(regressor_keys):
            if ":" in k:
                continue
            discrete_mask[i] = float(main_discrete_mask[ptr])
            ptr += 1

        return main_discrete_mask, discrete_mask

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
        # max_num_regressors: int = 5,
        context: dict[str, np.ndarray] | None = None,
        discrete_prob: float = 0.5,
        keep_intercept: bool = False,
        min_num_categories: int = 2,
        max_num_categories: int = 4,
        main_discrete_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Build design matrix based on the respective configuration of free intrinsic parameters.
        The design matrix should have the shape of
        """
        # Provide context
        context = context or {}
        regressor_keys = list(design_config.keys())
        has_intercept = ("1" in regressor_keys) and keep_intercept

        # Exclude intercept from regressor keys
        regressor_keys = [k for k in regressor_keys if k != "1"]
        num_regressors = len(regressor_keys)

        # Split regressor keys further into effect keys: main vs. interaction
        main_effect_keys = [k for k in regressor_keys if ":" not in k]
        interaction_keys = [k for k in regressor_keys if ":" in k]

        # Construct per-parameter column blocks
        block_width = max_num_categories - 1
        num_cols = num_regressors * block_width + (1 if has_intercept else 0)

        # Design matrix always includes intercept column if present in design_config
        design_matrix = np.zeros((num_obs, num_cols))

        # Generate discrete mask for the non-intercept regressors if none provided
        if main_discrete_mask is None:
            main_discrete_mask = self.build_random_discrete_mask(
                num_regressors=len(main_effect_keys),
                discrete_prob=discrete_prob,
            )

        discrete_mask = self.build_random_discrete_mask(
            num_regressors=len(main_effect_keys),
            discrete_prob=discrete_prob
        )

        # Fill in the matrix
        col_idx = 0
        if has_intercept:
            design_matrix[:, 0] = 1.0
            col_idx = 1

        # Map regressor keys to the start of its associated block (first column)
        start_col = {k: col_idx + j * block_width for j, k in enumerate(regressor_keys)}

        # Main effect
        j = 0
        for key in main_effect_keys:
            start = start_col[key]

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
            j += 1

        # Interaction effect
        for key in interaction_keys:
            start = start_col[key]

            if key in context:
                col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                if col.shape[0] != num_obs:
                    raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")
                design_matrix[:, start] = col

        return design_matrix

    def build_parameter_indices(
        self,
        intrinsic_params: list[str],
        num_regressors: int,
        num_categories: int,
    ) -> np.ndarray:
        """
        Build an array of positional indices for the intrinsic parameters.
        """
        # Get number of intrinsic params
        num_intrinsic_params = len(intrinsic_params)
        indices = np.tile(
            np.repeat(np.linspace(0.0, 1.0, num_intrinsic_params), num_categories - 1),
            num_regressors
        )
        return indices

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
        intrinsic_params: list[str],
        prior_fun: dict[str, Callable | dict[str, Callable]],
        keep_intercept: bool = False,
    ) -> np.ndarray:
        """
        Sample parameter matrix based on the given parameter mask and the associated priors.
        This has shape (num_regressors, num_intrinsic_params).

        """
        num_regressors, num_intrinsic_params = parameter_mask.shape
        parameter_matrix = np.zeros((num_regressors, num_intrinsic_params))

        # Handle empty parameter_mask
        if num_regressors == 0:
            return parameter_matrix

        # Heuristic: treat the very first row as intercept if any nonzero exists there.
        has_intercept = keep_intercept and parameter_mask[0].any()

        for design_index in range(num_regressors):
            is_intercept_row = has_intercept and (design_index == 0)

            for param_index, intrinsic_param in enumerate(intrinsic_params):
                # Sample prior if parameter or regressor is not masked
                if parameter_mask[design_index, param_index] != 1.0:
                    continue

                if parameter_mask[design_index, param_index] == 1.0:
                    if (design_index == 0) and has_intercept:
                        sampler = prior_fun[intrinsic_param]["intercept"]
                    else:
                        sampler = prior_fun[intrinsic_param]["slope"]

                    parameter_matrix[design_index, param_index] = sampler()
        return parameter_matrix


    def sample_dummies(
        self,
        num_obs: int,
        min_num_categories: int = 2,
        max_num_categories: int = 4,
    ):
        """
        Generate categorical dummy encoding for the design matrix.
        """
        # Randomly generate a num_categories
        num_categories = np.random.randint(min_num_categories, max_num_categories + 1)

        p = float(1.0 / num_categories)
        one_hot = np.random.multinomial(1, [p] * num_categories, size=num_obs)
        dummies = one_hot[:, :num_categories - 1]
        return dummies
