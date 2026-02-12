import numpy as np
from typing import Callable

from bayesgpt.utils.simulator_utils import shifted_softplus


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
        fixed_params: list[str] = None,
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
                if name in intrinsic_params and name not in fixed_params:
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

        for k, spec in prior_fun.items():
            # If the user provides a fixed value for a fixed intrinsic parameter,
            # use that value. Otherwise, default to zero.
            if k in fixed_intrinsics and k in fixed_values:
                fixed_value = fixed_values[k]
            else:
                fixed_value = 0.0

            if k in free_intrinsics:
                # SciPy priors
                if hasattr(spec, "rvs"):
                    # print("Identified SciPy priors")
                    scale = spec.std()
                    # scale = spec.mean()
                    sampler = lambda rv=spec: rv.rvs()
                    priors[k] = {
                        "intercept": sampler,
                        "slope": lambda std=scale: np.random.normal(0.0, 1.0)
                    }
                else:
                    priors[k] = {
                        "intercept": spec,
                        "slope": lambda: np.random.normal(0.0, 1.0)
                    }
            else:
                priors[k] = {
                    "intercept": lambda v=fixed_value: v,
                    "slope": lambda: 0.0
                }

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
        keep_intercept: bool = False,
        add_interaction: bool = False,
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
            free_intrinsics=free_intrinsics,
            fixed_intrinsics=fixed_intrinsics,
            add_interaction=add_interaction
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

        # Fill in the matrix
        col_idx = 0
        if has_intercept:
            design_matrix[:, 0] = 1.0
            col_idx = 1

        # Map regressor keys to the start of its associated block (first column)
        start_col = {k: col_idx + j * block_width for j, k in enumerate(regressor_keys)}

        # Main effect
        for j, key in enumerate(main_effect_keys):
            start = start_col[key]

            if key in context:
                col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                if col.shape[0] != num_obs:
                    raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")

                design_matrix[:, start] = col
                continue

            if main_discrete_mask[j] == 1:
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
                x = np.random.standard_normal(size=num_obs).astype(np.float32, copy=False)
                # x = np.random.uniform(size=num_obs).astype(np.float32, copy=False)

                design_matrix[:, start] = x

        # Interaction effect
        for key in interaction_keys:
            start = start_col[key]

            # Allow explicit override via context if desired
            if key in context:
                col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                if col.shape[0] != num_obs:
                    raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")
                design_matrix[:, start] = col
                continue

            try:
                a, b = key.split(":")
            except ValueError:
                continue

            # first-column indices of parent blocks
            a0 = start_col.get(a)
            b0 = start_col.get(b)
            if (a0 is None) or (b0 is None):
                continue

            design_matrix[:, start] = design_matrix[:, a0] * design_matrix[:, b0]
        return design_matrix

    def build_link_functions(
            self,
            intrinsic_params: list[str],
            link_fun: Callable | dict | None,
            default_link_fun: Callable | None = None,
    ) -> dict[str, Callable]:
        """Build a per-intrinsic-parameter link-function mapping.

        Parameters
        ----------
        intrinsic_params
            List of intrinsic parameter names.
        link_fun
            Either a single callable (applied to all parameters), a dictionary
            mapping parameter name -> callable, or None.
        default_link_fun
            Optional default callable used to fill missing keys when `link_fun`
            is a dict. If None and keys are missing, raises.

        Returns
        -------
        link_funs
            Dictionary mapping each intrinsic parameter name to a callable.
        """
        # None -> default to softplus
        if link_fun is None:
            return {k: shifted_softplus for k in intrinsic_params}

        # Single callable -> broadcast to all params
        if callable(link_fun):
            return {k: link_fun for k in intrinsic_params}

        # Dict -> validate and (optionally) fill missing
        if not isinstance(link_fun, dict):
            raise TypeError(
                "link_fun must be a callable, a dict[str, callable], or None; "
                f"got {type(link_fun)}"
            )

        unknown = [k for k in link_fun.keys() if k not in intrinsic_params]
        if unknown:
            raise KeyError(
                "Unknown link_fun keys not in intrinsic_params: " + ", ".join(map(str, unknown))
            )

        missing = [k for k in intrinsic_params if k not in link_fun]
        if missing:
            if default_link_fun is None:
                raise KeyError(
                    "Missing link_fun for intrinsic_params: " + ", ".join(map(str, missing))
                )
            return {k: link_fun.get(k, default_link_fun) for k in intrinsic_params}

        return {k: link_fun[k] for k in intrinsic_params}

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
            np.repeat(np.linspace(0.0, 0.5, num_intrinsic_params), num_categories - 1),
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

    def build_column_labels(
        self,
        design_config: dict[str, list[str]],
        max_num_categories: int = 4,
        keep_intercept: bool = False,
    ) -> list[str]:
        """
        Build human-readable column labels aligned with build_design_matrix() output.

        Only for prior predictive checks.
        """
        regressor_keys = list(design_config.keys())
        has_intercept = ("1" in regressor_keys) and keep_intercept
        regressor_keys = [k for k in regressor_keys if k != "1"]

        block_width = max_num_categories - 1
        labels: list[str] = []

        if has_intercept:
            labels.append("1")

        for key in regressor_keys:
            # Reserve a whole block; first col usually holds continuous regressor
            labels.append(key)
            for j in range(1, block_width):
                labels.append(f"{key}[{j+1}]")  # 2..block_width as dummy-like slots

        return labels
