import numpy as np


class ContextManager:
    def __init__(
        self,
        parameter_names: list[str],
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
        num_samples: int,
        *,
        context: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Build design_matrix with shape (num_samples, num_design_factors).
        Column order follows insertion order of design_config keys.
        - Key "1" is the intercept column (all ones).
        - Other keys may pull their column from `context[key]` (length=num_samples);
          if absent, a U(0,1) column is generated.
        """
        context = context or {}
        design_keys = list(design_config.keys())
        num_design_factors = len(design_keys)
        design_matrix = np.empty((int(num_samples), num_design_factors), dtype=np.float32)

        for design_index, key in enumerate(design_keys):
            if key == "1":
                design_matrix[:, design_index] = 1.0
            else:
                if key in context:
                    col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                    if col.shape[0] != num_samples:
                        raise ValueError(
                            f"context['{key}'] length {col.shape[0]} != num_samples {num_samples}"
                        )
                    design_matrix[:, design_index] = col
                else:
                    design_matrix[:, design_index] = np.random.uniform(
                        0.0, 1.0, size=num_samples
                    ).astype(np.float32)
        return design_matrix

    def build_parameter_mask(
        self,
        design_config: dict[str, list[str]],
        intrinsic_names: list[str],
    ) -> np.ndarray:
        """
        Build parameter_mask with shape (num_design_factors, num_intrinsic_params).
        parameter_mask[design_index, param_index] == 1 iff the design key affects that intrinsic.
        """
        design_keys = list(design_config.keys())
        num_design_factors = len(design_keys)
        num_intrinsic_params = len(intrinsic_names)

        parameter_mask = np.zeros((num_design_factors, num_intrinsic_params), dtype=np.float32)
        for design_index, key in enumerate(design_keys):
            for intrinsic in design_config[key]:
                if intrinsic in intrinsic_names:
                    param_index = intrinsic_names.index(intrinsic)
                    parameter_mask[design_index, param_index] = 1.0
        return parameter_mask


    def sample_parameter_matrix(
        self,
        parameter_mask: np.ndarray,
        priors: dict[str, dict[str, callable]],
        intrinsic_names: list[str],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Build parameter_matrix with shape (num_design_factors, num_intrinsic_params) by sampling
        entry-wise wherever parameter_mask==1.
        - Row 0 (intercept) uses priors[intrinsic]['intercept']()
        - Rows >=1 (slopes)  use priors[intrinsic]['slope']()
        - Entries with mask==0 are 0.0
        """
        if rng is None:
            rng = np.random.default_rng()

        num_design_factors, num_intrinsic_params = parameter_mask.shape
        parameter_matrix = np.zeros((num_design_factors, num_intrinsic_params), dtype=np.float32)

        for design_index in range(num_design_factors):
            for param_index, intrinsic in enumerate(intrinsic_names):
                if parameter_mask[design_index, param_index] == 1.0:
                    sampler = (
                        priors[intrinsic]["intercept"]
                        if design_index == 0
                        else priors[intrinsic]["slope"]
                    )
                    parameter_matrix[design_index, param_index] = np.float32(sampler())
        return parameter_matrix


