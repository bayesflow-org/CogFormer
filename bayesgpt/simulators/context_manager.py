import numpy as np


class ContextManager:
    def __init__(
        self,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        self.mask = None  # Delay mask creation until dims are inferred

    def build_mask(self, fixed_parameters: list[str] = None) -> dict[str, float]:
        """
        Mask away the fixed parameters as 0.0
        """
        mask = {}
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
