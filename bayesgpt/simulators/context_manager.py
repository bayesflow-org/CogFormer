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

    def apply_mask(
        self,
        sampled_params: np.ndarray | dict[str, float],
        mask: np.ndarray | dict[str, float],
    ) -> dict[str, float]:

        masked_params = {}
        if isinstance(sampled_params, dict):
            for key, value in sampled_params.items():
                masked_params[key] = np.float32(value) * float(mask[key])
        elif isinstance(sampled_params, np.ndarray):
            for i, val in enumerate(sampled_params):
                masked_params[i] = np.float32(val) * float(mask[i])
        return masked_params
