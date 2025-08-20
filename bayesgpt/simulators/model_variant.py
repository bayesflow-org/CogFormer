import numpy as np
from .model import Model
from ..utils.simulator_utils import Tokenizer


class ModelVariant:
    """
    Encapsulates a model variant with free and fixed parameters,
    providing simulation and inference interface with parameter masking.

    Parameters
    ----------
    name : str
        Name or identifier for this variant.
    model : type
        A callable class implementing a `.simulate(params: dict, batch_size: int)` method.
    parameter_manager : ParameterManager
        Configuration object that manages sampling, value fixing, and masking of parameters.
    """

    def __init__(
        self,
        name: str,
        model: type[Model],
        tokenizer: Tokenizer
    ):
        self.name = name
        self.model: Model = model()
        self.parameter_manager = tokenizer
        self.parameter_names = list(tokenizer.parameter_names)


    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Simulate data and return full parameter vectors.

        Returns
        -------
        dict with keys:
        - "sim_data" : np.ndarray of shape (batch_size, ...)
            Simulation outputs.
        - "full_params" : np.ndarray of shape(batch_size, num_parameters)
            Sampled set of parameters.
        """

        # Sample and collect params
        sampled_parameters = self.parameter_manager.sample(batch_size)
        params_dicts = [
            self.parameter_manager.combine(sampled_parameters, i) for i in range(batch_size)
        ]

        # Simulate
        sim_data = np.stack([
            self.model.simulate(param_dict, batch_size=1).squeeze(axis=0)
            for param_dict in params_dicts
        ])

        # Convert to matrices
        num_params = self.parameter_manager.num_parameters
        full_params = np.full((batch_size, num_params), np.nan, dtype=np.float32)

        for i, param_dict in enumerate(params_dicts):
            for j, name in enumerate(self.parameter_names):
                if name in param_dict:
                    full_params[i, j] = param_dict[name]

        return {"sim_data": sim_data, "full_params": full_params}


    def get_mask(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated binary mask for which parameters are free.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Binary mask indicating which parameters are free.
        """
        mask = self.parameter_manager.get_mask()
        return np.tile(mask, (batch_size, 1)).astype(np.float32)


    def get_conditioning_vector(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated conditioning vector with fixed/default values.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Conditioning vector with fixed/default values.
        """
        condition = self.parameter_manager.get_conditioning_vector()
        return np.tile(condition, (batch_size, 1)).astype(np.float32)
