import numpy as np
from collections.abc import Callable
from .base_simulator import BaseSimulator
from ..utils.simulator_utils import ParameterHandler


class VariantSimulator:
    """
    A configurable variant of a simulation model.

    Encapsulates a specific combination of free and fixed parameters
    along with a model class, enabling parameter sampling and simulation.

    Parameters
    ----------
    name : str
        Identifier for this variant.

    model_cls : type[BaseModel]
        A subclass of `BaseModel` to instantiate as the simulator.

    free_parameters : dict of {str: Callable[[int], np.ndarray]}
        Dictionary of free parameter names mapped to sampling callables.

    fixed_parameters : dict of {str: float}
        Dictionary of fixed parameter names mapped to constant values.
    """

    def __init__(
        self,
        name: str,
        model_cls: type[BaseSimulator],
        free_parameters: dict[str, Callable[[int], np.ndarray]],
        fixed_parameters: dict
    ):
        self.name = name
        self.model = model_cls()
        self.param_handler = ParameterHandler(free_parameters, fixed_parameters)

    def sample(self, batch_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Samples a batch of free parameters and simulates data.

        Parameters
        ----------
        batch_size : int
            Number of simulation samples to generate.

        Returns
        -------
        x_batch : np.ndarray
            Simulated model output of shape (batch_size, ...).

        theta_batch : dict of {str: np.ndarray}
            Dictionary of free parameters, each of shape (batch_size,).
        """
        theta_batch = self.param_handler.sample(batch_size)
        x_batch = []

        for i in range(batch_size):
            full_params = self.param_handler.merge(theta_batch, i)
            x = self.model.simulate(full_params, batch_size=1)
            x_batch.append(x[0])  # shape (1, ...) → take [0]

        return np.stack(x_batch), theta_batch
