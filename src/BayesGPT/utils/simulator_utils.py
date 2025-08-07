import numpy as np
from collections.abc import Callable
from ..simulators.base_simulator import BaseSimulator


class ParameterHandler:
    """
    Handles sampling of free parameters from user-defined callables
    and merging with fixed parameters for simulation input.

    This class assumes all free parameters are defined as callables that
    accept a batch size `n` and return an array of `n` samples.

    Parameters
    ----------
    free_parameters : dict of {str: Callable[[int], np.ndarray]}
        Dictionary mapping each free parameter name to a callable function
        that takes an integer `batch_size` and returns a 1D NumPy array of samples.

    fixed_parameters : dict of {str: float}
        Dictionary of fixed parameter names and their scalar values.

    Attributes
    ----------
    parameter_order : list of str
        List of free parameter names in the order they are defined.
    """

    def __init__(
            self,
            free_parameters: dict[str, Callable[[int], np.ndarray]],
            fixed_parameters: dict
    ):
        self.free_parameters = free_parameters
        self.fixed_parameters = fixed_parameters
        self.parameter_order = list(free_parameters.keys())


    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Samples a batch of values for each free parameter.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate for each parameter.

        Returns
        -------
        dict of {str: np.ndarray}
            A dictionary mapping free parameter names to sampled values.
            Each array has shape (batch_size,).
        """
        return {
            name: sampler(batch_size)
            for name, sampler in self.free_parameters.items()
        }


    def merge(self, theta_batch: dict[str, np.ndarray], index: int) -> dict:
        """
        Combines a single sampled instance of free parameters with fixed parameters.

        Parameters
        ----------
        theta_batch : dict of {str: np.ndarray}
            A batch of sampled free parameters. Each array should have shape (batch_size,).

        index : int
            The index of the sample to extract from the batch.

        Returns
        -------
        dict of {str: float}
            A dictionary containing the full parameter set for a single simulation.
        """
        merged = {k: v[index] for k, v in theta_batch.items()}
        merged.update(self.fixed_parameters)
        return merged
