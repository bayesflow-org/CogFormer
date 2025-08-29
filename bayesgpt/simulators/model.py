from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Union, Literal
import numpy as np

class Model(ABC):
    """
    Abstract base class for simulation-based inference models.
    """

    @abstractmethod
    def simulate(
        self,
        params: dict[str, np.ndarray],
        batch_size: int,
        num_samples: int,
        flatten: bool = True
    ) -> Union[np.ndarray, Mapping[str, np.ndarray]]:
        """
        Simulate data from the model.

        Parameters
        ----------
        params : dict of {str: np.ndarray}
            Parameters for simulation, with arrays of shape (batch_size, dims).
        batch_size : int
            Number of simulations to run.
        num_samples : int
            Number of samples per simulation (e.g., trials in DDM).
        flatten : bool
            Whether to flatten the simulated data.

        Returns
        -------
        np.ndarray or Mapping[str, np.ndarray]
            Simulated data (e.g., response times or choices) in np.float32.
        """
        pass
