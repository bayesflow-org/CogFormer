from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Union, Optional
import numpy as np


class Model(ABC):
    """
    Abstract base class for simulation-based inference models.

    Defines the interface for simulating data with num_samples trials per simulation,
    with batching handled by NestedModelFamily.
    """

    @abstractmethod
    def simulate(
        self,
        params: dict[str, np.ndarray],
        num_samples: int,
        context: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Mapping[str, np.ndarray]]:
        """
        Simulate data for a single model run with num_samples trials.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Parameters for simulation, with arrays of shape (dims,) or scalars.
        num_samples : int
            Number of samples (trials) per simulation.
        context : np.ndarray, optional
            Context array to condition the simulation, shape (context_shape,).

        Returns
        -------
        Union[np.ndarray, Mapping[str, np.ndarray]]
            Simulated data, shape (num_samples, ...) for arrays or a mapping with arrays of shape (num_samples, ...), in np.float32.
        """
        pass