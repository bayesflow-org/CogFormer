import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all simulators.

    All model subclasses must implement a `simulate` method that accepts
    a parameter dictionary and returns a simulation output.
    """

    @abstractmethod
    def simulate(self, params: dict[str, float], batch_size: int) -> np.ndarray:
        pass
