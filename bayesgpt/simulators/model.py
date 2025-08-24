import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from collections.abc import Mapping


class Model(ABC):
    """
    Abstract base class for all simulators.

    All model subclasses must implement a `simulate` method that accepts
    a parameter dictionary and returns a simulation output as either a
    NumPy array or a mapping containing simulation data.
    """

    @abstractmethod
    def simulate(
        self, params: dict[str, float], batch_size: int
    ) -> Union[np.ndarray, Mapping[str, np.ndarray]]:
        pass
