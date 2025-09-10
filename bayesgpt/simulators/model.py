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
        params: dict[str, np.ndarray | float],
        num_samples: int,
        context: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Mapping[str, np.ndarray]]:

        pass