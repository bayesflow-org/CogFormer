import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod
from collections.abc import Mapping


class Model(ABC):
    """
    Abstract base class for simulation-based inference models.

    Defines the interface for simulating data with num_obs trials per simulation,
    with batching handled by NestedModelFamily.
    """

    def prepare_params(
            self,
            params: dict[str, float],
            num_obs: int,
    ) -> dict[str, float]:
        """Optional: models can normalize/broadcast shapes here. Default: passthrough."""
        return params

    @abstractmethod
    def simulate(
        self,
        params: dict[str, np.ndarray],
        context: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Mapping[str, np.ndarray]]:

        pass
