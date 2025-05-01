from collections.abc import Callable, Sequence
from functools import partial

import numpy as np


class SimulatorFamily:
    def __init__(self):
        super().__init__()
        self.simulators = []

    def add(
        self,
        simulator: Callable,
        simulator_name: str = None,
        variable_names: Sequence[str] = None,
        fixed_variables: dict = None,
    ):
        if fixed_variables is not None:
            simulator = partial(simulator, **fixed_variables)

        self.simulators.append(
            {
                "simulator": simulator,
                "simulator_name": simulator_name,
                "variable_names": variable_names or [],
            }
        )

    def run(self, batch_size: int = None, **kwargs):
        if batch_size is None:
            for v in kwargs.values():
                if isinstance(v, Sequence) and not isinstance(v, str):
                    batch_size = len(v)
                    break
            batch_size = batch_size or 1

        # Broadcast all parameters to batch size
        full_kwargs = {
            k: np.array(v)
            if isinstance(v, Sequence) and not isinstance(v, str)
            else np.full(batch_size, v)
            for k, v in kwargs.items()
        }

        # Run each model
        results = {model["name"]: [] for model in self.simulators}
        for i in range(batch_size):
            for simulator in self.simulators:
                local_params = {
                    k: full_kwargs[k][i]
                    for k in simulator["params"]
                    if k in full_kwargs
                }
                output = simulator["simulator"](**local_params)
                results[simulator["simulator_name"]].append(output)

        return results  # Results are lists of dictionaries, one per model
