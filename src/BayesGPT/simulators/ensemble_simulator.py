from collections.abc import Callable, Sequence
from functools import partial

import numpy as np


class EnsembleSimulator:
    def __init__(self):
        """
        Initialize an empty model family.

        Each model is stored with its function, parameter names, and optional name.
        """
        super().__init__()
        self.simulators = []

    def add(
        self,
        simulator: Callable,
        simulator_name: str = None,
        variable_names: Sequence[str] = None,
        fixed_variables: dict = None,
    ):
        """
        Add a model to the family.

        Parameters
        ----------
        simulator : callable
            The model function to add.
        variable_names : list of str, optional
            Names of parameters the model expects.
        simulator_name : str, optional
            Name to assign to the model. Defaults to the function name.
        fixed_variables : dict, optional
            Dictionary of parameter names and values to be fixed using `functools.partial`.

        Raises
        ------
        TypeError
            If `func` is not callable.
        """
        if not isinstance(simulator, Callable):
            raise TypeError(f"Provided func '{simulator}' is not callable.")

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
        """
        Run all models across a batch of input values.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to run per model. If not provided, inferred from the length
            of the first sequence-type parameter.
        **kwargs : dict
            Parameter values. Each key should match one or more of the model parameter names.

        Returns
        -------
        dict
            Dictionary of model outputs. Each key is a model name mapping to a list
            of outputs for each batch sample.
        """
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
        results = {simulator["simulator_name"]: [] for simulator in self.simulators}
        for i in range(batch_size):
            for simulator in self.simulators:
                local_params = {
                    k: full_kwargs[k][i]
                    for k in simulator["variable_names"]
                    if k in full_kwargs
                }
                output = simulator["simulator"](**local_params)
                results[simulator["simulator_name"]].append(output)

        return results  # Results are lists of dictionaries, one per model
