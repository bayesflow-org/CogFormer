from collections.abc import Callable, Sequence
from functools import partial
import numpy as np


class EnsembleSimulator:
    """
    Base class for managing and executing a collection of simulators
    with varying parameter requirements.
    """

    def __init__(self):
        """
        Initializes an empty ensemble of simulators.
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
        Adds a simulator to the ensemble.

        Parameters
        ----------
        simulator : callable
            The simulation function to register.
        simulator_name : str, optional
            A unique name for the simulator. Defaults to the function name.
        variable_names : Sequence[str], optional
            List of parameter names required by the simulator.
        fixed_variables : dict, optional
            Fixed parameters to be bound using functools.partial.

        Raises
        ------
        TypeError
            If `simulator` is not callable.
        """
        if not isinstance(simulator, Callable):
            raise TypeError(f"Provided func '{simulator}' is not callable.")

        if fixed_variables is not None:
            simulator = partial(simulator, **fixed_variables)

        self.simulators.append(
            {
                "simulator": simulator,
                "simulator_name": simulator_name or simulator.__name__,
                "variable_names": variable_names or [],
            }
        )

    def run(self, batch_size: int = None, **kwargs):
        """
        Run all compatible simulators with provided parameters.

        Each simulator runs only if all of its required parameters are present.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to generate. Inferred if not specified.
        **kwargs : dict
            All possible simulation parameters. Scalars are broadcast.

        Returns
        -------
        dict
            Dictionary of simulator outputs. Each key is a simulator name mapping
            to a list of outputs per batch sample. Simulators with missing parameters
            are skipped.
        """
        if batch_size is None:
            for v in kwargs.values():
                if isinstance(v, Sequence) and not isinstance(v, str):
                    batch_size = len(v)
                    break
            batch_size = batch_size or 1

        full_kwargs = {
            k: np.array(v)
            if isinstance(v, Sequence) and not isinstance(v, str)
            else np.full(batch_size, v)
            for k, v in kwargs.items()
        }

        results = {}
        for simulator in self.simulators:
            name = simulator["simulator_name"]
            required_params = simulator["variable_names"]
            if not all(p in full_kwargs for p in required_params):
                continue  # Skip if not all required params are provided

            results[name] = []
            for i in range(batch_size):
                local_params = {k: full_kwargs[k][i] for k in required_params}
                output = simulator["simulator"](**local_params)
                results[name].append(output)

        return results
