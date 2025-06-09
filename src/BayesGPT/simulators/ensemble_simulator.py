from bayesflow.simulators import make_simulator
import keyword


class EnsembleSimulator:
    """
    Ensemble of simulation models built on BayesFlow's make_simulator abstraction.

    This class standardizes parameter passing and output formatting, ensuring
    compatibility with BayesFlow training pipelines, while providing flexible access
    to individual simulators via both attribute- and dict-style access.

    Attribute-style access (e.g., `ensemble.model_A`) is available for simulator names
    that are valid Python identifiers and do not conflict with existing attributes or
    reserved keywords. For all simulators, dict-style access (`ensemble['model-A']`) is
    always supported and recommended for programmatic or unconventional names.

    Parameters
    ----------
    simulators_config : dict
        A dictionary mapping simulator names to configuration dictionaries.
        Each config should contain:
            - 'simulator': Callable simulation function
            - 'parameter_names': list of str

    Examples
    --------
    >>> ensemble = EnsembleSimulator(simulators_config)
    >>> ensemble.model_A(batch_size=5)            # Attribute-style (if allowed)
    >>> ensemble['model-A'](batch_size=5)         # Dict-style (always works)
    >>> for name, sim in ensemble:
    ...     print(name, sim)
    >>> ensemble.list_attribute_accessible()
    """

    def __init__(self, simulators_config):
        self.simulators = {}
        self._attr_names = set()  # For tracking attribute-accessible names
        for name, config in simulators_config.items():
            simulator_fn = config["simulator"]
            sim = make_simulator(simulator_fn)
            self.simulators[name] = sim

            # Safe attribute-style access
            if (
                name.isidentifier()
                and not keyword.iskeyword(name)
                and not hasattr(self, name)
            ):
                setattr(self, name, sim)
                self._attr_names.add(name)
            # else: always accessible via dict-style

    def get(self, name):
        """
        Retrieve a simulator by name.

        Parameters
        ----------
        name : str
            Name of the simulator to retrieve.

        Returns
        -------
        simulator : Callable
            The simulator instance associated with the given name.

        Raises
        ------
        KeyError
            If the simulator name does not exist.
        """
        if name not in self.simulators:
            raise KeyError(f"Simulator '{name}' not found in ensemble.")
        return self.simulators[name]

    def __getitem__(self, name):
        """Allow dict-style access to simulators."""
        return self.get_simulator(name)

    def __iter__(self):
        """Iterate over (name, simulator) pairs in the ensemble."""
        return iter(self.simulators.items())

    def run(self, batch_size=1, parameters=None):
        """
        Runs each simulator in the ensemble with shared or model-specific parameters.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to generate per simulator. Default is 1.
        parameters : dict, optional
            Dictionary mapping simulator names to parameter dictionaries.
            Each dict should contain the required keys defined by that simulator.

        Returns
        -------
        dict
            A dictionary mapping simulator names to their simulation outputs
            in BayesFlow format: {'sim_data': ..., 'parameters': ...}
        """
        results = {}
        for name, simulator in self.simulators.items():
            sim_params = parameters.get(name, None) if parameters else None
            if sim_params is not None:
                results[name] = simulator(batch_size=batch_size, parameters=sim_params)
            else:
                results[name] = simulator(batch_size=batch_size)
        return results

    def list_attribute_accessible(self):
        """
        List all simulator names accessible as attributes.

        Returns
        -------
        list of str
            Simulator names available via attribute-style access.
        """
        return sorted(self._attr_names)
