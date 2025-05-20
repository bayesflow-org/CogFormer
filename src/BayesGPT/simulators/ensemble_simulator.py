from bayesflow.simulators import make_simulator


class EnsembleSimulator:
    """
    Ensemble of simulation models built on BayesFlow's make_simulator abstraction.

    This version standardizes parameter passing and output formatting, ensuring
    compatibility with BayesFlow training pipelines while retaining flexibility
    for custom ensembles.
    """

    def __init__(self, simulators_config):
        """
        Initializes an ensemble of BayesFlow-compatible simulators.

        Parameters
        ----------
        simulators_config : dict
            A dictionary mapping simulator names to configuration dictionaries.
            Each config should contain:
                - 'simulator': Callable simulation function
                - 'parameter_names': list of str
        """
        self.simulators = {}
        for name, config in simulators_config.items():
            simulator_fn = config["simulator"]
            parameter_names = config["parameter_names"]
            self.simulators[name] = make_simulator(
                simulator_fn, parameter_names=parameter_names
            )

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
