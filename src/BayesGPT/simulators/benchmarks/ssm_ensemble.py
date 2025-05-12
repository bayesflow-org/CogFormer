from ssms.basic_simulators.simulator import simulator as ssm_simulator
from ssms.config import model_config
from BayesGPT.simulators.ensemble_simulator import EnsembleSimulator


class SSMEnsemble(EnsembleSimulator):
    """
    A specialized ModelFamily subclass that integrates simulators from the `ssms` package.

    Automatically registers common SSM variants like 'ddm', 'angle', and 'weibull'.
    """

    def __init__(self, models=None):
        """
        Initialize and register available SSM variants.

        Parameters
        ----------
        models : list of str, optional
            List of model names from `ssms` to include. Defaults to ['ddm', 'angle', 'weibull'].
        """
        super().__init__()
        self.available_models = models or ["ddm", "angle", "weibull"]
        self._add_ssm_variants()

    def _add_ssm_variants(self):
        """
        Add each selected SSM model from `ssms` as a simulator in the family.
        """

        for model_name in self.available_models:
            config = model_config[model_name]
            param_names = config["params"]

            def make_simulator(model_name=model_name, param_names=param_names):
                def sim_func(**kwargs):
                    try:
                        theta = {p: kwargs[p] for p in param_names}
                    except KeyError as e:
                        raise KeyError(
                            f"Missing required parameter '{e.args[0]}' for model '{model_name}'"
                        )
                    result = ssm_simulator(model=model_name, theta=theta, n_samples=1)
                    return {
                        "RT": result["rts"][0],
                        "choice": result["choices"][0],
                        "metadata": result["metadata"],
                    }

                return sim_func

            sim_func = make_simulator()
            self.add(sim_func, variable_names=param_names, simulator_name=model_name)
