from bayesgpt.deprecated.ensemble_simulator import EnsembleSimulator
from ssms.basic_simulators.simulator import simulator as ssm_simulator
from ssms.config import model_config


def make_ssm_simulator(model_name, param_names):
    """
    Factory for SSM-compatible simulator functions.

    Parameters
    ----------
    model_name : str
        Name of the model in `ssms`.
    param_names : list of str
        Names of the model parameters.

    Returns
    -------
    callable
        A simulation function compatible with BayesFlow.
    """

    def sim_func(**kwargs):
        try:
            theta = {p: kwargs[p] for p in param_names}
        except KeyError as e:
            raise KeyError(
                f"Missing required parameter '{e.args[0]}' for model '{model_name}'"
            )

        result = ssm_simulator(model=model_name, theta=theta, n_samples=1)
        output = {
            "RT": result["rts"][0],
            "choice": result["choices"][0],
        }
        output.update(result["metadata"])
        return output

    return sim_func


class SSMEnsemble(EnsembleSimulator):
    """
    BayesFlow-compatible ensemble of Sequential Sampling Models (SSMs)
    using the `ssms` package.
    """

    def __init__(self, models=None):
        models = models or ["ddm", "angle", "weibull", "lca", "race"]
        simulators_config = {}

        for model_name in models:
            config = model_config[model_name]
            param_names = config["params"]
            simulators_config[model_name] = {
                "simulator": make_ssm_simulator(model_name, param_names),
                "parameter_names": param_names,
            }

        super().__init__(simulators_config)
