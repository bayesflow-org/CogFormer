import pytest


@pytest.mark.parametrize(
    "simulator_name, kwargs, expected_keys",
    [
        pytest.param(
            "ddm",
            {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3},
            {"RT", "choice", "metadata"},
        ),
        pytest.param(
            "angle",
            {"v": 0.7, "a": 1.2, "z": 0.4, "t": 0.25, "theta": 1.57},
            {"RT", "choice", "metadata"},
        ),
        pytest.param(
            "weibull",
            {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.2, "alpha": 1.5, "beta": 1.5},
            {"RT", "choice", "metadata"},
        ),
    ],
)
def test_ssm_ensemble_variants(ssm_ensemble, simulator_name, kwargs, expected_keys):
    """
    Test that each SSMEnsemble variant produces valid outputs when provided the correct parameters.

    Parameters
    ----------
    ssm_ensemble : SSMEnsemble
        The simulator ensemble containing SSM variants.
    simulator_name : str
        The name of the variant being tested.
    kwargs : dict
        The input parameters for the simulator.
    expected_keys : set
        The expected keys in the simulator's output dictionary.
    """
    results = ssm_ensemble.run(batch_size=1, **kwargs)
    assert simulator_name in results
    assert (
        isinstance(results[simulator_name], list) and len(results[simulator_name]) == 1
    )
    output = results[simulator_name][0]
    assert expected_keys.issubset(output.keys())


@pytest.mark.parametrize(
    "present_params, expected_simulators",
    [
        pytest.param({"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3}, {"ddm"}, id="only-ddm"),
        pytest.param(
            {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3, "theta": 1.57},
            {"ddm", "angle"},
            id="ddm-angle",
        ),
        pytest.param(
            {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3, "alpha": 1.5, "beta": 1.5},
            {"ddm", "weibull"},
            id="ddm-weibull",
        ),
    ],
)
def test_ssm_partial_parameter_skip(ssm_ensemble, present_params, expected_simulators):
    """
    Parametrized test that checks which SSM simulators are run
    when only a subset of all possible parameters is provided.

    Parameters
    ----------
    ssm_ensemble : SSMEnsemble
        The simulator ensemble containing SSM variants.
    present_params : dict
        Dictionary of parameters to be passed to the ensemble.
    expected_simulators : set of str
        Set of simulator names expected to run given the provided parameters.
    """
    results = ssm_ensemble.run(batch_size=1, **present_params)
    for name in expected_simulators:
        assert name in results
        assert isinstance(results[name], list) and len(results[name]) == 1
    for name in {"ddm", "angle", "weibull"} - expected_simulators:
        assert name not in results
