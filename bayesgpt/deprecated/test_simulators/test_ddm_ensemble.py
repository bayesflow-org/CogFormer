import pytest


@pytest.mark.parametrize(
    "simulator_name, kwargs, expected_keys",
    [
        pytest.param("basic", {"drift": 1.0, "boundary": 1.5}, {"RT", "choice"}),
        pytest.param(
            "with_ndt",
            {"drift": 0.8, "boundary": 1.2, "ndt": 0.3},
            {"RT", "choice", "ndt"},
        ),
        pytest.param(
            "trajectory",
            {"drift": 0.7, "boundary": 1.1, "ndt": 0.2},
            {"RT", "choice", "trajectory"},
        ),
        pytest.param(
            "collapsing_bound",
            {"drift": 1.0, "initial_boundary": 1.3, "ndt": 0.25, "collapse_rate": 0.05},
            {"RT", "choice", "trajectory", "final_bound"},
        ),
    ],
)
def test_ddm_ensemble_variants(ddm_ensemble, simulator_name, kwargs, expected_keys):
    """
    Test that each DDMEnsemble variant produces valid outputs when provided the correct parameters.

    Parameters
    ----------
    ddm_ensemble : DDMEnsemble
        The simulator ensemble containing DDM variants.
    simulator_name : str
        The name of the variant being tested.
    kwargs : dict
        The input parameters for the simulator.
    expected_keys : set
        The expected keys in the simulator's output dictionary.
    """
    results = ddm_ensemble.run(batch_size=1, **kwargs)
    assert simulator_name in results
    assert (
        isinstance(results[simulator_name], list) and len(results[simulator_name]) == 1
    )
    output = results[simulator_name][0]
    assert expected_keys.issubset(output.keys())


@pytest.mark.parametrize(
    "present_params, expected_simulators",
    [
        pytest.param({"drift": 1.0, "boundary": 1.5}, {"basic"}, id="only-basic"),
        pytest.param(
            {"drift": 1.0, "boundary": 1.5, "ndt": 0.3},
            {"basic", "with_ndt", "trajectory"},
            id="basic-with_ndt-trajectory",
        ),
        pytest.param(
            {"drift": 1.0, "initial_boundary": 1.2, "ndt": 0.25, "collapse_rate": 0.05},
            {"collapsing_bound"},
            id="only-collapsing",
        ),
    ],
)
def test_ddm_partial_parameter_skip(ddm_ensemble, present_params, expected_simulators):
    """
    Parametrized test that checks which DDM simulators are run
    when only a subset of all possible parameters is provided.

    Parameters
    ----------
    ddm_ensemble : DDMEnsemble
        The simulator ensemble containing DDM variants.
    present_params : dict
        Dictionary of parameters to be passed to the ensemble.
    expected_simulators : set of str
        Set of simulator names expected to run given the provided parameters.
    """

    results = ddm_ensemble.run(batch_size=1, **present_params)
    for name in expected_simulators:
        assert name in results
        assert isinstance(results[name], list) and len(results[name]) == 1
    for name in {
        "basic",
        "with_ndt",
        "trajectory",
        "collapsing_bound",
    } - expected_simulators:
        assert name not in results
