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
