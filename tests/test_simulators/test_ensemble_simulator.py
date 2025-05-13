import pytest


def test_ensemble_simulator(ensemble_simulator):
    results = ensemble_simulator.run(x=[1, 2], y=[10, 20])
    assert "adder" in results
    assert results["adder"][0]["sum"] == 11
    assert results["adder"][1]["sum"] == 22


@pytest.mark.parametrize(
    "subset, expected",
    [
        pytest.param({"x": [1, 2], "y": [3, 4]}, {"adder"}, id="valid-adder"),
        pytest.param({"x": [1, 2]}, set(), id="missing-y"),
    ],
)
def test_ensemble_simulator_partial(ensemble_simulator, subset, expected):
    """
    Test that the generic EnsembleSimulator properly skips models
    when required parameters are not fully specified.

    Parameters
    ----------
    ensemble_simulator : ModelFamily
        The ensemble simulator instance.
    subset : dict
        Parameters to pass into the run method.
    expected : set of str
        Simulator names expected to be present in the output.
    """
    results = ensemble_simulator.run(**subset)
    assert set(results.keys()) == expected
