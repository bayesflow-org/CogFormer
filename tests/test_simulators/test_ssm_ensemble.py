import pytest


@pytest.mark.skipif(
    "ssm_ensemble" not in globals(), reason="SSMEnsemble fixture not available"
)
@pytest.mark.parametrize(
    "model_name, kwargs",
    [
        ("ddm", {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3}),
        ("angle", {"v": 1.0, "a": 1.2, "z": 0.4, "t": 0.25, "theta": 1.57}),
        (
            "weibull",
            {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.2, "alpha": 1.5, "beta": 1.5},
        ),
    ],
)
def test_ssm_ensemble_model_runs(ssm_ensemble, model_name, kwargs):
    """
    Parametrized test to verify that SSM variants simulate correctly when provided
    with all required parameters.

    Parameters
    ----------
    ssm_ensemble : SSMEnsemble
        Instance of the ensemble simulator.
    model_name : str
        The name of the SSM model.
    kwargs : dict
        Dictionary of required simulation parameters.
    """
    results = ssm_ensemble.run(batch_size=1, **kwargs)
    assert model_name in results
    assert len(results[model_name]) == 1
    output = results[model_name][0]
    assert "RT" in output and "choice" in output and "metadata" in output
