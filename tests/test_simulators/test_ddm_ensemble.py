import pytest


@pytest.mark.parametrize(
    "simulator_name, kwargs, expected_keys",
    [
        pytest.param("basic", dict(drift=1.0, boundary=1.5), {"RT", "choice"}),
        pytest.param(
            "with_ndt", dict(drift=0.8, boundary=1.2, ndt=0.3), {"RT", "choice", "ndt"}
        ),
        pytest.param(
            "trajectory",
            dict(drift=0.7, boundary=1.1, ndt=0.2),
            {"RT", "choice", "trajectory"},
        ),
        pytest.param(
            "collapsing_bound",
            dict(drift=1.0, initial_boundary=1.3, ndt=0.25, collapse_rate=0.05),
            {"RT", "choice", "trajectory", "final_bound"},
        ),
    ],
)
def test_ddm_variants(ddm_ensemble, simulator_name, kwargs, expected_keys):
    results = ddm_ensemble.run(simulator_name=simulator_name, batch_size=2, **kwargs)
    assert isinstance(results, list) and len(results) == 2
    for r in results:
        assert expected_keys.issubset(r.keys())
