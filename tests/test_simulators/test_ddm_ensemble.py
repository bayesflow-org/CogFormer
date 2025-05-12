import pytest


@pytest.mark.parametrize(
    "simulator_name, kwargs",
    [
        ("basic", dict(drift=1.0, boundary=1.5)),
        ("with_ndt", dict(drift=0.8, boundary=1.2, ndt=0.3)),
        ("trajectory", dict(drift=0.7, boundary=1.1, ndt=0.2)),
        (
            "collapsing_bound",
            dict(drift=1.0, initial_boundary=1.3, ndt=0.25, collapse_rate=0.05),
        ),
    ],
)
def test_ddm_ensemble(ddm_ensemble, simulator_name, kwargs):
    results = ddm_ensemble.run(simulator_name=simulator_name, batch_size=2, **kwargs)
    assert isinstance(results, list) and len(results) == 2
    for r in results:
        assert "RT" in r and "choice" in r
