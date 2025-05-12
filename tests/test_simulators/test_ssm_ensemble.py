import pytest


@pytest.mark.skipif("ssm_ensemble" not in globals(), reason="SSMFamily not available")
def test_ssm_ensemble(ssm_ensemble):
    results = ssm_ensemble.run(batch_size=2, v=[1.0, -1.0], a=1.5, z=0.5, t=0.3)
    assert "ddm" in results
    assert len(results["ddm"]) == 2
    for r in results["ddm"]:
        assert "RT" in r and "choice" in r
