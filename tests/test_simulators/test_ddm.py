import pytest
import numpy as np
from bayesgpt.simulators import NestedModelFamily
from bayesgpt.simulators.benchmarks import DDM

@pytest.fixture(scope="module")
def test_standard_ddm():
    ddm_priors = {
        "v": {"intercept": lambda: np.random.gamma(3.0, 0.8),
              "slope": lambda: np.random.normal(0.0, 3.0)},
        "a": {"intercept": lambda: np.random.gamma(10.0, 0.3),
              "slope": lambda: np.random.normal(0.0, 1.0)},
        "tau": {"intercept": lambda: np.random.gamma(3.0, 0.2),
                "slope": lambda: 0.0},
        "s_v": {"intercept": lambda: np.random.gamma(1.0, 0.2),
                "slope": lambda: 0.0},
        "s_tau": {"intercept": lambda: np.random.uniform(0.0, 0.4),
                  "slope": lambda: 0.0},
        "decay": {"intercept": lambda: np.random.gamma(1.0, 0.4),
                  "slope": lambda: 0.0},
    }

    model_family = NestedModelFamily(name="ddm", model=DDM(), prior_fun=ddm_priors)
    result = model_family.sample(
        batch_size=10,
        mask_randomizer_kwargs=dict(
            free_intrinsics={"v", "a", "tau", "s_v", "decay"},
            fixed_intrinsics={"s_tau"}
        ),
        num_obs=10,
        flatten_param_outputs=True
    )
    assert "rts" in result["sim_data"]
