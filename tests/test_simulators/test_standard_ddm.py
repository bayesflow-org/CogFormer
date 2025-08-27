import pytest
import numpy as np
from bayesgpt.simulators import NestedModelFamily
from bayesgpt.simulators.benchmarks import StandardDDM


def test_standard_ddm():
    model_family = NestedModelFamily(["v", "a", "z", "tau", "s_v", "sigma"])
    model_family.add_variant(
        "standard", StandardDDM, {"v": lambda bs, ctx=None: np.full(bs, 0.5)},
        {"a": 1.0, "z": 0.5, "tau": 0.3, "s_v": 0.0, "sigma": 1.0}
    )
    result = model_family.sample("standard", 10)
    assert "rts" in result["sim_data"]
    assert result["sim_data"]["rts"].shape == (10,)
    assert np.all(result["sim_data"]["choices"] >= 0)
