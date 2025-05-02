import numpy as np
from pytest import fixture


@fixture(scope="session")
def test_simulator_family():
    from BayesGPT.simulators.simulator_family import SimulatorFamily

    family = SimulatorFamily()

    def simulator1():
        return np.random.normal(0.0, 1.0, size=100)

    def simulator2():
        return np.random.normal(0.0, 0.1, size=100)

    family.add(simulator=simulator1, simulator_name="s1")
    family.add(simulator=simulator2, simulator_name="s2")

    results = family.run(batch_size=2)

    return results
