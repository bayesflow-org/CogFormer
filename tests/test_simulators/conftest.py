import pytest
from pytest import fixture

from bayesgpt.deprecated.lightweight_ensemble_simulator import EnsembleSimulator
from bayesgpt.deprecated.benchmarks.lightweight_ddm_ensemble import DDMEnsemble
from bayesgpt.deprecated.benchmarks import SSMEnsemble


@fixture
def ensemble_simulator():
    """
    Fixture for a basic ModelFamily/EnsembleSimulator-like model to test infrastructure.

    Returns
    -------
    ModelFamily
        A testable model family containing a simple arithmetic simulator.
    """
    simulator = EnsembleSimulator()
    simulator.add(
        lambda x, y: {"sum": x + y}, variable_names=["x", "y"], simulator_name="adder"
    )
    return simulator


@fixture
def ddm_ensemble():
    """
    Fixture for initializing and reusing a DDMEnsemble instance across tests.

    Returns
    -------
    DDMEnsemble
        An initialized ensemble of DDM variants.
    """
    return DDMEnsemble()


@fixture
def ssm_ensemble():
    """
    Fixture for initializing and reusing an SSMFamily (SSM ensemble) instance.

    Returns
    -------
    SSMFamily
        An ensemble containing variants like 'ddm', 'angle', etc., if ssms is available.
    """
    if SSMEnsemble is None:
        pytest.skip("ssms library not available")
    return SSMEnsemble(models=["ddm", "angle", "weibull"])
