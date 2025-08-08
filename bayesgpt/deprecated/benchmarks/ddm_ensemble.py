import numpy as np
from bayesgpt.deprecated.ensemble_simulator import EnsembleSimulator

# --- Simulator functions (unchanged) ---


def ddm_basic(drift, boundary):
    """
    Basic Drift Diffusion Model (DDM) simulator.

    Parameters
    ----------
    drift : float
        Drift rate.
    boundary : float
        Decision boundary.

    Returns
    -------
    dict
        Dictionary with simulated 'RT' and 'choice'.
    """

    rt = np.abs(boundary / drift) + np.random.normal(0, 0.1)
    choice = int(drift > 0)
    return {"RT": rt, "choice": choice}


def ddm_with_ndt(drift, boundary, ndt):
    """
    DDM with non-decision time (NDT).

    Parameters
    ----------
    drift : float
        Drift rate.
    boundary : float
        Decision boundary.
    ndt : float
        Non-decision time.

    Returns
    -------
    dict
        Dictionary with simulated 'RT', 'choice', and 'ndt'.
    """

    rt = np.abs(boundary / drift) + ndt + np.random.normal(0, 0.1)
    choice = int(drift > 0)
    return {"RT": rt, "choice": choice, "ndt": ndt}


def ddm_with_trajectory(drift, boundary, ndt):
    """
    DDM that tracks evidence accumulation over time.

    Parameters
    ----------
    drift : float
        Drift rate.
    boundary : float
        Decision boundary.
    ndt : float
        Non-decision time.

    Returns
    -------
    dict
        Dictionary with 'RT', 'choice', and 'trajectory'.
    """

    dt = 0.01
    x = 0
    trajectory = [x]
    for _ in range(100):
        x += drift * dt + np.random.normal(0, 0.1)
        trajectory.append(x)
        if np.abs(x) >= boundary:
            break
    rt = len(trajectory) * dt + ndt
    return {"RT": rt, "choice": int(x > 0), "trajectory": trajectory}


def ddm_collapsing_bound(drift, initial_boundary, ndt, collapse_rate):
    """
    DDM with a collapsing decision boundary over time.

    Parameters
    ----------
    drift : float
        Drift rate.
    initial_boundary : float
        Initial decision boundary.
    ndt : float
        Non-decision time.
    collapse_rate : float
        Linear collapse rate per time step.

    Returns
    -------
    dict
        Dictionary with 'RT', 'choice', 'trajectory', and 'final_bound'.
    """

    dt = 0.01
    x = 0
    trajectory = [x]
    for t in range(100):
        current_boundary = max(0.1, initial_boundary - collapse_rate * t * dt)
        x += drift * dt + np.random.normal(0, 0.1)
        trajectory.append(x)
        if np.abs(x) >= current_boundary:
            break
    rt = len(trajectory) * dt + ndt
    return {
        "RT": rt,
        "choice": int(x > 0),
        "trajectory": trajectory,
        "final_bound": current_boundary,
    }


class DDMEnsemble(EnsembleSimulator):
    """
    BayesFlow-compatible ensemble of DDM variants.

    Supports both attribute- and dict-style access to all DDM variant simulators.
    Example:
        ensemble = DDMEnsemble()
        result = ensemble.basic(batch_size=5, parameters={'drift': 1.0, 'boundary': 1.0})  # Attribute access
        result2 = ensemble['collapsing_bound'](batch_size=5, parameters={...})             # Dict access
        for name, sim in ensemble:
            print(name, sim)
        print("Attribute-accessible:", ensemble.list_attribute_accessible())
    """

    def __init__(self):
        simulators_config = {
            "basic": {
                "simulator": ddm_basic,
                "parameter_names": ["drift", "boundary"],
            },
            "with_ndt": {
                "simulator": ddm_with_ndt,
                "parameter_names": ["drift", "boundary", "ndt"],
            },
            "trajectory": {
                "simulator": ddm_with_trajectory,
                "parameter_names": ["drift", "boundary", "ndt"],
            },
            "collapsing_bound": {
                "simulator": ddm_collapsing_bound,
                "parameter_names": [
                    "drift",
                    "initial_boundary",
                    "ndt",
                    "collapse_rate",
                ],
            },
        }
        super().__init__(simulators_config)
