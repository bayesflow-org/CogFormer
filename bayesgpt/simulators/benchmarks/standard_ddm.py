import numpy as np
from typing import Union
from numba import njit, prange
from ..model import Model


class StandardDDM(Model):
    """
    Simulates response times and choices from a Drift Diffusion Model (DDM)
    using Euler-Maruyama discretization.

    Accumulation continues until evidence reaches upper or lower boundary.
    Returns both RT and binary choice (1 = upper, 0 = lower).
    """

    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        self.dt = dt
        self.max_steps = max_steps

    def simulate(
        self,
        params: dict[str, Union[float, np.ndarray]],
        num_trials: int = 1
    ) -> dict[str, np.ndarray]:
        """
        Simulate response times and choices for a batch of trials.

        Parameters
        ----------
        params : dict[str, float]
            Dictionary containing model parameters: v, a, z, tau, s_v, sigma.
        num_trials : int
            Number of trials to simulate.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys 'rts' and 'choices', each mapping to an array
            of shape (num_trials,) containing response times and choices.

        Raises
        ------
        ValueError
            If required parameters are missing or have invalid values.
        """

        # Sanity checks
        required_params = ["v", "a", "z", "tau", "s_v", "sigma"]
        if not all(k in params for k in required_params):
            raise ValueError(f"Missing parameters: {set(required_params) - set(params)}")

        # Convert parameters to arrays if scalars
        params = {k: np.full(num_trials, v) if np.isscalar(v) else v for k, v in params.items()}

        if not all(p.shape == (num_trials,) for p in params.values()):
            raise ValueError("All parameters must be scalars or arrays of shape (num_trials,)")
        
        # Validate
        if np.any(params["a"] <= 0) or np.any(params["sigma"] <= 0) or np.any((params["z"] <= 0) | (params["z"] >= 1)):
            raise ValueError("Invalid parameter values: a, sigma must be > 0, 0 < z < 1")
        if np.any(params["tau"] < 0) or np.any(params["s_v"]) < 0:
            raise ValueError("Invalid parameter values: tau, s_v must be >= 0")

        # Unpack params
        v = params["v"]
        a = params["a"]
        z = params["z"]
        tau = params["tau"]
        s_v = params["s_v"]
        sigma = params["sigma"]

        # Simulate
        result = _simulate_standard_ddm(
            v, a, z, tau, s_v, sigma, self.dt, self.max_steps, num_trials
        )
        return {"rts": result[:, 0], "choices": result[:, 1]}


@njit
def _simulate_standard_ddm(
    v: float | np.ndarray,
    a: float | np.ndarray,
    z: float | np.ndarray,
    tau: float | np.ndarray,
    s_v: float | np.ndarray,
    sigma: float | np.ndarray,
    dt: float,
    max_steps: int,
    num_trials: int,
) -> np.ndarray:
    """
    Internal function to simulate the standard DDM.

    Parameters
    ----------
    v : float or np.ndarray
        Drift rate.
    a : float or np.ndarray
        Boundary separation.
    z : float or np.ndarray
        Starting point as fraction of boundary (0 < z < 1).
    tau : float or np.ndarray
        Non-decision time.
    s_v : float or np.ndarray
        Drift variability.
    sigma : float or np.ndarray
        Diffusion noise.

    Returns
    -------
    np.ndarray of shape (num_trials, 2) for the following simulated data:
        - rts: reaction time for each trial
        - choices: choices for each trial
    """

    # Create buffers
    rts = np.zeros(num_trials)
    choices = np.zeros(num_trials)

    # Simulate
    for i in prange(num_trials):
        vi = np.random.normal(v, s_v)
        x = z * a
        t = 0.0

        for step in range(max_steps):
            x += vi * dt + sigma * np.sqrt(dt) * np.random.normal()
            t += dt
            if x >= a:
                rts[i] = t + tau
                choices[i] = 1
                break
            elif x <= 0:
                rts[i] = t + tau
                choices[i] = 0
                break
        else:
            rts[i] = np.nan
            choices[i] = np.nan

    # Store results
    result = np.zeros((num_trials, 2))
    result[:, 0] = rts
    result[:, 1] = choices
    return result
