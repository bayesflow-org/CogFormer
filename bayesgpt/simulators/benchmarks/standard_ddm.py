import numpy as np
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
        self, params: dict[str, float], batch_size: int
    ) -> dict[str, np.ndarray]:
        """
        Simulate response times and choices for a batch of trials.

        Parameters
        ----------
        params : dict[str, float]
            Dictionary containing model parameters: v, a, z, tau, s_v, sigma.
        batch_size : int
            Number of trials to simulate.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys 'rts' and 'choices', each mapping to an array
            of shape (batch_size,) containing response times and choices.

        Raises
        ------
        ValueError
            If required parameters are missing or have invalid values.
        """

        # Sanity checks
        required_params = ["v", "a", "z", "tau", "s_v", "sigma"]
        if not all(k in params for k in required_params):
            raise ValueError(f"Missing parameters: {set(required_params) - set(params)}")
        if params["a"] <= 0 or params["sigma"] <= 0 or not (0 < params["z"] < 1):
            raise ValueError("Invalid parameter values: a, sigma must be > 0, 0 < z < 1")
        if params["tau"] < 0 or params["s_v"] < 0:
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
            v, a, z, tau, s_v, sigma, self.dt, self.max_steps, batch_size
        )
        return {"rts": result[:, 0], "choices": result[:, 1]}


@njit
def _simulate_standard_ddm(
    v: float,
    a: float,
    z: float,
    tau: float,
    s_v: float,
    sigma: float,
    dt: float,
    max_steps: int,
    batch_size: int,
) -> np.ndarray:
    """
    Internal function to simulate the standard DDM.

    Parameters
    ----------
    v : float
        Drift rate.
    a : float
        Boundary separation.
    z : float
        Starting point as fraction of boundary (0 < z < 1).
    tau : float
        Non-decision time.
    s_v : float
        Drift variability.
    sigma : float
        Diffusion noise.

    Returns
    -------
    np.ndarray of shape (batch_size, 2) for the following simulated data:
        - rts: reaction time for each trial
        - choices: choices for each trial
    """

    # Create buffers
    rts = np.zeros(batch_size)
    choices = np.zeros(batch_size)

    # Simulate
    for i in prange(batch_size):
        vi = np.random.normal(v, s_v)
        x = z * a
        t = 0.0

        for _ in range(max_steps):
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
    result = np.zeros((batch_size, 2))
    result[:, 0] = rts
    result[:, 1] = choices
    return result
