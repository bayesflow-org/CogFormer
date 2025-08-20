import numpy as np
from ..model import Model


class SuperDDM(Model):
    """
    Simulates response times and choices from a Drift Diffusion Model (DDM)
    using Euler–Maruyama discretization under full variability consideration.

    Accumulation continues until evidence reaches upper or lower boundary.
    Returns both RT and binary choice (1 = upper, 0 = lower).

    Parameters expected in the input:
    - v     : float, drift rater
    - s_v   : float, variability in drift rate
    - a     : float, boundary separation
    - z     : float, starting point as fraction of boundary (0 < z < 1)
    - s_z   : float, variability in starting point
    - tau   : float, non-decision time
    - s_tau : float, variability in non-decision time
    - angle : float, slope of collapsing bounds
    - sigma : float, diffusion noise
    """
    def __init__(self, dt: float=0.1, max_steps: int=10000):
        self.dt = dt
        self.max_steps = max_steps


    def simulate(self, params: dict[str, float], batch_size: int) -> np.ndarray:

        v = params["v"]
        s_v = params["s_v"]
        a = params["a"]
        z = params["z"]
        s_z = params["s_z"]
        tau = params["tau"]
        s_tau = params["s_tau"]
        angle = params["angle"]


        rts = np.zeros(batch_size)
        choices = np.zeros(batch_size)

        return np.stack([rts, choices], axis=1)
