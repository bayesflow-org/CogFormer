import numpy as np
from collections.abc import Callable


class DDMSimulator:
    """
    Simulates response times from a Drift Diffusion Model (DDM)
    using Euler–Maruyama discretization.

    Simulates until the accumulated evidence reaches either the upper
    or lower boundary. Returns response times (RTs), ignoring choice
    for simplicity (can be added later).

    Parameters expected in the input:
    - v : float
        Drift rate.
    - a : float
        Boundary separation.
    - z : float
        Starting point as fraction of boundary (0 < z < 1).
    - t : float
        Non-decision time (added after threshold is hit).
    - eta : float
        Trial-to-trial variability in drift rate.
    - sigma : float
        Noise standard deviation in the diffusion process.
    """

    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, float], batch_size: int) -> np.ndarray:
        v_base = params["v"]
        a = params["a"]
        z_frac = params["z"]
        t_nd = params["t"]
        eta = params["eta"]
        sigma = params["sigma"]

        # Preallocate
        rts = np.zeros(batch_size)

        for i in range(batch_size):
            # Trial-specific drift rate
            v = np.random.normal(v_base, eta)

            # Initialize
            x = z_frac * a
            t = 0.0

            for _ in range(self.max_steps):
                x += v * self.dt + sigma * np.sqrt(self.dt) * np.random.randn()
                t += self.dt
                if x >= a or x <= 0:
                    rts[i] = t + t_nd
                    break
            else:
                # If max_steps reached, treat as non-response (NaN)
                rts[i] = np.nan

        return rts.reshape(-1, 1)