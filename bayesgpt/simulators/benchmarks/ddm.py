import numpy as np
from ..model import Model

# Accelerate with Numba or jax.jit
# Also, come up with a SuperDDM with variability for all parameters

class StandardDDM(Model):
    """
    Simulates response times and choices from a Drift Diffusion Model (DDM)
    using Euler–Maruyama discretization.

    Accumulation continues until evidence reaches upper or lower boundary.
    Returns both RT and binary choice (1 = upper, 0 = lower).

    Parameters expected in the input:
    - v     : float, drift rate
    - a     : float, boundary separation
    - z     : float, starting point as fraction of boundary (0 < z < 1)
    - t     : float, non-decision time
    - eta   : float, drift variability
    - sigma : float, diffusion noise
    """

    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, float], batch_size: int) -> np.ndarray:
        v_base = params["v"]
        a = params["a"]
        z_frac = params["z"]
        t_nd = params["tau"]
        eta = params["eta"]
        sigma = params["sigma"]

        rts = np.zeros(batch_size)
        choices = np.zeros(batch_size)

        for i in range(batch_size):
            v = np.random.normal(v_base, eta)
            x = z_frac * a
            t = 0.0

            for _ in range(self.max_steps):
                x += v * self.dt + sigma * np.sqrt(self.dt) * np.random.randn()
                t += self.dt
                if x >= a:
                    rts[i] = t + t_nd
                    choices[i] = 1
                    break
                elif x <= 0:
                    rts[i] = t + t_nd
                    choices[i] = 0
                    break
            else:
                rts[i] = np.nan
                choices[i] = np.nan

        return np.stack([rts, choices], axis=1)


class CollapsingBoundDDM(StandardDDM):
    """
    Drift Diffusion Model with symmetric linearly collapsing bounds.

    Inherits from StandardDDM and overrides simulate logic to implement
    dynamic thresholds:
        B(t) = a - angle * t

    Parameters
    ----------
    Same as StandardDDM, plus:
    - angle : float
        Collapse rate. If 0, reduces to standard DDM.
    """

    def simulate(self, params: dict[str, float], batch_size: int) -> np.ndarray:
        v_base = params["v"]
        a = params["a"]
        z_frac = params["z"]
        t_nd = params["tau"]
        eta = params["eta"]
        sigma = params["sigma"]
        angle = params["angle"]

        rts = np.zeros(batch_size)
        choices = np.zeros(batch_size)

        for i in range(batch_size):
            v = np.random.normal(v_base, eta)
            x = (2 * z_frac - 1.0) * a  # symmetric init between -a and +a
            t = 0.0

            for _ in range(self.max_steps):
                t += self.dt
                bound = max(a - angle * t, 1e-3)

                x += v * self.dt + sigma * np.sqrt(self.dt) * np.random.randn()

                if x >= bound:
                    rts[i] = t + t_nd
                    choices[i] = 1
                    break
                elif x <= -bound:
                    rts[i] = t + t_nd
                    choices[i] = 0
                    break
            else:
                rts[i] = np.nan
                choices[i] = np.nan

        return np.stack([rts, choices], axis=1)
