import numpy as np
from numba import njit
from utils.simulator_utils import _softplus, _softplus_vec


@njit
def sample_cdm_trial(
    mu: np.ndarray,
    a: float,
    lamda: float,
    tau: float,
    dt: float = 0.001,
    s: float = 1.0,
    max_iters: int = int(1e5),
) -> np.ndarray:
    c = np.sqrt(dt) * s
    # exponentially collapsing threshold
    t = np.arange(0, max_iters * dt, dt)
    threshold = a * np.exp(-lamda * t)
    x = np.zeros(2)
    for i_iter in range(max_iters):
        x += mu*dt + c * np.random.randn(2)
        if np.linalg.norm(x, 2) >= threshold[i_iter]:
            return np.array([tau + i_iter * dt, np.arctan2(x[1], x[0])/np.pi])
    # No decision within max_iters
    return np.array([-1.0, -1.0])
