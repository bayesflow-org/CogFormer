import numpy as np
from numba import njit, prange
from simulators import Model


@njit
def simulate_cdm_trial(
    v: np.ndarray,
    a: float,
    decay: float,
    tau: float,
    dt: float = 0.001,
    sigma: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    c = np.sqrt(dt) * sigma
    # exponentially collapsing threshold
    t = np.arange(0, max_steps * dt, dt)
    threshold = a * np.exp(-decay * t)
    x = np.zeros(2)
    for i_iter in range(max_steps):
        x += v * dt + c * np.random.randn(2)
        if np.linalg.norm(x, 2) >= threshold[i_iter]:
            return np.array([tau + i_iter * dt, np.arctan2(x[1], x[0])/np.pi])
    # No decision within max_steps
    return np.array([-1.0, -1.0])

@njit(parallel=True)
def simulate_cdm(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    decay: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000
):
    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)

    for i in prange(n):
        sim_trial = simulate_cdm_trial(
            v=v[i],
            a=a[i],
            decay=decay[i],
            tau=tau[i],
            dt=dt,
            sigma=sigma,
            max_steps=max_steps
        )
        sim_data[i] = sim_trial

    return sim_data

@njit
def sample_cdm_prior() -> np.ndarray:
    v_intercept = np.random.normal(1, 2)
    v_theta = 2.0 * np.pi * (np.random.beta(3.0, 3.0) - 0.5)
    v_slope = np.random.normal(0, 2)
    s_v = np.random.gamma(1, 0.2)
    a_intercept = np.random.gamma(10.0, 0.3)
    a_slope = np.random.normal(0.0, 1.0)
    decay = np.random.gamma(1, 0.4)
    tau = np.random.gamma(3.0, 0.2)
    s_tau = np.random.uniform(0, tau*2)
    return np.array(
        [
            v_intercept, v_theta, v_slope, s_v,
            a_intercept, a_slope, decay, tau, s_tau
        ]
    )

class CDM(Model):

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_cdm(**params, dt=self.dt, max_steps=self.max_steps)
        return {"rts": results[:, 0], "choices": results[:, 1]}
