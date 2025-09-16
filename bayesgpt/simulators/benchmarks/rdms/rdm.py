import numpy as np
from numba import njit, prange
from simulators import Model

@njit
def sample_rdm_trial(
    v: np.ndarray,
    a: float,
    decay: float,
    tau: float,
    dt: float = 0.001,
    s: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    num_alternatives = v.shape[0]
    c = s * np.sqrt(dt)
    # exponentially collapsing threshold
    t = np.arange(0, max_steps * dt, dt)
    threshold = a * np.exp(-decay * t)
    X = np.zeros(num_alternatives)
    for i_iter in range(max_steps):
        noise = np.random.randn(int(num_alternatives)).astype(np.float32) * c
        for i in range(num_alternatives):
            X[i] += v[i] * dt + noise[i]
            if X[i] >= threshold[i_iter]:
                return np.array([np.float32(tau + i_iter * dt), np.float32(i)])
    # No decision within max_steps
    return np.array([-1.0, -1.0], dtype=np.float32)

@njit(parallel=True)
def simulate_rdm(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    decay: np.ndarray,
    s: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000
):
    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)

    for i in prange(n):
        sim_trial = sample_rdm_trial(
            v=v[i],
            a=a[i],
            decay=decay[i],
            tau=tau[i],
            s=s,
            dt=dt,
            max_steps=max_steps
        )
        sim_data[i] = sim_trial

    return sim_data

@njit
def sample_rdm_prior() -> np.ndarray:
    v_intercept = np.random.gamma(3.0, 0.8)
    v_diff = np.random.normal(0.0, 2.0)
    v_slope = np.random.normal(0.0, 3.0)
    a_intercept = np.random.gamma(10.0, 0.3)
    a_slope = np.random.normal(0.0, 1.0)
    lamda = np.random.gamma(1, 0.4)
    tau = np.random.gamma(3.0, 0.2)
    return np.array(
        [v_intercept, v_diff, v_slope, a_intercept, a_slope, lamda, tau]
    )

class RDM(Model):

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_rdm(**params, dt=self.dt, max_steps=self.max_steps)
        return {"rts": results[:, 0], "choices": results[:, 1]}
