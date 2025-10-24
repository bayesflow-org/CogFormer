import numpy as np
from numba import njit, prange
from simulators import Model


@njit
def simulate_ddm_trial(
    v: float,
    a: float,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    z: float = 0.0,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> (float, float):

    v_i = np.random.normal(v, s_v)
    tau_i = tau
    if s_tau > 0.0:
        low = tau - s_tau * 0.5
        high = tau + s_tau * 0.5
        tau_i = np.random.uniform(low, high)
        if tau_i < 0.0:
            tau_i = 0.0

    # initialize
    a0 = a if a > 1e-6 else 1e-6
    d  = decay if decay > 0.0 else 0.0

    x = z * a0
    t = tau_i

    for _ in range(max_steps):
        t += dt
        bound = a0 * np.exp(-d * t)
        if bound < 1e-3:
            bound = 1e-3
        x += v_i * dt + sigma * np.sqrt(dt) * np.random.normal()
        if x >= bound:
            return np.array([t, 1.0], dtype=np.float32)
        if x <= -bound:
            return np.array([t, 0.0], dtype=np.float32)
    # No decision within max_steps
    return np.array([-1.0, -1.0], dtype=np.float32)

@njit(parallel=True)
def simulate_ddm(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    s_tau: np.ndarray,
    s_v: np.ndarray,
    decay: np.ndarray,
    z: float = 0.5,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:

    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)

    for i in prange(n):
        sim_trial = simulate_ddm_trial(
            v=v[i],
            a=a[i],
            tau=tau[i],
            s_tau=s_tau[i],
            s_v=s_v[i],
            decay=decay[i],
            z=z,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        sim_data[i] = sim_trial

    return sim_data

@njit
def sample_ddm_prior():
    v_intercept = np.random.gamma(3.0, 0.8)
    v_slope     = np.random.normal(0.0, 3.0)
    s_v         = np.random.gamma(1.0, 0.2)
    a_intercept = np.random.gamma(10.0, 0.3)
    a_slope     = np.random.normal(0.0, 1.0)
    decay       = np.random.gamma(1.0, 0.4)
    tau         = np.random.gamma(3.0, 0.2)
    s_tau       = np.random.uniform(0.0, tau * 2.0)
    return np.array([v_intercept, v_slope, s_v, a_intercept, a_slope, decay, tau, s_tau], dtype=np.float32)


class DDM(Model):

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4):
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_ddm(**params, z=0.5, sigma=1.0, dt=self.dt, max_steps=self.max_steps)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        return {"rts": rts, "choices": choices}

    def sample(self, batch_size: int, params: dict[str, np.ndarray], context=None):
        rts = []
        choices = []
        for i in range(batch_size):
            results = self.simulate(params=params, context=context)
            rts.append(results["rts"])
            choices.append(results["choices"])

        return {"rts": np.array(rts), "choices": np.array(choices)}