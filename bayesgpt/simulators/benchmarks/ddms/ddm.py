import numpy as np
from numba import njit, prange
from scipy.stats import halfnorm

from .ddm_priors import ddm_baseline_priors

from bayesgpt.simulators import Model


@njit
def simulate_ddm_trial(
    v: float,
    a: float,
    tau: float,
    s_v: float,
    s_tau: float,
    decay: float = 0.0,
    z: float = 0.0,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
):
    v_i = np.random.normal(v, s_v)
    tau_i = tau + np.random.uniform(-s_tau * tau, s_tau * tau)

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
    s_v: np.ndarray,
    s_tau: np.ndarray,
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
            z=z,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        sim_data[i] = sim_trial

    return sim_data

class DDM(Model):

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4):
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_ddm(**params, z=0.5, sigma=1.0, dt=self.dt, max_steps=self.max_steps)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        return {"rts": rts, "choices": choices}

    def sample(self, batch_size: int | tuple, num_obs: int = 500, context=None):
        # Infer batch size if none exist
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        params = {k: [] for k in ddm_baseline_priors().keys()}
        rts = []
        choices = []
        # prior_draws = sample_ddm_baseline_priors()
        # for k, v in prior_draws.items():
        #     prior_draws[k] = np.zeros((batch_size, 1), dtype=np.float32)

        for i in range(batch_size):
            prior_draw = ddm_baseline_priors()
            for k, v in prior_draw.items():
                params[k].append(v)
                # prior_draws[k][i] = v
                # priors[k] = np.full(num_obs, v)
            priors = {k: np.full(num_obs, v) for k, v in prior_draw.items()}
            results = self.simulate(params=priors, context=context)
            rts.append(results["rts"])
            choices.append(results["choices"])

        prior_draws = {k: np.array(v)[:, None] for k, v in params.items()}
        sim_data = {"rts": np.array(rts), "choices": np.array(choices)}
        return prior_draws | sim_data