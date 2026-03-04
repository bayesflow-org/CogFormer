import numpy as np
from numba import njit, prange

from bayesgpt.simulators import Model
from bayesgpt.utils.simulator_utils import as_1d


@njit
def simulate_cdm_trial(
    v: np.ndarray,
    a: float,
    tau: float,
    s_v: float,
    s_tau: float,
    decay: float = 0.0,
    dt: float = 0.001,
    sigma: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    # Inter-trial variability
    v_i = v + s_v * np.random.randn(2)
    tau_i = tau + np.random.uniform(0, s_tau)

    c = np.sqrt(dt) * sigma
    # exponentially collapsing threshold
    t = np.arange(0, max_steps * dt, dt)
    threshold = a * np.exp(-decay * t)
    x = np.zeros(2)
    for i_iter in range(max_steps):
        x += v_i * dt + c * np.random.randn(2)
        if np.linalg.norm(x, 2) >= threshold[i_iter]:
            return np.array([tau_i + i_iter * dt, np.arctan2(x[1], x[0])])
    # No decision within max_steps
    return np.array([-4.0, -4.0])

@njit(parallel=True)
def simulate_cdm(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    decay: float = 0.0,
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
            tau=tau[i],
            s_v=s_v[i],
            s_tau=s_tau[i],
            decay=decay,
            dt=dt,
            sigma=sigma,
            max_steps=max_steps
        )
        sim_data[i] = sim_trial

    return sim_data

class CDM(Model):

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4, decay: float = 0.0):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.decay = decay

    def prepare_params(
            self,
            params: dict[str, np.ndarray],
            num_obs: int,
            context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        CDM expects:
          - v: (num_obs, 2) Cartesian drift per trial
          - a, tau, decay: (num_obs,)
        Polar form: v (magnitude) and v_theta (angle in radians).
        """
        context = context or {}

        v_theta  = as_1d(params["v_theta"], num_obs, "v_theta")
        v        = as_1d(params["v"],       num_obs, "v")
        a        = as_1d(params["a"],       num_obs, "a")
        tau      = as_1d(params["tau"],     num_obs, "tau")
        s_v      = as_1d(params["s_v"],     num_obs, "s_v")
        s_tau    = as_1d(params["s_tau"],   num_obs, "s_tau")

        # Polar -> Cartesian
        v_x = v * np.cos(v_theta)
        v_y = v * np.sin(v_theta)
        v   = np.stack([v_x, v_y], axis=1)

        return {"v": v, "a": a, "tau": tau, "s_v": s_v, "s_tau": s_tau}

    @staticmethod
    def build_context(num_obs: int, theta_mode: str = "random_uniform") -> dict[str, np.ndarray]:
        """
        Optional helper to generate per-trial headings when using (v, v_theta).
        theta_mode:
          - 'random_uniform': theta ~ U[-pi, pi)
          - 'zeros': theta = 0 for all trials
        """
        if theta_mode == "random_uniform":
            theta = np.random.uniform(-np.pi, np.pi, size=num_obs)
        elif theta_mode == "zeros":
            theta = np.zeros(num_obs)
        else:
            raise ValueError("Unsupported theta_mode.")
        return {"theta": theta}

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        theta_mode = np.random.choice(["random_uniform", "zeros"])  # Randomly pick per batch
        return CDM.build_context(num_obs, theta_mode)

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_cdm(**params, decay=self.decay, dt=self.dt, max_steps=self.max_steps)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        return {"rts": rts, "choices": choices}
