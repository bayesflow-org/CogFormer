import numpy as np
from numba import njit, prange

from bayesgpt.simulators import Model
from bayesgpt.utils.simulator_utils import as_1d


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
            return np.array([tau + i_iter * dt, np.arctan2(x[1], x[0]) / np.pi])
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

class CDM(Model):

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps

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
        Accepts either:
          - (v_x, v_y), or
          - (v, v_theta) [polar], or
          - preformed v with shape (num_obs, 2).
        Optionally uses context['theta'] (len == num_obs) to override v_theta.
        """
        context = context or {}

        v_angle  = as_1d(params["v_angle"],  num_obs, "v_angle")
        v_length = as_1d(params["v_length"], num_obs, "v_length")
        a        = as_1d(params["a"],        num_obs, "a")
        tau      = as_1d(params["tau"],      num_obs, "tau")
        decay    = as_1d(params["decay"],    num_obs, "decay")
        s_v      = as_1d(params["s_v"],      num_obs, "s_v")
        s_tau    = as_1d(params["s_tau"],     num_obs, "s_tau")
        
        # Polar -> Cartesian
        v_x = v_length * np.cos(v_angle)
        v_y = v_length * np.sin(v_angle)
        v   = np.stack([v_x, v_y], axis=1)

        # Inter-trial drift variability (isotropic)
        if np.any(s_v > 0):
            v = v + np.random.normal(0.0, s_v[:, None], size=(num_obs, 2))

        # Inter-trial nondecision variability
        if np.any(s_tau > 0):
            low  = tau - 0.5 * s_tau
            high = tau + 0.5 * s_tau
            tau  = np.random.uniform(low, high)
            tau  = np.maximum(tau, 0.0)

        return {"v": v, "a": a, "tau": tau, "decay": decay}

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
        results = simulate_cdm(**params, dt=self.dt, max_steps=self.max_steps)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        return {"rts": rts, "choices": choices}
