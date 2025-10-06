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

        def as_1d(x):
            x = np.asarray(x).reshape(-1)
            if x.size == 1:
                x = np.full((num_obs,), x.item())
            if x.size != num_obs:
                raise ValueError(f"Expected length {num_obs}, got {x.size}")
            return x

        # Build v (num_obs, 2)
        if "v" in params and np.asarray(params["v"]).ndim == 2 and np.asarray(params["v"]).shape[1] == 2:
            v_vec = np.asarray(params["v"])
            if v_vec.shape[0] != num_obs:
                raise ValueError(f"v has {v_vec.shape[0]} rows; expected {num_obs}")
        elif "v_x" in params and "v_y" in params:
            v_x = as_1d(params["v_x"])
            v_y = as_1d(params["v_y"])
            v_vec = np.stack([v_x, v_y], axis=1)
        elif "v" in params and "v_theta" in params:
            v_mag = as_1d(params["v"])
            v_th = as_1d(params["v_theta"])
            # Allow context override of angles (optional, RDM-inspired)
            if "theta" in context:
                theta_ctx = as_1d(context["theta"])
                v_th = theta_ctx
            v_vec = np.stack([v_mag * np.cos(v_th), v_mag * np.sin(v_th)], axis=1)
        else:
            raise ValueError("Provide (v_x,v_y) or (v,v_theta) or v as (n,2).")

        a = as_1d(params["a"])
        tau = as_1d(params["tau"])
        decay = as_1d(params["decay"])

        return {
            "v": v_vec.astype(np.float32, copy=False),
            "a": a.astype(np.float32, copy=False),
            "tau": tau.astype(np.float32, copy=False),
            "decay": decay.astype(np.float32, copy=False),
        }

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
        print(f"Theta mode: {theta_mode}")
        return CDM.build_context(num_obs, theta_mode)

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_cdm(**params, dt=self.dt, max_steps=self.max_steps)
        return {"rts": results[:, 0], "choices": results[:, 1]}
