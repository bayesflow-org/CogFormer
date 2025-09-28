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
    sigma: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    # Infer number of alternatives
    num_alternatives = v.shape[0]
    c = sigma * np.sqrt(dt)
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
    sigma: float = 1.0,
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
            sigma=sigma,
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
    decay = np.random.gamma(1, 0.4)
    tau = np.random.gamma(3.0, 0.2)
    return np.array(
        [v_intercept, v_diff, v_slope, a_intercept, a_slope, decay, tau]
    )

class RDM(Model):

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4):
        self.dt = dt
        self.max_steps = max_steps

    def prepare_params(
        self,
        params: dict[str, np.ndarray],
        num_obs: int,
        context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Build per-trial K-way drift from scalars; no dtype control here.
        Expects in `params`: v, optional v_diff, a, tau, decay (len == num_obs or scalars).
        Expects in `context`: 'correct_idx' (len == num_obs), optional 'num_alternatives'.
        """
        context = context or {}

        if "correct_idx" not in context:
            raise ValueError("RDM requires context['correct_idx'].")
        correct_idx = np.asarray(context["correct_idx"]).reshape(-1)
        if correct_idx.shape[0] != num_obs:
            raise ValueError(f"correct_idx length {correct_idx.shape[0]} != num_obs {num_obs}")

        num_alternatives = int(context.get("num_alternatives", int(correct_idx.max()) + 1))
        if num_alternatives < 1:
            raise ValueError("num_alternatives must be >= 1.")

        # Helper: broadcast to (num_obs,) if scalar; validate length; no dtype conversion.
        def as_1d(x):
            a = np.asarray(x).reshape(-1)
            if a.size == 1:
                a = np.full((num_obs,), a.item())
            if a.size != num_obs:
                raise ValueError(f"Parameter has length {a.size}, expected {num_obs}.")
            return a

        v_base = as_1d(params["v"])
        v_diff = as_1d(params.get("v_diff", np.zeros_like(v_base)))
        a      = as_1d(params["a"])
        tau    = as_1d(params["tau"])
        decay  = as_1d(params["decay"])

        # Build per-trial K-vector drift
        v_correct   = v_base + 0.5 * v_diff
        v_incorrect = v_base - 0.5 * v_diff
        v = np.full((num_obs, num_alternatives), 0.0)
        for i in range(num_obs):
            v[i, :] = v_incorrect[i]
            v[i, correct_idx[i]] = v_correct[i]

        return {"v": v, "a": a, "tau": tau, "decay": decay}

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_rdm(**params, dt=self.dt, max_steps=self.max_steps)
        return {"rts": results[:, 0], "choices": results[:, 1]}
