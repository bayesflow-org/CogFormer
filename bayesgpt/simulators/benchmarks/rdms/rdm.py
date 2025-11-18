import numpy as np
from numba import njit, prange
from simulators import Model
from utils.simulator_utils import as_1d

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
                return np.array([tau + i_iter * dt, i], dtype=np.float32)
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

        v_base = as_1d(params["v"], num_obs)
        v_diff = as_1d(params["v_diff"], num_obs)
        a      = as_1d(params["a"], num_obs)
        tau    = as_1d(params["tau"], num_obs)
        decay  = as_1d(params["decay"], num_obs)


        # Build per-trial K-vector drift
        v_correct   = v_base + 0.5 * v_diff
        v_incorrect = v_base - 0.5 * v_diff
        v = np.full((num_obs, num_alternatives), 0.0)
        for i in range(num_obs):
            v[i, :] = v_incorrect[i]
            v[i, correct_idx[i]] = v_correct[i]

        return {"v": v, "a": a, "tau": tau, "decay": decay}

    @staticmethod
    def build_context(num_obs: int, num_alternatives: int) -> dict[str, np.ndarray]:
        correct_idx = np.random.randint(0, num_alternatives, size=num_obs)
        return {"correct_idx": correct_idx, "num_alternatives": num_alternatives}

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        num_alternatives = np.random.randint(2, 5)  # Randomly pick 2-4 alternatives per batch
        return RDM.build_context(num_obs, num_alternatives)

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_rdm(**params, dt=self.dt, max_steps=self.max_steps)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        return {"rts": rts, "choices": choices}

