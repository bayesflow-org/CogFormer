import numpy as np
from numba import njit, prange

from bayesgpt.simulators import Model
from bayesgpt.utils.simulator_utils import as_1d

@njit
def sample_rdm_trial(
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
    # Infer number of alternatives
    num_alternatives = v.shape[0]
    c = sigma * np.sqrt(dt)

    # Inter-trial variability: sample drift and non-decision time for this trial
    v_i = np.empty(num_alternatives, dtype=np.float64)
    for k in range(num_alternatives):
        v_i[k] = np.random.normal(v[k], s_v)
    tau_i = tau + np.random.uniform(0, s_tau)

    # Exponentially collapsing threshold (decay=0 means fixed threshold)
    t = np.arange(0, max_steps * dt, dt)
    threshold = a * np.exp(-decay * t)

    X = np.zeros(num_alternatives)
    for i_iter in range(max_steps):
        noise = np.random.randn(int(num_alternatives)).astype(np.float32) * c
        for i in range(num_alternatives):
            X[i] += v_i[i] * dt + noise[i]
            if X[i] >= threshold[i_iter]:
                return np.array([tau_i + i_iter * dt, i], dtype=np.float32)
    # No decision within max_steps
    return np.array([-1.0, -1.0], dtype=np.float32)


@njit(parallel=True)
def simulate_rdm(
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
        sim_trial = sample_rdm_trial(
            v=v[i],
            a=a[i],
            tau=tau[i],
            s_v=s_v[i],
            s_tau=s_tau[i],
            decay=decay,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps
        )
        sim_data[i] = sim_trial

    return sim_data

class RDM(Model):
    """
    Race Diffusion Model with response coding.

    This implementation uses response coding where drift rates are specified
    directly for each accumulator without reference to stimulus correctness.

    For 2 alternatives:
        - v: base drift rate for accumulator 0
        - v_diff: drift difference (accumulator 1 gets v + v_diff)
        - When v_diff > 0, response 1 is favored
        - When v_diff < 0, response 0 is favored

    For K > 2 alternatives (requires context):
        - v: base drift rate applied to all accumulators
        - v_diff: added to accumulator specified by context['favored_idx']
    """

    def __init__(self, dt: float = 1e-3, max_steps: int = 1e4, decay: float = 0.0):
        self.dt = dt
        self.max_steps = max_steps
        self.decay = decay  # Fixed decay parameter (0.0 = no collapsing bounds)

    def prepare_params(
        self,
        params: dict[str, np.ndarray],
        num_obs: int,
        context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Build per-trial K-way drift rates using response coding.

        Parameters
        ----------
        params : dict
            v : base drift rate
            v_diff : drift difference between accumulators
            a : threshold
            tau : non-decision time
            s_v : inter-trial drift variability (std dev)
            s_tau : inter-trial non-decision time variability (uniform half-width)

        context : dict, optional
            num_alternatives : number of accumulators (default: 2)
            favored_idx : which accumulator gets the boost (default: 1 for 2-alt)
                For K > 2 alternatives, this specifies which accumulator
                receives v + v_diff while others receive v.
        """
        context = context or {}

        num_alternatives = int(context.get("num_alternatives", 2))
        if num_alternatives < 2:
            raise ValueError("num_alternatives must be >= 2.")

        v_base = as_1d(params["v"], num_obs)
        v_diff = as_1d(params["v_diff"], num_obs)
        a      = as_1d(params["a"], num_obs)
        tau    = as_1d(params["tau"], num_obs)
        s_v    = as_1d(params["s_v"], num_obs)
        s_tau  = as_1d(params["s_tau"], num_obs)

        if num_alternatives == 2:
            # Standard 2-alternative response coding:
            # accumulator 0 gets v, accumulator 1 gets v + v_diff
            v = np.zeros((num_obs, 2), dtype=np.float64)
            v[:, 0] = v_base
            v[:, 1] = v_base + v_diff
        else:
            # K > 2 alternatives: need favored_idx from context
            if "favored_idx" not in context:
                raise ValueError(
                    "For K > 2 alternatives, context['favored_idx'] is required "
                    "to specify which accumulator receives the drift boost."
                )
            favored_idx = np.asarray(context["favored_idx"]).reshape(-1)
            if favored_idx.shape[0] != num_obs:
                raise ValueError(
                    f"favored_idx length {favored_idx.shape[0]} != num_obs {num_obs}"
                )

            # All accumulators get v_base, favored one gets v_base + v_diff
            v = np.full((num_obs, num_alternatives), 0.0, dtype=np.float64)
            for i in range(num_obs):
                v[i, :] = v_base[i]
                v[i, int(favored_idx[i])] = v_base[i] + v_diff[i]

        return {"v": v, "a": a, "tau": tau, "s_v": s_v, "s_tau": s_tau}

    @staticmethod
    def build_context(num_obs: int, num_alternatives: int = 2) -> dict[str, np.ndarray]:
        """Build context for RDM simulation."""
        context = {"num_alternatives": num_alternatives}
        if num_alternatives > 2:
            # Randomly assign favored accumulator for each trial
            context["favored_idx"] = np.random.randint(0, num_alternatives, size=num_obs)
        return context

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        """Build default context (2 alternatives, no favored_idx needed)."""
        return {"num_alternatives": 2}

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_rdm(**params, decay=self.decay, dt=self.dt, max_steps=self.max_steps)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        return {"rts": rts, "choices": choices}

    def sample(self):
        raise NotImplementedError
