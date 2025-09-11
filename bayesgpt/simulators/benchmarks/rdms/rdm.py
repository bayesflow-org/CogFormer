import numpy as np
from numba import njit
from typing import Optional, Mapping
from simulators.model import Model


# @njit
# def sample_rdm_prior() -> np.ndarray:
#     v_intercept = np.random.gamma(3.0, 0.8)
#     v_diff = np.random.normal(0.0, 2.0)
#     v_slope = np.random.normal(0.0, 3.0)
#     a_intercept = np.random.gamma(10.0, 0.3)
#     a_slope = np.random.normal(0.0, 1.0)
#     lamda = np.random.gamma(1, 0.4)
#     tau = np.random.gamma(3.0, 0.2)
#     return np.array(
#         [v_intercept, v_diff, v_slope, a_intercept, a_slope, lamda, tau]
#     )
#
# @njit
# def sample_design_mats(num_trials: int, num_alternatives: int) -> np.ndarray:
#     mat_1 = np.random.random(num_trials)
#     mat_2 = np.random.random(num_trials)
#     mat_3 = np.random.randint(0, num_alternatives, num_trials)
#     return np.stack((mat_1, mat_2, mat_3), axis=1)

@njit
def simulate_rdm_trial(
    v: np.ndarray,
    a: float,
    lamda: float,
    tau: float,
    dt: float = 0.001,
    s: float = 1.0,
    max_iters: int = int(1e5),
) -> np.ndarray:
    num_alternatives = v.shape[0]
    c = s * np.sqrt(dt)
    # exponentially collapsing threshold
    t = np.arange(0, max_iters * dt, dt)
    threshold = a * np.exp(-lamda * t)
    X = np.zeros(num_alternatives, dtype=np.float32)
    for i_iter in range(max_iters):
        noise = np.random.randn(num_alternatives).astype(np.float32) * c
        for i in range(num_alternatives):
            X[i] += v[i] * dt + noise[i]
            if X[i] >= threshold[i_iter]:
                return np.array([np.float32(tau + i_iter * dt), np.float32(i)], dtype=np.float32)
    # No decision within max_iters
    return np.array([-1.0, -1.0], dtype=np.float32)


# @njit
# def sample_generative_model(
#     batch_size: int = 32,
#     num_trials: int = 200,
#     num_alternatives: int = 2,
#     dt: float = 0.001,
#     s: float = 1.0,
#     max_iters: int = int(1e5)
# ) -> tuple:
#     prior_draws = np.empty((batch_size, 7), dtype=np.float32)
#     design_mats = np.empty((batch_size, num_trials, 3), dtype=np.float32)
#     data = np.empty((batch_size, num_trials, 2), dtype=np.float32)
#     for batch in range(batch_size):
#         prior_draws[batch] = sample_rdm_prior()
#         design_mats[batch] = sample_design_mats(num_trials, num_alternatives)
#         x_v = design_mats[batch, :, 0]
#         x_a = design_mats[batch, :, 1]
#         # context dependent drift difference
#         v_diff = prior_draws[batch, 1] + prior_draws[batch, 2] * x_v
#         v_correct = prior_draws[batch, 0] + v_diff / 2
#         v_incorrect = prior_draws[batch, 0] - v_diff / 2
#         # set all drift rates to v_incorrect
#         v = v_incorrect[:, None] * np.ones((num_trials, num_alternatives))
#         # set correct alternative to v_correct
#         correct_idx = design_mats[batch, :, 2].astype(np.int32)
#         for i in range(num_trials):
#             v[i, correct_idx[i]] = v_correct[i]
#         # context dependent threshold
#         a = softplus(prior_draws[batch, 3] + prior_draws[batch, 4] * x_a)
#         for i in range(num_trials):
#             data[batch, i] = sample_rdm_trial(
#                 v[i, :], a[i], prior_draws[batch, 5],
#                 prior_draws[batch, 6], dt, s, max_iters
#             )
#     return prior_draws, design_mats, data
#
#
# class SimulationRDM:
#     def __init__(
#         self,
#         num_trials: int = 200,
#         num_alternatives: int = 2,
#         dt: float = 0.001,
#         s: float = 1.0,
#         max_iters: int = int(1e5)
#     ):
#         self.num_trials = num_trials
#         self.num_alternatives = num_alternatives
#         self.dt = dt
#         self.s = s
#         self.max_iters = max_iters
#
#     def sample(self, batch_shape) -> dict:
#         batch_size = batch_shape[0]
#         prior_draws, design_mats, data = sample_generative_model(
#             batch_size,
#             self.num_trials,
#             self.num_alternatives,
#             self.dt,
#             self.s,
#             self.max_iters
#         )
#         stim_context = to_categorical(design_mats[:, :, -1:], num_classes=4)
#         context = np.concatenate([design_mats[:, :, :2], stim_context], axis=-1)
#         return dict(
#             prior_draws=prior_draws,
#             design_mats=context,
#             data=data
#         )

# Reuse your existing numba trial kernel:
# simulate_rdm_trial(v: (K,), a: float, lamda: float, tau: float, ...)

class RDM(Model):
    """
    Rank-based DDM with context-dependent drift and bound.
    Expects the following *scalar* parameters per simulation:
      - v_intercept, v_diff, v_slope, a_intercept, a_slope, lamda, tau
    And a trial-wise context provided via `context`:
      - Either as a dict with keys {"x_v": (Ntr,), "x_a": (Ntr,), "correct_idx": (Ntr,)}
      - Or as an array of shape (Ntr, 2+K): [x_v, x_a, one-hot(correct_idx,K)]
    Returns:
      {"rts": (Ntr,), "choices": (Ntr,)}
    """

    def __init__(
        self,
        dt: float = 1e-3,
        s: float = 1.0,
        max_iters: int = int(1e5),
        num_alternatives: int = 2
    ):
        self.dt = dt
        self.s = s
        self.max_iters = max_iters
        self.num_alternatives = num_alternatives

    def prepare_params(self, params: dict[str, np.ndarray | float], num_samples: int):
        # Ensure all high-level params are scalars (collapse if vectors slipped in)
        out = dict(params)
        for k in ("v_intercept", "v_diff", "v_slope", "a_intercept", "a_slope", "lamda", "tau"):
            if k in out:
                x = np.asarray(out[k])
                out[k] = float(x.ravel()[0])
        return out

    def _unpack_context(
        self,
        context: Mapping | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Supports:
          - dict: {"x_v": (T,), "x_a": (T,), "correct_idx": (T,)}
          - array: (T, 2+K) with one-hot block for correct alternative
        """
        if isinstance(context, dict):
            x_v = np.asarray(context["x_v"], dtype=np.float32)
            x_a = np.asarray(context["x_a"], dtype=np.float32)
            correct_idx = np.asarray(context["correct_idx"], dtype=np.int32)
            return x_v, x_a, correct_idx

        arr = np.asarray(context)
        T = arr.shape[0]
        K = arr.shape[1] - 2
        x_v = arr[:, 0].astype(np.float32, copy=False)
        x_a = arr[:, 1].astype(np.float32, copy=False)
        one_hot = arr[:, 2:].astype(np.float32, copy=False)
        # argmax is safe because inputs are either one-hot or close
        correct_idx = np.argmax(one_hot, axis=1).astype(np.int32)
        # sanity fallback if K was not passed explicitly
        if self.num_alternatives != K:
            self.num_alternatives = K
        return x_v, x_a, correct_idx

    def simulate(
        self,
        params: dict[str, float],
        num_samples: int,
        context: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        if context is None:
            raise ValueError("RDMModel.simulate requires `context` for trial-wise regressors.")
        x_v, x_a, correct_idx = self._unpack_context(context)
        T = x_v.shape[0]
        K = self.num_alternatives

        v_intercept = float(params["v_intercept"])
        v_diff0     = float(params["v_diff"])
        v_slope     = float(params["v_slope"])
        a_intercept = float(params["a_intercept"])
        a_slope     = float(params["a_slope"])
        lamda       = float(params["lamda"])
        tau         = float(params["tau"])

        # Compute context-driven drift/threshold per trial
        v_diff = v_diff0 + v_slope * x_v                  # (T,)
        v_corr = v_intercept + 0.5 * v_diff               # (T,)
        v_inc  = v_intercept - 0.5 * v_diff               # (T,)
        a_t    = _softplus_vec(a_intercept + a_slope * x_a)  # (T,)

        # Allocate outputs
        rts = np.empty(T, dtype=np.float32)
        choices = np.empty(T, dtype=np.float32)

        # Allocate per-trial drift vector (K,)
        v_trial = np.empty(K, dtype=np.float32)

        # Run trials
        for t in range(T):
            # fill with v_incorrect
            for k in range(K):
                v_trial[k] = v_inc[t]
            # set correct alternative
            v_trial[int(correct_idx[t])] = v_corr[t]

            out = simulate_rdm_trial(
                v=v_trial, a=float(a_t[t]), lamda=lamda, tau=tau,
                dt=self.dt, s=self.s, max_iters=self.max_iters
            )
            rts[t] = out[0]
            choices[t] = out[1]

        return {"rts": rts, "choices": choices}
