import numpy as np
from numba import njit, prange
from keras.utils import to_categorical
from utils.simulator_utils import shifted_softplus

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

@njit
def sample_design_mats(num_trials: int, num_alternatives: int) -> np.ndarray:
    mat_1 = np.random.random(num_trials)
    mat_2 = np.random.random(num_trials)
    mat_3 = np.random.randint(0, num_alternatives, num_trials)
    return np.stack((mat_1, mat_2, mat_3), axis=1)

@njit
def sample_rdm_trial(
    v: np.ndarray,
    a: float,
    lamda: float,
    tau: float,
    dt: float = 0.001,
    s: float = 1.0,
    max_iters: int = int(1e4),
) -> np.ndarray:
    num_alternatives = v.shape[0]
    c = s * np.sqrt(dt)
    # exponentially collapsing threshold
    t = np.arange(0, max_iters * dt, dt)
    threshold = a * np.exp(-lamda * t)
    X = np.zeros(num_alternatives)
    for i_iter in range(max_iters):
        noise = np.random.randn(int(num_alternatives)).astype(np.float32) * c
        for i in range(num_alternatives):
            X[i] += v[i] * dt + noise[i]
            if X[i] >= threshold[i_iter]:
                return np.array([np.float32(tau + i_iter * dt), np.float32(i)])
    # No decision within max_iters
    return np.array([-1.0, -1.0], dtype=np.float32)

@njit(parallel=True)
def sample_generative_model(
    batch_size: int = 32,
    num_trials: int = 200,
    num_alternatives: int = 2,
    dt: float = 0.001,
    s: float = 1.0,
    max_iters: int = int(1e5)
) -> tuple:
    prior_draws = np.empty((batch_size, 7), dtype=np.float32)
    design_mats = np.empty((batch_size, num_trials, 3), dtype=np.float32)
    data = np.empty((batch_size, num_trials, 2), dtype=np.float32)

    for batch in prange(batch_size):
        prior_draws[batch] = sample_rdm_prior()
        design_mats[batch] = sample_design_mats(num_trials, num_alternatives)
        x_v = design_mats[batch, :, 0]
        x_a = design_mats[batch, :, 1]
        correct_idx = design_mats[batch, :, 2].astype(np.int32)

        for i in range(num_trials):
            # context dependent drift difference
            v_diff = prior_draws[batch, 1] + prior_draws[batch, 2] * x_v[i]
            v_correct = prior_draws[batch, 0] + v_diff / 2.0
            v_incorrect = prior_draws[batch, 0] - v_diff / 2.0
            # drift rates for all alternatives
            v = np.full(num_alternatives, v_incorrect, dtype=np.float32)
            v[correct_idx[i]] = v_correct
            # context dependent threshold
            a = shifted_softplus(prior_draws[batch, 3] + prior_draws[batch, 4] * x_a[i])
            # simulate trial
            data[batch, i] = sample_rdm_trial(
                v, a, prior_draws[batch, 5],
                prior_draws[batch, 6], dt,
                s, max_iters,
            )

    return prior_draws, design_mats, data

class SimulationRDM:
    def __init__(
        self,
        num_trials: int = 200,
        num_alternatives: int = 2,
        dt: float = 0.001,
        s: float = 1.0,
        max_iters: int = int(1e5)
    ):
        self.num_trials = num_trials
        self.num_alternatives = num_alternatives
        self.dt = dt
        self.s = s
        self.max_iters = max_iters

    def sample(self, batch_shape) -> dict:
        batch_size = batch_shape[0]
        prior_draws, design_mats, data = sample_generative_model(
            batch_size,
            self.num_trials,
            self.num_alternatives,
            self.dt,
            self.s,
            self.max_iters
        )
        stim_context = to_categorical(design_mats[:, :, -1:], num_classes=4)
        context = np.concatenate([design_mats[:, :, :2], stim_context], axis=-1)
        return dict(
            prior_draws=prior_draws,
            design_mats=context,
            data=data
        )
