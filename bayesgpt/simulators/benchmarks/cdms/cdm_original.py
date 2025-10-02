import numpy as np
from numba import njit, prange
from utils.simulator_utils import shifted_softplus

@njit
def sample_cdm_prior() -> np.ndarray:
    mu_intercept = np.random.normal(1, 2)
    mu_theta = 2.0*np.pi * (np.random.beta(3.0, 3.0) - 0.5)
    mu_slope = np.random.normal(0, 2)
    mu_var = np.random.gamma(1, 0.2)
    a_intercept = np.random.gamma(10.0, 0.3)
    a_slope = np.random.normal(0.0, 1.0)
    lamda = np.random.gamma(1, 0.4)
    tau = np.random.gamma(3.0, 0.2)
    tau_var = np.random.uniform(0, tau*2)
    return np.array(
        [
            mu_intercept, mu_theta, mu_slope, mu_var,
            a_intercept, a_slope, lamda, tau, tau_var
        ]
    )

@njit
def sample_design_mats(num_trials: int) -> np.ndarray:
    mat_1 = np.random.random(num_trials)
    mat_2 = np.random.random(num_trials)
    return np.stack((mat_1, mat_2), axis=1)

@njit
def sample_cdm_trial(
    mu: np.ndarray,
    a: float,
    lamda: float,
    tau: float,
    dt: float = 0.001,
    s: float = 1.0,
    max_iters: int = int(1e5),
) -> np.ndarray:
    c = np.sqrt(dt) * s
    # exponentially collapsing threshold
    t = np.arange(0, max_iters * dt, dt)
    threshold = a * np.exp(-lamda * t)
    x = np.zeros(2)
    for i_iter in range(max_iters):
        x += mu*dt + c * np.random.randn(2)
        if np.linalg.norm(x, 2) >= threshold[i_iter]:
            return np.array([tau + i_iter * dt, np.arctan2(x[1], x[0])])
    # No decision within max_iters
    return np.array([-1.0, -1.0])

@njit(parallel=True)
def sample_generative_model(
    batch_size: int = 32,
    num_trials: int = 200,
    dt: float = 0.001,
    s: float = 1.0,
    max_iters: int = int(1e5)
) -> tuple:
    prior_draws = np.empty((batch_size, 9), dtype=np.float32)
    design_mats = np.empty((batch_size, num_trials, 2), dtype=np.float32)
    data = np.empty((batch_size, num_trials, 2), dtype=np.float32)

    for batch in prange(batch_size):
        prior_draws[batch] = sample_cdm_prior()
        design_mats[batch] = sample_design_mats(num_trials)
        x_v = design_mats[batch, :, 0]
        x_a = design_mats[batch, :, 1]
        # precompute drift direction
        cos_theta = np.cos(prior_draws[batch, 1])
        sin_theta = np.sin(prior_draws[batch, 1])
        # bounds for non-decision time variability
        low = prior_draws[batch, 7] - prior_draws[batch, 8] / 2.0
        high = prior_draws[batch, 7] + prior_draws[batch, 8] / 2.0

        # loop once across trials
        for i in range(num_trials):
            # context-dependent drift length
            drift_length = shifted_softplus(
                prior_draws[batch, 0] + prior_draws[batch, 2] * x_v[i]
            )
            # base mu
            mu0 = drift_length * cos_theta
            mu1 = drift_length * sin_theta
            # drift inter-trial variability
            mu0 += np.random.normal(0.0, prior_draws[batch, 3])
            mu1 += np.random.normal(0.0, prior_draws[batch, 3])
            # context-dependent threshold
            a = shifted_softplus(
                prior_draws[batch, 4] + prior_draws[batch, 5] * x_a[i]
            )
            # non-decision time inter-trial variability
            ndt = np.random.uniform(low, high)
            # simulate trial
            data[batch, i] = sample_cdm_trial(
                np.array([mu0, mu1], dtype=np.float32),
                a, prior_draws[batch, 6],
                ndt, dt, s, max_iters,
            )

    return prior_draws, design_mats, data

class SimulationCDM:
    def __init__(
        self,
        num_trials: int = 200,
        dt: float = 0.001,
        s: float = 1.0,
        max_iters: int = int(1e5)
    ):
        self.num_trials = num_trials
        self.dt = dt
        self.s = s
        self.max_iters = max_iters

    def sample(self, batch_shape) -> dict:
        batch_size = batch_shape[0]
        prior_draws, design_mats, data = sample_generative_model(
            batch_size,
            self.num_trials,
            self.dt,
            self.s,
            self.max_iters
        )
        return dict(
            prior_draws=prior_draws,
            design_mats=design_mats,
            data=data
        )