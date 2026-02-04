import numpy as np
from numba import njit
from scipy import norm


def cdm_priors():
    return {
        "v": norm(loc=1.0, scale=0.4),
        "v_theta": norm(loc=0.0, scale=1.5),
        "a": norm(loc=0.25, scale=0.5),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.0, scale=0.6),
        "s_tau": norm(loc=-1.5, scale=0.7),
    }

@njit
def cdm_baseline_priors() -> np.ndarray:
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

@njit
def cdm_full_priors():
    return {
        "v":        {"intercept": np.random.normal(1.0, 2.0),
                     "slope": 0.0},
        "a":        {"intercept": np.random.gamma(10.0, 0.3),
                     "slope": 0.0},
        "tau":      {"intercept": np.random.gamma(3.0, 0.2),
                     "slope": 0.0},
        "decay":    {"intercept": np.random.gamma(1, 0.4),
                     "slope": 0.0},
        "s_v":      {"intercept": np.random.gamma(1, 0.2),
                     "slope": 0.0},
        "s_tau":    {"intercept": np.random.uniform(0, 2.0),    # Ideally, upper bound = tau * 2
                     "slope": 0.0},
    }

@njit
def cdm_custom_priors():
    return {
        "v":        {"intercept": np.random.normal(1.0, 2.0),
                     "slope": np.random.normal(0.0, 3.0)},
        "a":        {"intercept": np.random.gamma(10.0, 0.3),
                     "slope": np.random.normal(0.0, 1.0)},
        "tau":      {"intercept": np.random.gamma(3.0, 0.2),
                     "slope": np.random.normal(0.0, 1.0)},
        "decay":    {"intercept": np.random.gamma(1, 0.4),
                     "slope": np.random.normal(0.0, 1.0)},
        "s_v":      {"intercept": np.random.gamma(1, 0.2),
                     "slope": np.random.normal(0.0, 1.0)},
        "s_tau":    {"intercept": np.random.uniform(0, 2.0),
                     "slope": np.random.normal(0.0, 1.0)}
    }