import numpy as np
from numba import njit

@njit
def rdm_baseline_priors() -> np.ndarray:
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

@njit
def rdm_priors():
    return {
        "v":        {"intercept": np.random.gamma(3.0, 0.8),
                     "slope": 0.0},
        "v_diff":   {"intercept": np.random.normal(0.0, 2.0),
                     "slope": 0.0},
        "a":        {"intercept": np.random.gamma(10.0, 0.3),
                     "slope": 0.0},
        "tau":      {"intercept":  np.random.gamma(3.0, 0.2),
                     "slope": 0.0},
        "decay":    {"intercept": np.random.gamma(1, 0.4),
                     "slope": 0.0}
    }

@njit
def rdm_full_priors():
    return {
        "v":        {"intercept": np.random.gamma(3.0, 0.8),
                     "slope": np.random.normal(0.0, 3.0)},
        "v_diff":   {"intercept": np.random.normal(0.0, 2.0),
                     "slope": np.random.normal(0.0, 1.0)},
        "a":        {"intercept": np.random.gamma(10.0, 0.3),
                     "slope": np.random.normal(0.0, 1.0)},
        "tau":      {"intercept":  np.random.gamma(3.0, 0.2),
                     "slope": np.random.normal(0.0, 1.0)},
        "decay":    {"intercept": np.random.gamma(1, 0.4),
                     "slope": np.random.normal(0.0, 1.0)}
    }