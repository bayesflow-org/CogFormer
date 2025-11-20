import numpy as np
from scipy.stats import halfnorm


def ddm_baseline_priors():
    return {
        "v":     np.random.gamma(2., 1.),
        # "v":     np.random.normal(1., 1.),
        "a":     np.random.normal(-.1, 0.3),
        # "decay": 0.0 if flat_bound else np.random.gamma(1.0, 0.4),
        "tau":   np.random.normal(-1.5, 0.3),
        "s_v":   halfnorm(loc=0.0, scale=1.0),
        "s_tau": np.random.beta(1.0, 3.0),
    }

def ddm_test_priors():
    return {
        "v": np.random.gamma(2.5, 0.5, size=1000),
        "a": np.random.lognormal(0, 0.5, size=1000),
        "tau": np.random.gamma(2.0, 0.2, size=1000),
        "s_v":  np.random.gamma(0.5, 0.2, size=1000),
        "s_tau": np.random.gamma(0.5, 0.2, size=1000),
    }

def ddm_priors():
    return {
        "v":        {"intercept": lambda: np.random.gamma(2.5, 0.5),
                     "slope": lambda: 0.0},
        "a":        {"intercept": lambda: np.random.lognormal(0, 0.5),
                     "slope": lambda: 0.0},
        "tau":      {"intercept": lambda: np.random.gamma(2.0, 0.2),
                     "slope": lambda: 0.0},
        "s_v":      {"intercept": lambda: np.random.gamma(0.5, 0.2),
                     "slope": lambda: 0.0},
        "s_tau":    {"intercept": lambda: np.random.gamma(0.5, 0.2),
                     "slope": lambda: 0.0}
    }

def ddm_log_priors():
    return {
        "v":        {"intercept": lambda: np.random.gamma(2., 1.),
                     "slope": lambda: 0.0},
        "a":        {"intercept": lambda: np.random.normal(-1, 0.3),
                     "slope": lambda: 0.0},
        "tau":      {"intercept": lambda: np.random.normal(-1.5, 0.3),
                     "slope": lambda: 0.0},
        "s_v":      {"intercept": lambda: halfnorm(loc=0.0, scale=1.0),
                     "slope": lambda: 0.0},
        "s_tau":    {"intercept": lambda: np.random.beta(1.0, 3.0),
                     "slope": lambda: 0.0}
    }

