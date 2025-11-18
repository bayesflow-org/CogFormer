import numpy as np


def ddm_baseline_priors():
    return {
        "v":     np.random.gamma(1.5, 0.5),
        "a":     np.random.gamma(8.0, 0.2),
        # "decay": 0.0 if flat_bound else np.random.gamma(1.0, 0.4),
        "tau":   np.random.gamma(3.0, 0.2),
        "s_v":   np.random.gamma(1.0, 0.2),
        "s_tau": np.random.uniform(0.0, 0.4),
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
        "v":        {"intercept": lambda: np.random.gamma(2.5, 0.5),
                     "slope": lambda: 0.0},
        "a":        {"intercept": lambda: np.random.normal(0, 0.05),
                     "slope": lambda: 0.0},
        "tau":      {"intercept": lambda: np.random.normal(-1.0, 0.3),
                     "slope": lambda: 0.0},
        "s_v":      {"intercept": lambda: np.random.normal(-1.2, 0.5),
                     "slope": lambda: 0.0},
        "s_tau":    {"intercept": lambda: np.random.normal(0.5, 0.2),
                     "slope": lambda: 0.0}
    }
