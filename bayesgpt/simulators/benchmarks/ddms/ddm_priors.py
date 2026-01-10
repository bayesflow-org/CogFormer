import numpy as np
from scipy.stats import halfnorm


def ddm_baseline_priors():
    return {
        "v":        lambda: np.random.gamma(2., 1.),
        "a":        lambda: np.random.normal(-.1, 0.3),
        "tau":      lambda: np.random.normal(-1.5, 0.3),
        "s_v":      lambda: halfnorm.rvs(loc=0.0, scale=1.0),
        "s_tau":    lambda: np.random.beta(1.0, 3.0),
        # "decay":  np.random.gamma(1.0, 0.4),
    }

def ddm_test_priors():
    return {
        "v":        np.random.gamma(2.5, 0.5, size=1000),
        "a":        np.random.lognormal(0, 0.5, size=1000),
        "tau":      np.random.gamma(2.0, 0.2, size=1000),
        "s_v":      np.random.gamma(0.5, 0.2, size=1000),
        "s_tau":    np.random.gamma(0.5, 0.2, size=1000),
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
        "s_v":      {"intercept": lambda: halfnorm.rvs(loc=0.0, scale=1.0),
                     "slope": lambda: 0.0},
        "s_tau":    {"intercept": lambda: np.random.beta(1.0, 3.0),
                     "slope": lambda: 0.0}
    }

def ddm_full_priors():
    return {
        "v":        {"intercept": lambda: np.random.gamma(2., 1.),
                     "slope": lambda: np.random.normal(0., 1.)},
        "a":        {"intercept": lambda: np.random.normal(-1, 0.3),
                     "slope": lambda: np.random.normal(0., 1.)},
        "tau":      {"intercept": lambda: np.random.normal(-1.5, 0.3),
                     "slope": lambda: np.random.normal(0., 0.5)},
        "s_v":      {"intercept": lambda: halfnorm.rvs(loc=0.0, scale=1.0),
                     "slope": lambda: np.random.normal(0., 1.)},
        "s_tau":    {"intercept": lambda: np.random.beta(1.0, 3.0),
                     "slope": lambda: np.random.normal(0., 1.)}
    }
