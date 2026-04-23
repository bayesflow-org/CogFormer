import numpy as np
from scipy.stats import halfnorm


def lba_baseline_priors():
    return {
        "v":        np.random.gamma(2., 1.),
        "a":        np.random.normal(-.1, 0.3),
        "tau":      np.random.normal(-1.5, 0.3),
        "s_v":      halfnorm(loc=0.0, scale=1.0),
        "s_tau":    np.random.beta(1.0, 3.0),
    }

def lba_priors():
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

def lba_log_priors():
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

def lba_full_priors():
    return {
        "v":        {"intercept": lambda: np.random.gamma(2., 1.),
                     "slope": lambda: np.random.normal(0., 1.)},
        "a":        {"intercept": lambda: np.random.normal(-1, 0.3),
                     "slope": lambda: np.random.normal(0., 1.)},
        "tau":      {"intercept": lambda: np.random.normal(-1.5, 0.3),
                     "slope": lambda: np.random.normal(0., 1.)},
        "s_v":      {"intercept": lambda: halfnorm.rvs(loc=0.0, scale=1.0),
                     "slope": lambda: np.random.normal(0., 1.)},
        "s_tau":    {"intercept": lambda: np.random.beta(1.0, 3.0),
                     "slope": lambda: np.random.normal(0., 1.)}
    }
