import numpy as np
from scipy.stats import halfnorm, gamma, norm, beta, lognorm


def ddm_priors():
    return {
        "v": norm(loc=1., scale=1.), #lognorm(s=0.5),
        "a": norm(loc=2., scale=0.75),
        "tau": norm(loc=-1., scale=0.75),
        "s_v": norm(loc=-1.5, scale=1.),
        "s_tau": norm(loc=-1., scale=1.5)
    }


def ddm_baseline_priors():
    return {
        "v":        lambda: np.random.gamma(2., 1.),
        "a":        lambda: np.random.normal(-.1, 0.3),
        "tau":      lambda: np.random.normal(-1.5, 0.3),
        # "a":        lambda: np.random.lognormal(0.0, 0.25), #np.random.normal(-.1, 0.3),
        # "tau":      lambda: np.random.beta(2.0, 2.0) * 0.5,#np.random.normal(-1.5, 0.3),
        "s_v":      lambda: halfnorm.rvs(loc=0.0, scale=1.0),
        "s_tau":    lambda: np.random.beta(1.0, 3.0),
        # "s_tau":    lambda: np.random.beta(2.0, 5.0), #np.random.beta(1.0, 3.0),
        # "decay":  np.random.gamma(1.0, 0.4),
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

def ddm_baseline_priors2():
    return {
        "v":        lambda: np.random.gamma(5., 0.5),
        "a":        lambda: np.random.gamma(8, 0.25), #np.random.normal(-.1, 0.3),
        "tau":      lambda: np.random.gamma(2.0, 0.25),#np.random.normal(-1.5, 0.3),
        "s_v":      lambda: halfnorm.rvs(loc=0.0, scale=1.0),
        "s_tau":    lambda: np.random.beta(1.0, 3.0), #np.random.beta(1.0, 3.0),
        # "decay":  np.random.gamma(1.0, 0.4),
    }

def ddm_test_priors():
    return {
        "v":        lambda: np.random.lognormal(3.0, 1.5),
        "a":        lambda: np.random.gamma(3, 0.25),
        "tau":      lambda: np.random.beta(5.0, 2.0),
        "s_v":      lambda: halfnorm.rvs(loc=0.0, scale=0.25),
        "s_tau":    lambda: np.random.beta(2.0, 5.0),
        # "decay":  np.random.gamma(1.0, 0.4),
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
