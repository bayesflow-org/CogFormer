import numpy as np
from scipy.stats import halfnorm, gamma, norm, beta, lognorm


def ddm_priors():
    return {
        "v": norm(loc=1., scale=0.8),
        "a": norm(loc=0.25, scale=0.5),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.5, scale=1.),
        "s_tau": norm(loc=-1.5, scale=0.7)
    }

def ddm_priors1():
    return {
        "v": norm(loc=1., scale=1.),
        "a": norm(loc=2., scale=0.75),
        "tau": norm(loc=-1., scale=0.75),
        "s_v": norm(loc=-1.5, scale=1.),
        "s_tau": norm(loc=-1., scale=1.5)
    }

def ddm_priors2():
    return {
        "v": norm(loc=1., scale=0.8),
        "a": norm(loc=0.25, scale=0.5),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1, scale=0.6),
        "s_tau": norm(loc=-1.5, scale=0.7)
    }
