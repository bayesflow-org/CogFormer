from scipy.stats import norm

def ddm_priors():
    return {
        "v": norm(loc=1.0, scale=0.8),
        "a": norm(loc=0.5, scale=0.5),
        "z": norm(loc=0.0, scale=0.8),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.2, scale=1.0),
        "s_tau": norm(loc=-1.5, scale=0.7)
    }
