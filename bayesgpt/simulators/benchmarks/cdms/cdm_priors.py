from scipy.stats import norm


def cdm_priors():
    return {
        "v": norm(loc=1.0, scale=0.8),
        "v_theta": norm(loc=0.0, scale=0.5),
        "a": norm(loc=0.5, scale=0.5),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.0, scale=0.6),
        "s_tau": norm(loc=-1.5, scale=0.7),
    }