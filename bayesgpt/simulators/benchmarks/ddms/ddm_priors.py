from scipy.stats import norm


# def ddm_priors():  # 5 params, no z
#     return {
#         "v": norm(loc=1.0, scale=0.8),
#         "a": norm(loc=0.5, scale=0.5),
#         "tau": norm(loc=-1.2, scale=0.5),
#         "s_v": norm(loc=-1.0, scale=0.6),
#         "s_tau": norm(loc=-1.5, scale=0.7)
#     }


def ddm_priors():
    return {
        "v": norm(loc=1.0, scale=0.8),
        "a": norm(loc=0.5, scale=0.5),
        "z": norm(loc=0.0, scale=0.8),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.5, scale=0.4),
        "s_tau": norm(loc=-1.5, scale=0.7)
    }


# def ddm_priors():
#     return {
#         "v": norm(loc=1., scale=0.8),
#         "a": norm(loc=0.25, scale=0.5),
#         "tau": norm(loc=-1.2, scale=0.5),
#         "s_v": norm(loc=-1.5, scale=1.),
#         "s_tau": norm(loc=-1.5, scale=0.7)
#     }

# def ddm_priors1():
#     return {
#         "v": norm(loc=1., scale=1.),
#         "a": norm(loc=2., scale=0.75),
#         "tau": norm(loc=-1., scale=0.75),
#         "s_v": norm(loc=-1.5, scale=1.),
#         "s_tau": norm(loc=-1., scale=1.5)
#     }