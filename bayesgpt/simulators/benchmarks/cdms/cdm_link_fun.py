from bayesgpt.utils.simulator_utils import exponential, sigmoid, shifted_softplus


def cdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_theta": sigmoid,
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus
    }

def cdm_link_fun2():
    return {
        "v": exponential,
        "v_theta": sigmoid,
        "a": exponential,
        "tau": exponential,
        "s_v": exponential,
        "s_tau": exponential
    }