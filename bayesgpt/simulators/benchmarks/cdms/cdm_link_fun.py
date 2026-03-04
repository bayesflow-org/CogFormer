from bayesgpt.utils.simulator_utils import exponential, scaled_sigmoid, shifted_softplus


def cdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_theta": scaled_sigmoid,
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus
    }

def cdm_link_fun2():
    return {
        "v": exponential,
        "v_theta": scaled_sigmoid,
        "a": exponential,
        "tau": exponential,
        "s_v": exponential,
        "s_tau": exponential
    }