from bayesgpt.utils.simulator_utils import exponential, sigmoid, shifted_softplus

def rdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_diff": shifted_softplus,
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": sigmoid
    }

def rdm_link_fun2():
    return {
        "v": exponential,
        "v_diff": shifted_softplus,
        "a": exponential,
        "tau": exponential,
        "s_v": exponential,
        "s_tau": sigmoid
    }
