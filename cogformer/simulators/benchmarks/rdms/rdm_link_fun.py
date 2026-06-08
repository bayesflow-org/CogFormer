from cogformer.utils.simulator_utils import exponential, sigmoid, shifted_softplus

def rdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_diff": shifted_softplus,
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus,
    }
