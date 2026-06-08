from cogformer.utils.simulator_utils import shifted_softplus

def lba_link_fun():
    return {
        "v":     shifted_softplus,
        "v_diff": shifted_softplus,
        "a":     shifted_softplus,
        "tau":   shifted_softplus,
        "s_v":   shifted_softplus,
        "s_tau": shifted_softplus,
    }