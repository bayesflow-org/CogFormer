import numpy as np

from cogformer.utils.simulator_utils import sigmoid, shifted_softplus, exponential, scaled_sigmoid

def ddm_link_fun():
    return {
        "v": shifted_softplus,
        "a": shifted_softplus,
        "z": scaled_sigmoid,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus
    }
