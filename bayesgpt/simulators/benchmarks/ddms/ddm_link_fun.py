import numpy as np

from bayesgpt.utils.simulator_utils import sigmoid, softplus, shifted_softplus


def ddm_link_fun():
    return {
        "v": shifted_softplus,
        "a": shifted_softplus,
        "tau": sigmoid,
        "s_v": shifted_softplus,
        "s_tau": sigmoid
    }