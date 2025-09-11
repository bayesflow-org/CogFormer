import numpy as np
from numba import njit


@njit
def _softplus(x: float) -> float:
    # stable softplus for scalar
    if x > 20.0:
        return x
    elif x < -20.0:
        return np.exp(x)
    else:
        return np.log1p(np.exp(x))

@njit
def _softplus_vec(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.size):
        out[i] = _softplus(float(x[i]))
    return out