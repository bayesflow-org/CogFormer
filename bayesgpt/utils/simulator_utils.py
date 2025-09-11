import numpy as np
from numba import njit
from typing import Dict, Tuple


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

def generate_regressors(
    params: Dict[str, np.ndarray],
    num_samples: int,
    param_dims: Dict[str, int],
    fixed_parameters: set[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    regressors, regressed = {}, {}
    for name, vec in params.items():
        if name in fixed_parameters:
            continue
        arr = np.asarray(vec, dtype=np.float32)
        dim = param_dims.get(name, 1)

        # Scalar → broadcast
        if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
            regressed[name] = np.full(num_samples, float(arr[0] if arr.ndim else arr), dtype=np.float32)
            regressors[name] = np.ones((num_samples, 1), dtype=np.float32)
            continue

        # Coeff vector → linear regression expansion
        if arr.ndim == 1 and arr.size == dim:
            X = np.c_[np.ones((num_samples, 1), dtype=np.float32),
                      np.random.rand(num_samples, max(0, dim - 1)).astype(np.float32)]
            regressed[name] = (X @ arr.reshape(-1, 1)).ravel().astype(np.float32)
            regressors[name] = X
            continue

        raise ValueError(f"{name}: unsupported shape {arr.shape} for dim={dim}")
    return regressors, regressed
