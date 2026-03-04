import numpy as np

def exponential(x: np.ndarray):
    return np.exp(x)

def sigmoid(x: np.ndarray):
    return 1. / (1. + np.exp(-x))

def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x))

def softplus(x: np.ndarray):
    sp = np.log1p(1. + np.exp(x))
    return sp

def shifted_softplus(x: np.ndarray):
    """
    Vectorized shifted softplus: f(x) = log(1 + exp(x)) - log(2)
    This ensures f(0) = 0 and is stable for large |x|.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).

    Returns
    -------
    np.ndarray
        Same shape as x.
    """
    # Stable softplus: max(x,0) + log1p(exp(-|x|))
    sp = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    return sp

def scaled_sigmoid(
    x: float | np.ndarray,
    lower_bound: float | np.ndarray,
    upper_bound: float | np.ndarray
) -> float:
    """
    Apply a sigmoid transformation and rescale to a bounded interval.
    """
    return lower_bound + (upper_bound - lower_bound) / (1.0 + np.exp(-x))



def as_1d(x, n: int, allow_scalar: bool = True) -> np.ndarray:
    """Return a 1D array of length n. Broadcast scalars if allowed; else validate."""
    a = np.asarray(x)
    if a.ndim == 0:
        if allow_scalar:
            return np.full((n,), a.item())
    a = a.reshape(-1)
    if a.size == 1 and allow_scalar:
        return np.full((n,), a.item())
    if a.size != n:
        raise ValueError(f"array has length {a.size}; expected {n}.")
    return a

def inspect(out: dict, verbose=False):
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        elif isinstance(v, list):
            print(k, len(v) if not verbose else (v[i] for i in v))
        elif isinstance(v, dict):
            print(v.keys())
        else:
            print(k, v)
