import numpy as np


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
    return sp - np.log(2.0)


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