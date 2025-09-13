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
