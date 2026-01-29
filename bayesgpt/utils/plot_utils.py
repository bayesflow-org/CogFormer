import random
import numpy as np
import matplotlib.pyplot as plt


def make_quadratic(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "black"):
    """
    Utility to make subplots quadratic to avoid visual illusions
    in, e.g., recovery plot.
    """

    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())

    span = upper - lower
    if span == 0:
        span = 1.0
    eps = span * 0.1

    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))

    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color=color,
        alpha=0.7,
        linestyle="dashed",
    )

def credible_interval(x: np.ndarray, prob: float = 0.95, axis: int = None, **kwargs) -> np.ndarray:
    """
    Compute credible interval from samples using quantiles.

    Parameters
    ----------
    x : array_like
        Input array of samples from a posterior distribution or bootstrap samples.
    prob : float, default 0.95
        Coverage probability of the credible interval (between 0 and 1).
        For example, 0.95 gives a 95% credible interval.
    axis : Sequence[int]
        Axis or axes along which the credible interval is computed.
        Default is None (flatten array).

    Returns
    -------
    a numpy array of shape (2, ...) with the first dimension indicating the
    lower and upper bounds of the credible interval.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate posterior samples
    >>> samples = np.random.normal(size=(10, 1000, 3))

    >>> # Different coverage probabilities
    >>> credible_interval(samples, prob=0.5, axis=1)  # 50% CI
    >>> credible_interval(samples, prob=0.99, axis=1)  # 99% CI
    """

    # Input validation
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be between 0 and 1, got {prob}")

    # Calculate tail probabilities
    alpha = 1 - prob
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Compute quantiles
    return np.quantile(x, q=(lower_q, upper_q), axis=axis, **kwargs)

def hex_code():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def bf_colors():
    colors = {
        # BF
        "intercept": "#1B9E77",
        "main_effect": "#1AC995",
        "interaction": "#10E0A2",
    }
    return colors

def bayesgpt_cm_colors():
    colors = {
        # BayesGPT-CM
        "intercept": "#15435F",
        "main_effect": "#007396",
        "interaction": "#3EB1C8",
    }
    return colors

def bayesgpt_vi_colors():
    colors = {
        # BayesGPT-VI
        "intercept": "#4E2A84",
        "main_effect": "#6969FF",
        "interaction": "#47b5ff",
    }

    return colors

def bayesgpt_fm_colors():
    colors = {
        # BayesGPT-FM
        "intercept": "#59315F",
        "main_effect": "#EC008C",
        "interaction": "#FF6969",
    }

    return colors

def staedtler_fineliner():
    colors = {
        "bf": bf_colors(),
        "vi": bayesgpt_vi_colors(),
        "fm": bayesgpt_fm_colors(),
        "cm": bayesgpt_cm_colors()
    }
    return colors
