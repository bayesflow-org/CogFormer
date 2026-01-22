import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Callable

from utils.plot_utils import hex_code, make_quadratic


def matrix_recovery(
    true: np.ndarray,
    pred: np.ndarray,
    free_params: list[str],
    fixed_params: list[str],
    params_mask: np.ndarray = None,
    max_num_categories: int = 2,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#ff7f0e",
    interaction_color: str = "#ff7f0e",
    uncertainty_agg: Callable = None,
    slope_color: str = "#6969ff",
    figsize: tuple | None = None,
    group_by_category: bool = False,
    param_names: list[str] | None = None,
):

    params = free_params + fixed_params
    num_params = len(params)

    if group_by_category:
        n_rows = (true.shape[1] - 1) // (max_num_categories - 1) + 1
    else:
        n_rows = true.shape[1]

    if figsize is None:
        figsize = (3.0 * num_params, 3.0 * n_rows)

    fig, axarr = plt.subplots(n_rows, num_params, figsize=figsize, squeeze=False)
    for r in range(n_rows):
        row_color = intercept_color if r == 0 else slope_color
        regressor_index = (r - 1) // (max_num_categories - 1) + 1
        category_index = (r - 1) % (max_num_categories - 1) + 1
        row_ylabel = "1" if r == 0 else fr"$u_{regressor_index}$ | $c_{category_index}$"

        for c in range(num_params):
            ax = axarr[r, c]
            x = true[:, r, c]
            y = pred[:, r, c]
            mask = params_mask[r, c]

            if mask == 1.0:

                sns.scatterplot(x=x, y=y, ax=ax, color=row_color)
                make_quadratic(ax, x, y)

                if r == 0 or (r > 0 and params[c] in free_params):
                    corr = np.corrcoef(x, y)[0, 1]
                    metric_label = f"r = {corr:.3f}"
                    ax.text(0.1, 0.95, metric_label, ha="left", va="center", transform=ax.transAxes, size=12)

                ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.3)
                sns.despine(ax=ax)

                ax.set_ylabel(row_ylabel if c == 0 else "")
                ax.set_xlabel("Ground Truth" if r == n_rows - 1 else "")
                ax.set_title(param_names[c] if r == 0 else "")
            else:
                # make it look like a solid block
                ax.set_facecolor((0.0, 0.0, 0.0, 0.05))
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

    fig.tight_layout()
    return fig

def make_quadratic(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray):
    """Make subplot quadratic + add y=x guide."""
    lower = min(x_data.min(), y_data.min())
    upper = max(x_data.max(), y_data.max())

    # Safeguard plotting issue with fixed and/or masked parameters
    span = upper - lower
    if span == 0:
        span = 1.0
    eps = span * 0.1

    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))
    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color="black",
        alpha=0.9,
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
