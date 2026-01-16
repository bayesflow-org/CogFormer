import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot_utils import hex_code, make_quadratic


def matrix_recovery(
    true: np.ndarray,
    pred: np.ndarray,
    free_params: list[str],
    fixed_params: list[str],
    max_num_categories: int = 2,
    intercept_color: str = "#000787",
    slope_color: str = "#FF6969",
    figsize: tuple | None = None,
    group_by_category: bool = False,
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
            ax.set_title(params[c] if r == 0 else "")

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
