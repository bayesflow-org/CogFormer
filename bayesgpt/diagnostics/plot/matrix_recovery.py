import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot_utils import hex_code


def matrix_recovery(
    true: np.ndarray,
    pred: np.ndarray,
    free_params: list[str],
    fixed_params: list[str],
    max_num_categories: int = 3,
    intercept_color: str = "#000787",
    figsize: tuple | None = None,
):

    params = free_params + fixed_params
    num_params = len(params)

    n_rows = true.shape[1]

    if figsize is None:
        figsize = (3.0 * num_params, 2.6 * n_rows)

    fig, axarr = plt.subplots(n_rows, num_params, figsize=figsize, squeeze=False)
    slope_color = ""
    for r in range(n_rows):
        if r % max_num_categories == 1:
            slope_color = hex_code()
        row_color = intercept_color if r == 0 else slope_color
        row_ylabel = "intercept (1)" if r == 0 else f"u_{r}"

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

            ax.set_ylabel(row_ylabel)

            if c == 0:
                ax.set_ylabel(row_ylabel)
            else:
                ax.set_ylabel("")

            if r == n_rows - 1:
                ax.set_xlabel("Ground Truth")
            else:
                ax.set_xlabel("")

            if r == 0:
                ax.set_title(params[c])

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
