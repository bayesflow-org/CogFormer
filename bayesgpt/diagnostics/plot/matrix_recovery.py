import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def matrix_recovery(
    true: np.ndarray,
    pred: np.ndarray,
    params: list[str],
    intercept_color: str = "#000787",
    slope_color: str = "#A23BEC",
    figsize: tuple | None = None,
):
    num_params = len(params)

    n_rows = true.shape[1]

    if figsize is None:
        figsize = (3.0 * num_params, 2.6 * n_rows)

    fig, axarr = plt.subplots(n_rows, num_params, figsize=figsize, squeeze=False)

    for r in range(n_rows):
        row_color = intercept_color if r == 0 else slope_color
        row_ylabel = "intercept (1)" if r == 0 else f"u_{r}"

        for c in range(num_params):
            ax = axarr[r, c]
            x = true[:, r, c]
            y = pred[:, r, c]

            sns.scatterplot(x=x, y=y, ax=ax, color=row_color)

            make_quadratic(ax, x, y)

            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.3)
            sns.despine(ax=ax)

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
