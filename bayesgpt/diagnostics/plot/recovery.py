import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def recovery(
    true: np.ndarray,
    pred: np.ndarray,
    params: list[str],
    color: str = "#000787",
    figsize: tuple = None
):

    num_params = len(params)

    if figsize is None:
        figsize = (3 * num_params, 3)

    f, axarr = plt.subplots(1, num_params, figsize=figsize)

    for i, ax in enumerate(axarr.flatten()):
        sns.scatterplot(x=true[:, i], y=pred[:, i], ax=ax, color=color)

        make_quadratic(ax, true[:, i], pred[:, i])

        ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Ground Truth")
        if i % num_params == 0:
            ax.set_ylabel("Estimation")
        ax.set_title(params[i])
        sns.despine(ax=ax)

    f.tight_layout()
    return f

def make_quadratic(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray):
    """
    Utility to make subplots quadratic to avoid visual illusions
    in, e.g., recovery plot.
    """
    lower = min(x_data.min(), y_data.min())
    upper = max(x_data.max(), y_data.max())
    eps = (upper - lower) * 0.1
    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))
    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color="black",
        alpha=0.9,
        linestyle="dashed",
    )
