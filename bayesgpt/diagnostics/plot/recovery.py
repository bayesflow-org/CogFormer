import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bayesgpt.utils.plot_utils import make_quadratic

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
