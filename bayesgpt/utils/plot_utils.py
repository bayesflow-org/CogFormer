import numpy as np
import matplotlib.pyplot as plt


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
