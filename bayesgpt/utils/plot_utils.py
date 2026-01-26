import random
import numpy as np
import matplotlib.pyplot as plt


def make_quadratic(ax: plt.Axes, x: np.ndarray, y: np.ndarray):
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
        color="black",
        alpha=0.9,
        linestyle="dashed",
    )

def hex_code():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def staedtler_fineliner():
    colors = {
        # BF
        "bf_intercept": "#1B9E77",
        "bf_main_effect": "#1AC995",
        "bf_interaction": "#10E0A2",

        # BayesGPT-VI
        "vi_intercept": "#15435F",
        "vi_main_effect": "#007396",
        "vi_interaction": "#3EB1C8",

        # BayesGPT-FM
        "fm_intercept": "#6969FF",
        "fm_main_effect": "#7570B3",
        "fm_interaction": "#9E9AC8",

        # BayesGPT-CM
        "cm_intercept": "#AD1457",
        "cm_main_effect": "#EC008C",
        "cm_interaction": "#FF69FF",
    }
    return colors
