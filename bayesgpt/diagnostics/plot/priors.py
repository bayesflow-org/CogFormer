import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def priors(
    samples: dict[str, np.ndarray],
    color: str = "#000787",
    figsize: tuple = None,
):
    params = list(samples.keys())
    num_params = len(params)

    if figsize is None:
        figsize = (3 * num_params, 3)

    f, axarr = plt.subplots(1, num_params, figsize=figsize, sharey=True)

    for i, ax in enumerate(axarr):
        sns.histplot(samples[params[i]], ax=ax, kde=True, color=color)
        ax.set_title(params[i])

    f.tight_layout()
    return f