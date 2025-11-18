import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sim_data(
    data: np.ndarray,
    label: str,
    color: str = '#000787',
    figsize: tuple = None
):

    batch_size = 6 if data.shape[0] > 6 else data.shape[0]

    if figsize is None:
        figsize = (3 * batch_size, 3)

    f, axarr = plt.subplots(1, batch_size, figsize=figsize, sharey=True)

    for i, ax in enumerate(axarr.flatten()):
        sns.histplot(data, ax=ax, color=color, kde=True, legend=False)
        ax.set_xlabel(label)
        sns.despine(ax=ax)

    f.tight_layout()
    return f
