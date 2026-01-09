import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def correlation(
    true: np.ndarray,
    pred: np.ndarray,
    free_params: list[str],
    fixed_params: list[str],
    figsize: tuple | None = None,
):
    params = free_params + fixed_params
    num_params = len(params)

    n_rows = true.shape[1]

    correlations = np.zeros((n_rows, num_params))

    if figsize is None:
        figsize = (num_params, n_rows)

    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)

    for r in range(n_rows):
        for c in range(num_params):
            x = true[:, r, c]
            y = pred[:, r, c]
            if r == 0 or (r > 0 and params[c] in free_params):
                correlations[r, c] = np.corrcoef(x, y)[0, 1]

    ax = sns.heatmap(correlations, xticklabels=False, yticklabels=False, cbar=False)

    return fig