import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from bayesgpt.utils.plot_utils import make_quadratic


def regressor_recovery(
    true: np.ndarray,
    pred: np.ndarray,
    params: list[str],
    param_matrices: np.ndarray,
    max_num_regressors: int = 5,
    max_num_categories: int = 3,
    keep_intercept: bool = True,
    color: str = "#000787",
    figsize: tuple = None
):
    num_obs = param_matrices.shape[0]
    num_params = len(params)

    # Reshape the data
    num_elements_per_param = max_num_regressors * max_num_categories + (1 if keep_intercept else 0)
    num_cols = max_num_regressors + (1 if keep_intercept else 0)
    param_matrices = param_matrices.reshape(num_obs, num_elements_per_param, num_params)

    if keep_intercept:
        intercept = param_matrices[:, 0, :]
        regressors = param_matrices[:, 1:, :]
    else:
        regressors = param_matrices

    regressors = regressors.reshape(num_obs, max_num_regressors, max_num_categories, num_params)


    if figsize is None:
        figsize = (3 * num_params, 3 * num_cols)

    f, axarr = plt.subplots(num_params, num_cols, figsize=figsize, sharex=True, sharey=True)

    for i, ax in enumerate(axarr.flatten()):
        row = np.floor(i / num_params)
        col = np.int(i % num_params)


    f.tight_layout()
    return f
