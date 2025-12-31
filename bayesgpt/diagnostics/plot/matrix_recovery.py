import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def recovery_matrix(
    true_param_matrix: np.ndarray,
    pred_param_matrix: np.ndarray,
    params: list[str],
    regressors: list[str] | None = None,
    color: str = "#000787",
    figsize: tuple | None = None,
):
    """
    Recovery plot extended to a parameter matrix B.

    Expected shapes:
      - true_param_matrix, pred_param_matrix: (num_sims, num_regressors, num_intrinsic_params)
          num_sims  : number of simulations / datasets
          num_regressors         : number of coefficient rows (0=intercept, 1..=regressors)
          num_intrinsic_params         : number of base parameters (len(params))

    Layout:
      - Rows    : intercept + regressors (num_regressors)
      - Columns : parameters (num_intrinsic_params)

    Each panel plots:
      x = true_param_matrix[:, k, p]
      y = pred_param_matrix[:, k, p]
    """
    if true_param_matrix.shape != pred_param_matrix.shape:
        raise ValueError(f"true_param_matrix and pred_param_matrix must have same shape. Got {true_param_matrix.shape} vs {pred_param_matrix.shape}")

    if true_param_matrix.ndim != 3:
        raise ValueError(f"true_param_matrix and pred_param_matrix must be 3D arrays of shape (num_sims, num_regressors, num_intrinsic_params). Got ndim={true_param_matrix.ndim}")

    num_sims, num_regressors, num_intrinsic_params = true_param_matrix.shape

    if len(params) != num_intrinsic_params:
        raise ValueError(f"len(params) must equal num_intrinsic_params={num_intrinsic_params}. Got len(params)={len(params)}")

    # Row labels: intercept + regressor names
    if regressors is None:
        row_labels = ["intercept"] + [f"regressor_{k}" for k in range(1, num_regressors)]
    else:
        if len(regressors) != num_regressors - 1:
            raise ValueError(f"len(regressors) must be num_regressors-1={num_regressors-1}. Got {len(regressors)}")
        row_labels = ["intercept"] + list(regressors)

    if figsize is None:
        # readable default similar to your current sizing heuristic
        figsize = (3 * num_intrinsic_params, 3 * num_regressors)

    f, axarr = plt.subplots(
        num_regressors, num_intrinsic_params,
        figsize=figsize,
        sharex=False,
        sharey=False,
        squeeze=False
    )

    for k in range(num_regressors):
        for p in range(num_intrinsic_params):
            ax = axarr[k, p]
            x = true_param_matrix[:, k, p]
            y = pred_param_matrix[:, k, p]

            sns.scatterplot(x=x, y=y, ax=ax, color=color, s=18, linewidth=0)

            make_quadratic(ax, x, y)

            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.3)
            sns.despine(ax=ax)

            # Column headers = base parameters
            if k == 0:
                ax.set_title(params[p])

            # Row labels on the left-most column
            if p == 0:
                ax.set_ylabel(f"{row_labels[k]}\nEstimation")
            else:
                ax.set_ylabel("")

            # Only bottom row gets x-labels to reduce clutter
            if k == num_regressors - 1:
                ax.set_xlabel("Ground Truth")
            else:
                ax.set_xlabel("")

    f.tight_layout()
    return f


def make_quadratic(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray):
    """
    Utility to make subplots quadratic to avoid visual illusions
    in, e.g., recovery plot.
    """
    lower = min(x_data.min(), y_data.min())
    upper = max(x_data.max(), y_data.max())
    eps = (upper - lower) * 0.1 if upper > lower else 1.0

    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))
    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color="black",
        alpha=0.9,
        linestyle="dashed",
    )
