import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections.abc import Callable

from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.utils.plot_utils import make_quadratic


def adaptive_recovery(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    variable_names: list[str] = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    uncertainty_agg: Callable = None,
    figsize: tuple = None
):
    if variable_names is None:
        variable_names = design_config["1"]

    context_manager = ContextManager()
    parameter_mask = context_manager.build_parameter_mask(
        design_config=design_config,
        max_num_categories=num_categories,
        intrinsic_params=intrinsic_params,
        keep_intercept=True
    )
    print(parameter_mask)
    num_rows, num_cols = parameter_mask.shape

    if figsize is None:
        figsize = (3 * num_cols, 2.8 * num_rows)

    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    regressor_keys = list(design_config.keys())
    for r in range(num_rows):
        # Initialize regressor key and color

        if r > 0:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            regressor_key = regressor_keys[regressor_id]
            ylabel = fr"${regressor_key} | c_{category_id}$"
            color = interaction_color if ":" in regressor_key else main_effect_color
        else:
            ylabel = r"$1$"
            color = intercept_color

        for c in range(num_cols):
            ax = axarr[r, c]
            x = true[..., r, c]
            y = pred[..., r, c]
            mask = parameter_mask[r, c]
            if mask == 1.0:
                sns.scatterplot(x=x, y=y, ax=ax, color=color)
                make_quadratic(ax, x, y)

                corr = np.corrcoef(x, y)[0, 1]
                metric_label = f"r = {corr:.3f}"
                ax.text(0.1, 0.95, metric_label, ha="left", va="center", transform=ax.transAxes, size=12)

                ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.2)
                sns.despine(ax=ax)
                ax.set_ylabel(ylabel if c == 0 else "")
                ax.set_xlabel("Ground Truth" if r == num_rows - 1 else "")
                ax.set_title(variable_names[c] if r == 0 else "")

            else:
                ax.set_facecolor(color)
                ax.patch.set_alpha(0.05)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

    fig.tight_layout()
    return fig

def credible_interval(x: np.ndarray, prob: float = 0.95, axis: int = None, **kwargs) -> np.ndarray:
    """
    Compute credible interval from samples using quantiles.

    Parameters
    ----------
    x : array_like
        Input array of samples from a posterior distribution or bootstrap samples.
    prob : float, default 0.95
        Coverage probability of the credible interval (between 0 and 1).
        For example, 0.95 gives a 95% credible interval.
    axis : Sequence[int]
        Axis or axes along which the credible interval is computed.
        Default is None (flatten array).

    Returns
    -------
    a numpy array of shape (2, ...) with the first dimension indicating the
    lower and upper bounds of the credible interval.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate posterior samples
    >>> samples = np.random.normal(size=(10, 1000, 3))

    >>> # Different coverage probabilities
    >>> credible_interval(samples, prob=0.5, axis=1)  # 50% CI
    >>> credible_interval(samples, prob=0.99, axis=1)  # 99% CI
    """

    # Input validation
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be between 0 and 1, got {prob}")

    # Calculate tail probabilities
    alpha = 1 - prob
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Compute quantiles
    return np.quantile(x, q=(lower_q, upper_q), axis=axis, **kwargs)

if __name__ == "__main__":
    design_config = {
        "1": ["v", "a", "z", "tau"],
        "u_1": ["v", "a", "tau"],
        "u_2": ["v", "a", "z"],
        "u_1:u_2": ["v", "a"]
    }

    intrinsic_params = design_config["1"]
    num_params = len(intrinsic_params)
    num_regressors = len(list(design_config.keys())) - 1
    num_categories = 3
    batch_size = 10
    true = np.random.normal(1, 1, (batch_size, num_regressors * (num_categories - 1) + 1, num_params))
    pred = np.random.normal(1, 1, (batch_size, num_regressors * (num_categories - 1) + 1, num_params))
    print(true.shape, pred.shape)

    fig = adaptive_recovery(true, pred, design_config, intrinsic_params, max_num_categories=num_categories)
    fig.savefig("adaptive_recovery.pdf")
    print("success")
