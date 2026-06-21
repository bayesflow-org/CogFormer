import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from cogformer.simulators.context_manager import ContextManager
from cogformer.utils.plot_utils import cogformer_cm_colors


def compute_ecdf_bands(
    num_datasets: int,
    prob: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact pointwise confidence bands for the ECDF of a uniform distribution.

    Uses the fact that the k-th order statistic of n i.i.d. Uniform(0,1) samples
    follows a Beta(k, n−k+1) distribution, giving exact closed-form bands.

    Parameters
    ----------
    num_datasets : int
        Number of datasets (determines ECDF resolution).
    prob : float, optional, default: 0.95
        Confidence level for the bands.

    Returns
    -------
    x : np.ndarray of shape (num_datasets,)
        Sorted x-axis positions (k / num_datasets).
    lower : np.ndarray of shape (num_datasets,)
        Lower confidence band.
    upper : np.ndarray of shape (num_datasets,)
        Upper confidence band.
    """
    k = np.arange(1, num_datasets + 1)
    x = k / num_datasets
    lower = beta_dist.ppf((1 - prob) / 2, k, num_datasets - k + 1)
    upper = beta_dist.ppf((1 + prob) / 2, k, num_datasets - k + 1)
    return x, lower, upper


def adaptive_ecdf(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    variable_names: list,
    intrinsic_params: list = None,
    parameter_mask: np.ndarray = None,
    max_num_categories: int = 2,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    figsize: tuple | None = None,
    title_fontsize: int = 20,
    label_fontsize: int = 14,
    tick_fontsize: int = 11,
    legend_fontsize: int = 12,
    legend_location: str = "lower right",
    prob: float = 0.95,
    difference: bool = True,
) -> plt.Figure:
    """
    Adaptive (masked) ECDF calibration plot over a regressor-by-parameter grid.

    For each active cell, computes the fractional rank of the true parameter value
    within the posterior draws and plots the resulting empirical CDF against the
    expected uniform CDF, with simultaneous confidence bands.

    Parameters
    ----------
    true : np.ndarray of shape (num_datasets, num_rows, num_cols)
        True parameter values.
    pred : np.ndarray of shape (num_datasets, num_draws, num_rows, num_cols)
        Posterior draws.
    design_config : dict
        Design configuration mapping regressor names to lists of parameter names.
    variable_names : list of str
        Column (parameter) names for subplot titles.
    intrinsic_params : list of str, optional
        Intrinsic parameter names (passed to ContextManager if mask not provided).
    parameter_mask : np.ndarray, optional
        Boolean mask of shape (num_rows, num_cols). Built from ContextManager if None.
    max_num_categories : int, optional, default: 2
        Maximum number of categories (used for row layout).
    intercept_color : str
        Color for intercept row.
    main_effect_color : str
        Color for main effect rows.
    interaction_color : str
        Color for interaction rows.
    figsize : tuple or None
        Figure size. Defaults to (3.2 * num_cols, 2.9 * num_rows).
    title_fontsize : int
        Font size for column titles.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.
    legend_fontsize : int
        Font size for legend.
    legend_location : str
        Legend location (unused; legend is placed at figure bottom).
    prob : float, optional, default: 0.95
        Confidence level for simultaneous ECDF bands.
    num_simulations : int, optional, default: 1000
        Number of simulations for confidence band estimation.
    difference : bool, optional, default: False
        If True, plot ECDF − uniform diagonal instead of raw ECDF.

    Returns
    -------
    fig : plt.Figure
    """
    if parameter_mask is None:
        context_manager = ContextManager()
        parameter_mask = context_manager.build_parameter_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True,
        )
    elif parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    num_datasets, num_draws, num_rows, num_cols = pred.shape

    regressor_keys = list(design_config.keys())
    n_design_rows = 1 + (len(regressor_keys) - 1) * (max_num_categories - 1)
    num_rows = min(num_rows, n_design_rows)

    # Trim trailing rows that are fully masked (e.g. intercept-only designs).
    for r in range(num_rows - 1, -1, -1):
        if parameter_mask[r, :].any():
            num_rows = r + 1
            break

    if figsize is None:
        figsize = (3.2 * num_cols, 2.9 * num_rows)

    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    # Precompute confidence bands (same for all cells since num_datasets is fixed)
    band_x, band_lower, band_upper = compute_ecdf_bands(
        num_datasets=num_datasets,
        prob=prob,
    )

    regressor_keys = list(design_config.keys())
    legend_handles = None

    for r in range(num_rows):
        if r > 0:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            regressor_key = regressor_keys[regressor_id]
            ylabel = fr"${regressor_key}$" + (fr"$ | c_{category_id}$" if max_num_categories > 2 else "")
            tint = interaction_color if ":" in regressor_key else main_effect_color
        else:
            ylabel = r"$1$"
            tint = intercept_color

        for c in range(num_cols):
            ax = axarr[r, c]

            # Masked cell -> N/A tile
            if parameter_mask[r, c] != 1.0:
                ax.set_facecolor(tint)
                ax.patch.set_alpha(0.05)
                if r == num_rows - 1:
                    ax.set_xticks([0.0, 0.5, 1.0])
                    ax.set_xticklabels(["0.0", "0.5", "1.0"])
                    ax.tick_params(axis="x", length=0, labelcolor="none", labelsize=tick_fontsize)
                else:
                    ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(
                    0.5, 0.5, "N/A",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=18, weight="bold",
                    alpha=0.7, color=tint,
                )
                ax.set_xlabel("Fractional rank statistic" if r == num_rows - 1 else "", fontsize=label_fontsize)
                continue

            # Compute fractional ranks: for each dataset, fraction of draws < true value
            # pred[:, :, r, c]: (num_datasets, num_draws)
            # true[:, r, c]:    (num_datasets,)
            ranks = np.mean(pred[:, :, r, c] < true[:, r, c, np.newaxis], axis=1)  # (num_datasets,)
            ranks_sorted = np.sort(ranks)
            ecdf_y = np.arange(1, num_datasets + 1) / num_datasets

            if difference:
                ax.fill_between(
                    band_x,
                    band_lower - band_x,
                    band_upper - band_x,
                    color="grey", alpha=0.07,
                    label=f"{int(prob * 100)}% Confidence Band",
                )
                ax.axhline(0.0, color="black", linestyle="dashed", label="Ideal (Uniform)")
                ax.plot(ranks_sorted, ecdf_y - ranks_sorted, color=tint, alpha=1.0, label="ECDF Difference")
            else:
                ax.fill_between(
                    band_x,
                    band_lower,
                    band_upper,
                    color="grey", alpha=0.07,
                    label=f"{int(prob * 100)}% Confidence Band",
                )
                ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", label="Ideal (Uniform)")
                ax.plot(ranks_sorted, ecdf_y, color=tint, alpha=1.0, label="Empirical CDF")

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylabel(ylabel if c == 0 else "", fontsize=label_fontsize)
            ax.set_xlabel("Fractional rank statistic" if r == num_rows - 1 else "", fontsize=label_fontsize)

            if variable_names is not None and r == 0:
                ax.set_title(variable_names[c], fontsize=title_fontsize)

            ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
            ax.set_xticks(ticks)
            if not difference:
                ax.set_yticks(ticks)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.35)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.tight_layout()
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
        )
        fig.subplots_adjust(bottom=0.1)
    return fig


if __name__ == "__main__":

    color = cogformer_cm_colors()

    design_config = {
        "1": ["v", "a", "z", "tau"],
        "u_1": ["v", "a", "tau"],
        "u_2": ["v", "a", "z"],
        "u_1:u_2": ["v", "a"],
    }

    intrinsic_params = design_config["1"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]

    num_categories = 3
    batch_size = 200
    num_draws = 256

    cm = ContextManager()
    parameter_mask = cm.build_parameter_mask(
        design_config=design_config,
        max_num_categories=num_categories,
        intrinsic_params=intrinsic_params,
        keep_intercept=True,
    )
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    num_rows, num_cols = parameter_mask.shape

    # Good calibration: true and draws from same marginal → exchangeable → ECDF ≈ diagonal
    true = np.random.normal(0.0, 1.0, size=(batch_size, num_rows, num_cols))
    pred = np.random.normal(0.0, 1.0, size=(batch_size, num_draws, num_rows, num_cols))


    fig = adaptive_ecdf(
        true=true,
        pred=pred,
        design_config=design_config,
        parameter_mask=parameter_mask,
        variable_names=variable_names,
        max_num_categories=num_categories,
        intercept_color=color["intercept"],
        main_effect_color=color["main_effect"],
        interaction_color=color["interaction"],
        prob=0.95,
        difference=True,
    )

    fig.savefig("test_adaptive_ecdf.pdf")
