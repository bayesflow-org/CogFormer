import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from cogformer.simulators.context_manager import ContextManager
from cogformer.utils.plot_utils import cogformer_cm_colors


def compute_empirical_coverage(
    estimates: np.ndarray,
    targets: np.ndarray,
    widths: np.ndarray,
    prob: float = 0.95,
    interval_type: str = "central",
) -> dict:
    """
    Compute empirical coverage statistics for given interval widths.

    Parameters
    ----------
    estimates : np.ndarray of shape (num_datasets, num_post_draws, num_params)
        The posterior draws obtained from num_datasets
    targets : np.ndarray of shape (num_datasets, num_params)
        The true parameter values used for generating num_datasets
    widths : np.ndarray
        Array of interval widths to compute coverage for (values between 0 and 1)
    prob : float, optional, default: 0.95
        Confidence level for coverage confidence intervals
    interval_type : str, optional, default: "central"
        Type of credible interval. Either "central" or "leftmost"

    Returns
    -------
    dict
        Dictionary containing coverage statistics for each width and parameter
    """
    num_datasets, num_draws, num_params = estimates.shape
    num_widths = len(widths)

    # Sort once — rank order is independent of width
    sorted_samples = np.sort(estimates, axis=1)  # (num_datasets, num_draws, num_params)

    # Initialize output arrays
    coverage_estimates = np.zeros((num_widths, num_params))
    coverage_lower = np.zeros((num_widths, num_params))
    coverage_upper = np.zeros((num_widths, num_params))
    width_represented = np.zeros((num_widths, num_params))

    for w_idx, width in enumerate(widths):
        # Number of ranks to cover for this width
        n_ranks_covered = round((num_draws + 1) * width)

        if interval_type == "central":
            # Central interval: center around median
            low_rank = round(num_draws / 2 - n_ranks_covered / 2)
            high_rank = low_rank + n_ranks_covered - 1
        elif interval_type == "leftmost":
            # Leftmost interval: start from minimum
            low_rank = 0
            high_rank = n_ranks_covered - 1
        else:
            raise ValueError("interval_type must be 'central' or 'leftmost'")

        # Ensure ranks are within valid bounds
        low_rank = max(0, low_rank)
        high_rank = min(num_draws - 1, high_rank)

        # Actual width represented by these ranks
        actual_width = (high_rank - low_rank + 1) / (num_draws + 1)

        # Vectorized over all params: (num_datasets, num_params)
        is_covered = (
            (targets >= sorted_samples[:, low_rank, :]) &
            (targets <= sorted_samples[:, high_rank, :])
        )
        num_covered = np.sum(is_covered, axis=0)  # (num_params,)
        coverage_est = num_covered / num_datasets

        alpha_post = num_covered + 1
        beta_post = num_datasets - num_covered + 1

        if actual_width == 0 or actual_width == 1:
            # No variability possible at boundary
            ci_low = np.full(num_params, actual_width)
            ci_high = np.full(num_params, actual_width)
        else:
            ci_low = beta.ppf((1 - prob) / 2, alpha_post, beta_post)
            ci_high = beta.ppf((1 + prob) / 2, alpha_post, beta_post)

        coverage_estimates[w_idx] = coverage_est
        coverage_lower[w_idx] = ci_low
        coverage_upper[w_idx] = ci_high
        width_represented[w_idx] = actual_width

    return {
        "coverage_estimates": coverage_estimates,
        "coverage_lower": coverage_lower,
        "coverage_upper": coverage_upper,
        "width_represented": width_represented,
        "widths": widths,
    }


def adaptive_coverage(
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
    interval_type: str = "central",
    difference: bool = False,
) -> plt.Figure:
    """
    Adaptive (masked) empirical coverage plot over a regressor-by-parameter grid.
    """

    if parameter_mask is None:
        context_manager = ContextManager()
        parameter_mask = context_manager.build_parameter_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True
        )
    elif parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    batch_size, num_draws, num_rows, num_cols = pred.shape

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

    widths = np.arange(0, num_draws + 2) / (num_draws + 1)

    regressor_keys = list(design_config.keys())

    legend_handles = None

    for r in range(num_rows):
        # Match adaptive_recovery styling: intercept row vs regressor rows
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
                    # Invisible placeholder ticks so xlabel aligns with active cells
                    ax.set_xticks([0.0, 0.5, 1.0])
                    ax.set_xticklabels(['0.0', '0.5', '1.0'])
                    ax.tick_params(axis='x', length=0, labelcolor='none', labelsize=tick_fontsize)
                else:
                    ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=18,
                    weight="bold",
                    alpha=0.7,
                    color=tint,
                )
                ax.set_xlabel("Interval width" if r == num_rows - 1 else "", fontsize=label_fontsize)
                continue

            # Build per-cell arrays in the shape expected by compute_empirical_coverage
            estimates_cell = pred[:,:,r,c][..., None] # (batch, draws, 1)
            targets_cell = true[:, r, c][:, None]  # (batch, 1)

            cov_data = compute_empirical_coverage(
                estimates=estimates_cell,
                targets=targets_cell,
                widths=widths,
                prob=prob,
                interval_type=interval_type,
            )

            # For this cell, there's only one "param" dimension => index 0
            width_rep = cov_data["width_represented"][:, 0]
            cov_est = cov_data["coverage_estimates"][:, 0]
            cov_low = cov_data["coverage_lower"][:, 0]
            cov_high = cov_data["coverage_upper"][:, 0]

            if difference:
                diff_est = cov_est - width_rep
                diff_low = cov_low - width_rep
                diff_high = cov_high - width_rep

                ax.fill_between(
                    width_rep,
                    diff_low,
                    diff_high,
                    color="grey",
                    alpha=0.33,
                    label=f"{int(prob*100)}% Credible Interval",
                )
                ax.axhline(0.0, color="black", linestyle="dashed", label="Ideal Coverage")
                ax.plot(width_rep, diff_est, color=tint, alpha=1.0, label="Coverage Difference")
                ax.set_ylim(-0.55, 0.55)
            else:
                ax.fill_between(
                    width_rep,
                    cov_low,
                    cov_high,
                    color="grey",
                    alpha=0.33,
                    label=f"{int(prob*100)}% Credible Interval",
                )
                ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", label="Ideal Coverage")
                ax.plot(width_rep, cov_est, color=tint, alpha=1.0, label="Empirical Coverage")
                ax.set_ylim(-0.02, 1.02)

            # Labels/titles
            ax.set_ylabel(ylabel if c == 0 else "", fontsize=label_fontsize)
            ax.set_xlabel("Interval width" if r == num_rows - 1 else "", fontsize=label_fontsize)

            if variable_names is not None and r == 0:
                ax.set_title(variable_names[c], fontsize=title_fontsize)

            ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks if not difference else [-0.5, -0.25, 0.0, 0.25, 0.5])
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

    debug = True
    color = cogformer_cm_colors()

    if debug:
        design_config = {
            "1": ["v", "a", "z", "tau"],
            "u_1": ["v", "a", "tau"],
            "u_2": ["v", "a", "z"],
            "u_1:u_2": ["v", "a"],
        }

        intrinsic_params = design_config["1"]
        variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]

        num_params = len(intrinsic_params)
        num_regressors = len(design_config.keys()) - 1
        num_categories = 3

        # This must match ContextManager's row layout:
        # rows = 1 (intercept) + num_regressors * (num_categories - 1)
        num_rows = 1 + num_regressors * (num_categories - 1)
        num_cols = num_params

        batch_size = 200
        num_draws = 256  # posterior samples per dataset

        # Build the same parameter mask used by adaptive_recovery
        cm = ContextManager()
        parameter_mask = cm.build_parameter_mask(
            design_config=design_config,
            max_num_categories=num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True,
        )
        if parameter_mask.ndim == 3:
            parameter_mask = parameter_mask[0]

        # True values (batch, rows, cols)
        true = np.random.normal(0.0, 1.0, size=(batch_size, num_rows, num_cols))

        # Posterior draws (batch, draws, rows, cols)
        # Make draws roughly calibrated around truth (so the plot looks sensible)
        sigma = 0.5
        pred = true[:, None, :, :] + np.random.normal(0.0, sigma, size=(batch_size, num_draws, num_rows, num_cols))

        print("true:", true.shape)
        print("pred:", pred.shape)
        print("mask:", parameter_mask.shape)

        fig = adaptive_coverage(
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
            interval_type="central",
            difference=False,
            title_fontsize=18,
            label_fontsize=14,
        )

        fig.savefig("test_adaptive_coverage.pdf")
        print("Saved test_adaptive_coverage.pdf")