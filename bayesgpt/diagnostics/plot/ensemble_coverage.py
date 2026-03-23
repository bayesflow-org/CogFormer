import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.diagnostics.plot.adaptive_coverage import compute_empirical_coverage
from bayesgpt.utils.plot_utils import _add_mask_legend


def ensemble_coverage(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    design_configs: list[dict],
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_masks: list[np.ndarray] = None,
    variable_names: list[str] = None,
    colors: list[str] = None,
    n_colors: int = None,
    palette_lightness: float = 0.77,
    palette_start_hue: float = 0.05,
    labels: list[str] = None,
    figsize: tuple = None,
    title_fontsize: int = 24,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 17,
    thumb_scale: float = 1.0,
    prob: float = 0.95,
    interval_type: str = "central",
    difference: bool = True,
    ci_alpha: float = 0.07,
) -> plt.Figure:
    """
    Overlay empirical coverage plots for multiple design configs on a shared
    parameter matrix grid.

    Parameters
    ----------
    true_list : list of np.ndarray
        Ground-truth arrays (batch, num_rows_i, num_cols), one per config.
    pred_list : list of np.ndarray
        Posterior draw arrays (batch, draws, num_rows_i, num_cols), one per config.
    design_configs : list of dict
        Design configuration dicts (max 8).
    intrinsic_params : list of str
        Intrinsic parameter names (columns).
    max_num_categories : int
        Maximum number of categories used to size row blocks.
    parameter_masks : list of np.ndarray, optional
        Pre-built masks; constructed from design_configs if not provided.
    variable_names : list of str, optional
        Column header names.
    colors : list of str, optional
        One color per config. Defaults to seaborn husl palette of 8.
    labels : list of str, optional
        Legend labels, one per config.
    figsize : tuple, optional
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize : int
    prob : float
        Coverage confidence level.
    interval_type : str
        "central" or "leftmost".
    difference : bool
        If True (default), plots coverage − ideal instead of raw coverage.
    ci_alpha : float
        Alpha for the credible interval fill_between (barely visible).
    """
    n_configs = len(design_configs)
    if n_configs > 12:
        raise ValueError("ensemble_coverage supports at most 12 design configs.")
    if len(true_list) != n_configs or len(pred_list) != n_configs:
        raise ValueError("true_list, pred_list, and design_configs must have the same length.")

    if variable_names is None:
        variable_names = design_configs[0].get("1", intrinsic_params)

    if parameter_masks is None:
        context_manager = ContextManager()
        parameter_masks = [
            context_manager.build_parameter_mask(
                design_config=dc,
                max_num_categories=max_num_categories,
                intrinsic_params=intrinsic_params,
                keep_intercept=True
            )
            for dc in design_configs
        ]
    else:
        parameter_masks = [
            m[0] if m.ndim == 3 else m
            for m in parameter_masks
        ]

    if colors is None:
        colors = sns.husl_palette(n_configs if n_colors is None else n_colors, h=palette_start_hue, l=palette_lightness)

    if labels is None:
        labels = [f"Config {i + 1}" for i in range(n_configs)]

    num_rows = max(m.shape[0] for m in parameter_masks)
    num_cols = parameter_masks[0].shape[1]

    if figsize is None:
        figsize = (3 * num_cols, 3 * num_rows)

    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    # Row labels from first design config
    first_dc = design_configs[0]
    regressor_keys_0 = list(first_dc.keys())

    for r in range(num_rows):
        if r == 0:
            ylabel = r"$1$"
        else:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            if regressor_id < len(regressor_keys_0):
                regressor_key = regressor_keys_0[regressor_id]
                ylabel = fr"${regressor_key}$" + (
                    fr"$ | c_{category_id}$" if max_num_categories > 2 else ""
                )
            else:
                ylabel = f"Row {r}"

        for c in range(num_cols):
            ax = axarr[r, c]

            # Find which configs are active at this cell
            active_configs = [
                k for k in range(n_configs)
                if r < parameter_masks[k].shape[0] and parameter_masks[k][r, c] == 1.0
            ]

            if not active_configs:
                ax.set_facecolor("gray")
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
                    alpha=0.7, color="gray"
                )
                ax.set_xlabel("Interval width" if r == num_rows - 1 else "", fontsize=label_fontsize)
                continue

            # Draw the reference line once per active cell
            if difference:
                ax.axhline(0.0, color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)
            else:
                ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)

            for k in active_configs:
                num_draws = pred_list[k].shape[1]
                widths = np.arange(0, num_draws + 2) / (num_draws + 1)

                estimates_cell = pred_list[k][:, :, r, c][..., None]  # (batch, draws, 1)
                targets_cell = true_list[k][:, r, c][:, None]          # (batch, 1)

                cov_data = compute_empirical_coverage(
                    estimates=estimates_cell,
                    targets=targets_cell,
                    widths=widths,
                    prob=prob,
                    interval_type=interval_type,
                )

                width_rep = cov_data["width_represented"][:, 0]
                cov_est   = cov_data["coverage_estimates"][:, 0]
                cov_low   = cov_data["coverage_lower"][:, 0]
                cov_high  = cov_data["coverage_upper"][:, 0]

                color_k = colors[k]

                if difference:
                    y     = cov_est - width_rep
                    y_low = cov_low - width_rep
                    y_hi  = cov_high - width_rep
                else:
                    y     = cov_est
                    y_low = cov_low
                    y_hi  = cov_high

                ax.fill_between(width_rep, y_low, y_hi, color=color_k, alpha=ci_alpha)
                ax.plot(width_rep, y, color=color_k, alpha=1.0, label=labels[k])


            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.set_box_aspect(1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.35)

            ax.set_ylabel(ylabel if c == 0 else "", fontsize=label_fontsize)
            ax.set_xlabel("Interval width" if r == num_rows - 1 else "", fontsize=label_fontsize)
            if r == 0 and variable_names is not None:
                ax.set_title(variable_names[c], fontsize=title_fontsize)

    _add_mask_legend(fig, parameter_masks, colors, labels, num_rows, num_cols,
                     label_fontsize=legend_fontsize, thumb_scale=thumb_scale)
    return fig


if __name__ == "__main__":
    from bayesgpt.simulators.context_manager import ContextManager

    intrinsic_params = ["v", "a", "z", "tau", "p1", "p2", "p3", "p4"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$p_1$", r"$p_2$", r"$p_3$", r"$p_4$"]
    max_num_categories = 2
    batch_size = 200
    draws = 256

    cm = ContextManager()
    design_configs = [
        cm.build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=2,
            keep_intercept=True,
            add_interaction=True,
        )
        for _ in range(12)
    ]
    labels = [f"Config {i + 1}" for i in range(12)]

    def make_data(dc, batch, draws, max_cats):
        cm = ContextManager()
        mask = cm.build_parameter_mask(
            design_config=dc,
            max_num_categories=max_cats,
            intrinsic_params=intrinsic_params,
            keep_intercept=True
        )
        nr, nc = mask.shape
        true = np.random.normal(0, 1, (batch, nr, nc))
        # Good coverage: draws from same marginal as true → calibrated by exchangeability
        pred = np.random.normal(0, 1, (batch, draws, nr, nc))
        return true, pred

    true_list, pred_list = zip(*[make_data(dc, batch_size, draws, max_num_categories) for dc in design_configs])

    fig = ensemble_coverage(
        true_list=list(true_list),
        pred_list=list(pred_list),
        design_configs=design_configs,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        variable_names=variable_names,
        labels=labels,
    )
    fig.savefig("test_ensemble_coverage.pdf")
    print("done")
