import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections.abc import Callable

from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.utils.plot_utils import credible_interval, bayesgpt_cm_colors, _add_mask_legend


def ensemble_recovery(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    design_configs: list[dict],
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_masks: list[np.ndarray] = None,
    variable_names: list[str] = None,
    colors: list[str] = None,
    n_colors: int = 8,
    palette_lightness: float = 0.75,
    labels: list[str] = None,
    uncertainty_agg: Callable = credible_interval,
    title_fontsize: int = 20,
    label_fontsize: int = 14,
    legend_fontsize: int = 10,
    alpha: float = 0.5,
    figsize: tuple = None
):
    """
    Overlay recovery plots for multiple design configs on a shared parameter matrix grid.

    Parameters
    ----------
    true_list : list of np.ndarray
        Ground-truth arrays, one per design config. Each has shape
        (batch, num_rows_i, num_cols).
    pred_list : list of np.ndarray
        Posterior predictive arrays, one per design config. Each has shape
        (batch, draws, num_rows_i, num_cols) for full posteriors or
        (batch, num_rows_i, num_cols) for point estimates.
    design_configs : list of dict
        Design configuration dicts, one per entry (max 10).
    intrinsic_params : list of str
        Names of the intrinsic (column) parameters.
    max_num_categories : int
        Maximum number of categories used to size the row blocks.
    parameter_masks : list of np.ndarray, optional
        Pre-built parameter masks (one per design config). Built from
        design_configs if not provided.
    variable_names : list of str, optional
        Column header names; defaults to intrinsic_params.
    colors : list of str, optional
        One color per design config. Defaults to tab10 colormap.
    labels : list of str, optional
        Legend labels, one per design config.
    uncertainty_agg : Callable
        Function to compute credible intervals; signature (y, prob, axis).
    title_fontsize, label_fontsize : int
    alpha : float
        Scatter/errorbar transparency for overlaid configs.
    figsize : tuple, optional
    """
    n_configs = len(design_configs)
    if n_configs > 8:
        raise ValueError("ensemble_recovery supports at most 8 design configs.")
    if len(true_list) != n_configs or len(pred_list) != n_configs:
        raise ValueError("true_list, pred_list, and design_configs must have the same length.")

    if variable_names is None:
        variable_names = design_configs[0]["1"] if "1" in design_configs[0] else intrinsic_params

    # Build masks if not provided
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

    # Grid dimensions: max num_rows across all masks, fixed num_cols
    num_rows = max(m.shape[0] for m in parameter_masks)
    num_cols = parameter_masks[0].shape[1]

    # Default colors from husl palette
    if colors is None:
        colors = sns.husl_palette(n_colors, l=palette_lightness)

    if labels is None:
        labels = [f"Config {i + 1}" for i in range(n_configs)]

    if figsize is None:
        figsize = (3 * num_cols, 3 * num_rows)

    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Ensure axarr is always 2D
    if num_rows == 1 and num_cols == 1:
        axarr = np.array([[axarr]])
    elif num_rows == 1:
        axarr = axarr[np.newaxis, :]
    elif num_cols == 1:
        axarr = axarr[:, np.newaxis]

    # Row labels from the first design config
    first_dc = design_configs[0]
    regressor_keys_0 = list(first_dc.keys())

    # Per-config r-value maps for mask legend pixel alphas
    r_maps = [np.zeros((num_rows, num_cols)) for _ in range(n_configs)]

    for r in range(num_rows):
        # Determine ylabel from first design config structure
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

            # Collect data from all configs that are active at (r, c)
            active_entries = []
            for k in range(n_configs):
                mask_k = parameter_masks[k]
                if r >= mask_k.shape[0]:
                    continue
                if mask_k[r, c] != 1.0:
                    continue

                true_k = true_list[k][..., r, c]
                pred_k = pred_list[k]

                if pred_k.ndim == 4:
                    y_k = pred_k[..., r, c]
                    y_mean_k = y_k.mean(axis=1)
                    ci_k = uncertainty_agg(y_k, prob=0.9, axis=1)
                    y_lo_k, y_hi_k = ci_k[0], ci_k[1]
                    y_err_k = np.vstack([y_mean_k - y_lo_k, y_hi_k - y_mean_k])
                    active_entries.append((k, true_k, y_mean_k, y_err_k, colors[k], labels[k]))
                else:
                    y_mean_k = pred_k[..., r, c]
                    active_entries.append((k, true_k, y_mean_k, None, colors[k], labels[k]))

            if not active_entries:
                # N/A cell — no config covers this cell
                ax.set_facecolor("gray")
                ax.patch.set_alpha(0.05)
                if r == num_rows - 1:
                    ax.set_xticks([0.0, 0.5, 1.0])
                    ax.set_xticklabels(["0.0", "0.5", "1.0"])
                    ax.tick_params(axis="x", length=0, labelcolor="none")
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
                ax.set_xlabel("Ground Truth" if r == num_rows - 1 else "", fontsize=label_fontsize)
                continue

            # Compute global limits across all active configs for this cell
            all_x = np.concatenate([e[1] for e in active_entries])
            all_y = np.concatenate([e[2] for e in active_entries])
            lower = min(all_x.min(), all_y.min())
            upper = max(all_x.max(), all_y.max())
            span = upper - lower if upper != lower else 1.0
            eps = span * 0.1
            xlim = (lower - eps, upper + eps)
            ylim = (lower - eps, upper + eps)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.plot(xlim, ylim, color="black", alpha=0.4, linestyle="dashed", linewidth=1.0)

            # Per-config correlations — used for annotation and mask legend pixel alphas
            corrs = [np.corrcoef(e[1], e[2])[0, 1] for e in active_entries]
            mean_corr = float(np.mean(corrs))
            cell_alpha = alpha / len(active_entries)

            for idx, (cfg_k, x_k, y_mean_k, y_err_k, color_k, label_k) in enumerate(active_entries):
                r_maps[cfg_k][r, c] = corrs[idx]
                if y_err_k is not None:
                    ax.errorbar(
                        x_k, y_mean_k, yerr=y_err_k,
                        fmt="none", alpha=cell_alpha * 0.6,
                        linewidth=1.2, color=color_k
                    )
                sns.scatterplot(
                    x=x_k, y=y_mean_k, ax=ax,
                    color=color_k, alpha=cell_alpha,
                )
            ax.text(
                0.1, 0.95, f"r = {mean_corr:.3f}",
                ha="left", va="center",
                transform=ax.transAxes, size=12
            )

            ax.set_box_aspect(1)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.2)
            sns.despine(ax=ax)
            ax.set_ylabel(ylabel if c == 0 else "", fontsize=label_fontsize)
            ax.set_title(variable_names[c] if r == 0 else "", fontsize=title_fontsize)
            ax.set_xlabel("Ground Truth" if r == num_rows - 1 else "", fontsize=label_fontsize)

    pixel_alpha_maps = [np.clip(np.abs(r_maps[k]), 0.2, 1.0) for k in range(n_configs)]
    _add_mask_legend(fig, parameter_masks, colors, labels, num_rows, num_cols,
                     label_fontsize=legend_fontsize, pixel_alphas=pixel_alpha_maps)
    return fig


if __name__ == "__main__":
    intrinsic_params = ["v", "a", "z", "tau"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]
    max_num_categories = 2
    batch_size = 20
    draws = 50

    context_manager = ContextManager()
    design_configs = [
        context_manager.build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=2,
            keep_intercept=True,
            add_interaction=True,
        )
        for _ in range(8)
    ]
    labels = [f"Config {i + 1}" for i in range(8)]

    def make_data(dc, batch, draws, max_cats, poor_recovery=False):
        context_manager = ContextManager()
        mask = context_manager.build_parameter_mask(
            design_config=dc,
            max_num_categories=max_cats,
            intrinsic_params=intrinsic_params,
            keep_intercept=True
        )
        nr, nc = mask.shape
        true = np.random.normal(0, 1, (batch, nr, nc))
        if poor_recovery:
            # Draws unrelated to true → low correlation
            pred = np.random.normal(0, 1, (batch, draws, nr, nc))
        else:
            # Good recovery: draws centered tightly on true
            pred = true[:, None, :, :] + np.random.normal(0, 0.15, (batch, draws, nr, nc))
        return true, pred

    data = [
        make_data(dc, batch_size, draws, max_num_categories, poor_recovery=(i >= 4))
        for i, dc in enumerate(design_configs)
    ]
    true_list, pred_list = zip(*data)

    fig = ensemble_recovery(
        true_list=list(true_list),
        pred_list=list(pred_list),
        design_configs=design_configs,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        variable_names=variable_names,
        labels=labels,
        title_fontsize=18,
        label_fontsize=14,
    )
    fig.savefig("test_ensemble_recovery.pdf")
    print("done")
