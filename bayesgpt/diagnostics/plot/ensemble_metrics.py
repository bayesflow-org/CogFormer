import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns

from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.diagnostics.metric.adaptive_metrics import (
    _rmse,
    _contraction,
    _calibration_error,
    _log_gamma,
    _gamma_null_distribution,
)


def ensemble_metrics(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    design_configs: list[dict],
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_masks: list[np.ndarray] = None,
    variable_names: list[str] = None,
    normalize: str | None = "range",
    aggregation=np.median,
    resolution: int = 20,
    min_quantile: float = 0.005,
    max_quantile: float = 0.995,
    num_null_draws: int = 1000,
    log_gamma_quantile: float = 0.05,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    figsize: tuple | None = None,
    title_fontsize: int = 18,
    label_fontsize: int = 15,
    tick_fontsize: int = 13,
    annot_fontsize: int = 11,
    fmt: str = ".3f",
) -> plt.Figure:
    """
    Heatmap of diagnostic metrics averaged over multiple design configs.

    For each active cell (r, c), collects the metric value from every config
    that has that cell active, then displays the mean ± std and the count n
    of contributing configs. The colormap is driven by the mean.

    Metrics
    -------
    - (N)RMSE
    - Posterior Contraction
    - Calibration Error
    - Log Gamma

    Parameters
    ----------
    true_list : list of np.ndarray of shape (batch_size, num_rows_i, num_cols)
        Ground-truth arrays, one per config.
    pred_list : list of np.ndarray of shape (batch_size, num_draws, num_rows_i, num_cols)
        Posterior draw arrays, one per config.
    design_configs : list of dict
        Design configuration dicts. Defaults to 200 configs in the test script.
    intrinsic_params : list[str]
        Ordered list of intrinsic parameter names (columns).
    max_num_categories : int
        Maximum number of categories, determines the row layout.
    parameter_masks : list of np.ndarray, optional
        Pre-built masks; constructed from design_configs if not provided.
    variable_names : list[str], optional
        Display labels for columns.
    normalize : str or None, optional (default "range")
        RMSE normalisation mode.
    aggregation : callable, optional (default np.median)
        Aggregation function for per-dataset metric values.
    resolution : int, optional (default 20)
        Number of alpha levels for calibration error.
    min_quantile, max_quantile : float
        Quantile range for calibration error.
    num_null_draws : int, optional (default 1000)
        Draws for the log-gamma null distribution.
    log_gamma_quantile : float, optional (default 0.05)
        Null distribution quantile for log-gamma threshold.
    intercept_color, main_effect_color, interaction_color : str
        N/A tile accent colors.
    figsize : tuple, optional
    title_fontsize, label_fontsize, tick_fontsize, annot_fontsize : int
    fmt : str, optional (default ".3f")
        Format string for mean and std annotations.

    Returns
    -------
    plt.Figure
    """
    n_configs = len(design_configs)
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
                keep_intercept=True,
            )
            for dc in design_configs
        ]
    else:
        parameter_masks = [
            m[0] if m.ndim == 3 else m
            for m in parameter_masks
        ]

    num_rows = max(m.shape[0] for m in parameter_masks)
    num_cols = parameter_masks[0].shape[1]

    # Pre-compute shared quantities using the first config's shape
    batch_size, num_draws = pred_list[0].shape[0], pred_list[0].shape[1]

    alphas = np.linspace(min_quantile, max_quantile, resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    null_dist = _gamma_null_distribution(batch_size, num_draws, num_null_draws)
    null_quantile = np.quantile(null_dist, log_gamma_quantile)

    rmse_col = "RMSE" if normalize is None else "NRMSE"
    metric_names = [rmse_col, "Posterior Contraction", "Calibration Error", "Log Gamma"]

    # Accumulate per-cell metric values across configs
    accum = {
        m: [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for m in metric_names
    }

    for k in range(n_configs):
        mask_k = parameter_masks[k]
        true_k = true_list[k]
        pred_k = pred_list[k]

        for r in range(mask_k.shape[0]):
            for c in range(num_cols):
                if mask_k[r, c] != 1.0:
                    continue
                estimates = pred_k[:, :, r, c]
                targets = true_k[:, r, c]

                accum[rmse_col][r][c].append(
                    _rmse(estimates, targets, normalize, aggregation)
                )
                accum["Posterior Contraction"][r][c].append(
                    _contraction(estimates, targets, aggregation)
                )
                accum["Calibration Error"][r][c].append(
                    _calibration_error(estimates, targets, alphas, lowers, uppers, aggregation)
                )
                accum["Log Gamma"][r][c].append(
                    _log_gamma(estimates, targets, null_quantile)
                )

    # Build mean, std, and count grids
    mean_grids = {m: np.full((num_rows, num_cols), np.nan) for m in metric_names}
    std_grids  = {m: np.full((num_rows, num_cols), np.nan) for m in metric_names}
    count_grid = np.zeros((num_rows, num_cols), dtype=int)

    for r in range(num_rows):
        for c in range(num_cols):
            vals = accum[metric_names[0]][r][c]
            n = len(vals)
            count_grid[r, c] = n
            if n == 0:
                continue
            for m in metric_names:
                v = np.array(accum[m][r][c])
                mean_grids[m][r, c] = float(np.mean(v))
                std_grids[m][r, c]  = float(np.std(v, ddof=1) if n > 1 else 0.0)

    # Row labels and colors (from the widest config = first design config)
    first_dc = design_configs[0]
    regressor_keys_0 = list(first_dc.keys())
    row_labels = []
    row_colors = []
    for r in range(num_rows):
        if r == 0:
            row_labels.append(r"$1$")
            row_colors.append(intercept_color)
        else:
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            if regressor_id < len(regressor_keys_0):
                regressor_key = regressor_keys_0[regressor_id]
                row_labels.append(fr"${regressor_key}$")
                row_colors.append(
                    interaction_color if ":" in regressor_key else main_effect_color
                )
            else:
                row_labels.append(f"Row {r}")
                row_colors.append(main_effect_color)

    col_labels = variable_names if variable_names is not None else intrinsic_params

    metric_specs = [
        (rmse_col,                "mako",   0.0, 1.0),
        ("Posterior Contraction", "rocket", 0.0, 1.0),
        ("Calibration Error",     "crest",  0.0, 1.0),
        ("Log Gamma",             "flare",  None, None),
    ]

    cell_size = 1.0
    if figsize is None:
        figsize = (4 * num_cols * cell_size + 2.0, num_rows * cell_size + 2.5)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    active_any = count_grid > 0

    for ax, (metric_name, palette, vmin, vmax) in zip(axes, metric_specs):
        mean_grid = mean_grids[metric_name]
        std_grid  = std_grids[metric_name]

        if vmin is None:
            vmin = np.nanmin(mean_grid)
        if vmax is None:
            vmax = np.nanmax(mean_grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = sns.color_palette(palette, as_cmap=True)

        masked = np.ma.masked_where(~active_any, mean_grid)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")

        # Annotate active cells: mean / ±std / n=k
        for r in range(num_rows):
            for c in range(num_cols):
                n = count_grid[r, c]
                if n == 0:
                    continue
                rgba = cmap(norm(mean_grid[r, c]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                # Mean ± std (larger, bold), upper portion of cell
                ax.text(
                    c, r - 0.12,
                    f"{mean_grid[r, c]:{fmt}}\n±{std_grid[r, c]:{fmt}}",
                    ha="center", va="center",
                    fontsize=annot_fontsize,
                    fontweight="bold",
                    color=text_color,
                    linespacing=1.3,
                )
                # Dividing line
                ax.plot(
                    [c - 0.4, c + 0.4], [r + 0.12, r + 0.12],
                    color=text_color, linewidth=0.5, alpha=0.4,
                )
                # (n=k) — smaller font, lower portion of cell
                ax.text(
                    c, r + 0.28, f"(n={n})",
                    ha="center", va="center",
                    fontsize=annot_fontsize - 2,
                    color=text_color,
                )

        # N/A tiles for cells with no contributing config
        for r in range(num_rows):
            for c in range(num_cols):
                if count_grid[r, c] == 0:
                    tint = row_colors[r]
                    ax.add_patch(
                        Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor=tint, alpha=0.08, zorder=2,
                        )
                    )
                    ax.text(
                        c, r, "N/A",
                        ha="center", va="center",
                        fontsize=annot_fontsize + 1,
                        fontweight="bold",
                        color=tint, alpha=0.7, zorder=3,
                    )

        ax.set_title(metric_name, fontsize=title_fontsize)
        ax.set_xticks(range(num_cols))
        ax.set_xticklabels(col_labels, fontsize=tick_fontsize)
        ax.set_yticks(range(num_rows))
        ax.set_yticklabels(row_labels, fontsize=label_fontsize)
        ax.tick_params(axis="both", length=0)
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(num_rows - 0.5, -0.5)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    intrinsic_params = ["v", "a", "z", "tau"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]
    max_num_categories = 2
    n_configs = 200
    batch_size = 50
    draws = 64

    cm = ContextManager()
    design_configs = [
        cm.build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=2,
            keep_intercept=True,
            add_interaction=True,
        )
        for _ in range(n_configs)
    ]

    def make_data(dc, batch, draws, max_cats):
        mask = cm.build_parameter_mask(
            design_config=dc,
            max_num_categories=max_cats,
            intrinsic_params=intrinsic_params,
            keep_intercept=True,
        )
        nr, nc = mask.shape
        true = np.random.normal(0, 1, (batch, nr, nc))
        # Good recovery: draws centered tightly on true
        pred = true[:, None, :, :] + np.random.normal(0, 0.15, (batch, draws, nr, nc))
        return true, pred

    true_list, pred_list = zip(*[
        make_data(dc, batch_size, draws, max_num_categories)
        for dc in design_configs
    ])

    fig = ensemble_metrics(
        true_list=list(true_list),
        pred_list=list(pred_list),
        design_configs=design_configs,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        variable_names=variable_names,
        num_null_draws=200,
        title_fontsize=16,
        label_fontsize=13,
        annot_fontsize=8,
    )
    fig.savefig("test_ensemble_metrics.pdf", bbox_inches="tight")
    print("done")
