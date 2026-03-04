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


def adaptive_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
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
    Heatmap visualization of four adaptive diagnostic metrics arranged in a 2×2 grid.

    Each subplot shows a ``num_rows × num_cols`` heatmap (rows = regressors/intercept,
    cols = intrinsic parameters), colour-coded by metric value. Inactive cells are
    rendered as grey N/A tiles, consistent with ``adaptive_recovery`` and
    ``adaptive_coverage``.

    Metrics
    -------
    - (N)RMSE
    - Posterior Contraction
    - Calibration Error
    - Log Gamma

    Parameters
    ----------
    true : np.ndarray of shape (batch_size, num_rows, num_cols)
    pred : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
    design_config : dict
    intrinsic_params : list[str]
    max_num_categories : int
    parameter_mask : np.ndarray, optional
    variable_names : list[str], optional
    normalize : str or None, optional (default "range")
    aggregation : callable, optional (default np.median)
    resolution : int, optional (default 20)
    min_quantile, max_quantile : float
    num_null_draws : int, optional (default 1000)
    log_gamma_quantile : float, optional (default 0.05)
    intercept_color, main_effect_color, interaction_color : str
        N/A tile accent colours, matching the other adaptive plots.
    figsize : tuple, optional
    title_fontsize, label_fontsize, tick_fontsize, annot_fontsize : int
    fmt : str, optional (default ".3f")
        Format string for cell annotations.

    Returns
    -------
    plt.Figure
    """
    if parameter_mask is None:
        cm = ContextManager()
        parameter_mask = cm.build_parameter_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True,
        )
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    col_labels = variable_names if variable_names is not None else intrinsic_params
    regressor_keys = list(design_config.keys())

    batch_size, num_draws, num_rows, num_cols = pred.shape

    # ------------------------------------------------------------------
    # Pre-compute shared quantities
    # ------------------------------------------------------------------
    alphas = np.linspace(min_quantile, max_quantile, resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    null_dist = _gamma_null_distribution(batch_size, num_draws, num_null_draws)
    null_quantile = np.quantile(null_dist, log_gamma_quantile)

    # ------------------------------------------------------------------
    # Build metric grids  (NaN where masked)
    # ------------------------------------------------------------------
    rmse_col = "RMSE" if normalize is None else "NRMSE"

    grids = {
        rmse_col: np.full((num_rows, num_cols), np.nan),
        "Posterior Contraction": np.full((num_rows, num_cols), np.nan),
        "Calibration Error": np.full((num_rows, num_cols), np.nan),
        "Log Gamma": np.full((num_rows, num_cols), np.nan),
    }

    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] != 1.0:
                continue
            estimates = pred[:, :, r, c]
            targets = true[:, r, c]
            grids[rmse_col][r, c] = _rmse(estimates, targets, normalize, aggregation)
            grids["Posterior Contraction"][r, c] = _contraction(estimates, targets, aggregation)
            grids["Calibration Error"][r, c] = _calibration_error(
                estimates, targets, alphas, lowers, uppers, aggregation
            )
            grids["Log Gamma"][r, c] = _log_gamma(estimates, targets, null_quantile)

    # ------------------------------------------------------------------
    # Row / column axis labels
    # ------------------------------------------------------------------
    row_labels = []
    row_colors = []
    for r in range(num_rows):
        if r == 0:
            row_labels.append(r"$1$")
            row_colors.append(intercept_color)
        else:
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            regressor_key = regressor_keys[regressor_id]
            row_labels.append(fr"${regressor_key}$")
            row_colors.append(
                interaction_color if ":" in regressor_key else main_effect_color
            )

    # ------------------------------------------------------------------
    # Colormap specs  (seaborn palette name)
    # ------------------------------------------------------------------
    metric_specs = [
        (rmse_col,                "mako",   0.0, 1.0),
        ("Posterior Contraction", "rocket", 0.0, 1.0),
        ("Calibration Error",     "crest",  0.0, 1.0),
        ("Log Gamma",             "flare",  None, None),
    ]

    cell_size = 1.0  # inches per cell
    if figsize is None:
        figsize = (
            4 * num_cols * cell_size + 2.0,
            num_rows * cell_size + 2.5,
        )

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    for ax, (metric_name, palette, vmin, vmax) in zip(axes, metric_specs):
        grid = grids[metric_name]
        active = parameter_mask == 1.0

        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = sns.color_palette(palette, as_cmap=True)

        # Masked array — matplotlib leaves NaN cells transparent
        masked = np.ma.masked_where(~active, grid)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")

        # Annotate active cells with numeric values
        for r in range(num_rows):
            for c in range(num_cols):
                if active[r, c]:
                    rgba = cmap(norm(grid[r, c]))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "white" if luminance < 0.5 else "black"
                    ax.text(
                        c, r,
                        format(grid[r, c], fmt),
                        ha="center", va="center",
                        fontsize=annot_fontsize,
                        color=text_color,
                    )

        # Render inactive cells as tinted N/A tiles
        for r in range(num_rows):
            for c in range(num_cols):
                if not active[r, c]:
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
    from bayesgpt.simulators.context_manager import ContextManager
    from bayesgpt.utils.plot_utils import bayesgpt_cm_colors

    color = bayesgpt_cm_colors()

    design_config = {
        "1": ["v", "a", "z", "tau"],
        "u_1": ["v", "a", "tau"],
        "u_2": ["v", "a", "z"],
        "u_1:u_2": ["v", "a"],
    }
    intrinsic_params = design_config["1"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]

    num_categories = 2
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
    batch_size, num_draws = 200, 256

    true = np.random.normal(0.0, 1.0, (batch_size, num_rows, num_cols))
    pred = true[:, None, :, :] + np.random.normal(0.0, 0.5, (batch_size, num_draws, num_rows, num_cols))

    fig = adaptive_metrics(
        true=true,
        pred=pred,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=num_categories,
        parameter_mask=parameter_mask,
        variable_names=variable_names,
        intercept_color=color["intercept"],
        main_effect_color=color["main_effect"],
        interaction_color=color["interaction"],
    )

    fig.savefig("test_adaptive_metrics.pdf", bbox_inches="tight")
    print("Saved test_adaptive_metrics.pdf")