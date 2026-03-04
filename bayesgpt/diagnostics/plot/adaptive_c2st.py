import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.diagnostics.metric.adaptive_c2st import compute_c2st_accuracy
from bayesgpt.utils.plot_utils import bayesgpt_cm_colors


def adaptive_c2st(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    figsize: tuple | None = None,
    title_fontsize: int = 16,
    label_fontsize: int = 13,
    tick_fontsize: int = 11,
    annot_fontsize: int = 9,
    fmt: str = ".3f",
    n_splits: int = 5,
    hidden_layer_sizes: tuple = (32,),
    max_iter: int = 200,
    random_state: int = 0,
) -> plt.Figure:
    """
    Heatmap of C2ST accuracy over the adaptive regressor-by-parameter grid.

    For each active cell, trains a small MLP to distinguish posterior samples
    from two approximators (e.g. BayesGPT vs BayesFlow). Accuracy of 0.5
    means the posteriors are indistinguishable; 1.0 means fully separable.

    Parameters
    ----------
    pred_a : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
        Posterior samples from approximator A (e.g. BayesGPT).
    pred_b : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
        Posterior samples from approximator B (e.g. BayesFlow).
    design_config : dict
        Mapping from regressor keys to lists of active parameter names.
    intrinsic_params : list[str]
        Ordered list of intrinsic parameter names (column order).
    max_num_categories : int
        Maximum number of categories, determines the row layout.
    parameter_mask : np.ndarray of shape (num_rows, num_cols), optional
        Binary mask (1 = active). Built from ContextManager if not provided.
    variable_names : list[str], optional
        Display labels for columns. Falls back to intrinsic_params if not given.
    intercept_color : str
        N/A tile accent color for the intercept row.
    main_effect_color : str
        N/A tile accent color for main effect rows.
    interaction_color : str
        N/A tile accent color for interaction rows.
    figsize : tuple or None
        Figure size. Defaults to (3.2 * num_cols, 2.5 * num_rows).
    title_fontsize : int
    label_fontsize : int
    tick_fontsize : int
    annot_fontsize : int
    fmt : str
        Format string for cell annotations (default ".3f").
    n_splits : int, optional, default: 5
        Number of stratified cross-validation folds.
    hidden_layer_sizes : tuple, optional, default: (32,)
        Hidden layer sizes for the MLP classifier.
    max_iter : int, optional, default: 200
        Maximum training iterations per fold.
    random_state : int, optional, default: 0
        Random seed for reproducibility.

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
    _, _, num_rows, num_cols = pred_a.shape

    # Row labels and colors
    row_labels = []
    row_colors = []
    for r in range(num_rows):
        if r == 0:
            row_labels.append(r"$1$")
            row_colors.append(intercept_color)
        else:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            regressor_key = regressor_keys[regressor_id]
            row_labels.append(fr"${regressor_key}$")
            row_colors.append(interaction_color if ":" in regressor_key else main_effect_color)

    # Compute C2ST accuracy for each active cell
    grid = np.full((num_rows, num_cols), np.nan)
    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] != 1.0:
                continue
            grid[r, c] = compute_c2st_accuracy(
                samples_a=pred_a[:, :, r, c].ravel(),
                samples_b=pred_b[:, :, r, c].ravel(),
                n_splits=n_splits,
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
            )

    if figsize is None:
        figsize = (3.2 * num_cols, 2.5 * num_rows)

    fig, ax = plt.subplots(figsize=figsize)

    active = parameter_mask == 1.0
    # Colormap: green (0.5, indistinguishable) → red (1.0, fully separable)
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    cmap = plt.get_cmap("RdYlGn_r")

    masked = np.ma.masked_where(~active, grid)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="C2ST Accuracy")

    # Annotate active cells
    for r in range(num_rows):
        for c in range(num_cols):
            if active[r, c]:
                ax.text(
                    c, r,
                    format(grid[r, c], fmt),
                    ha="center", va="center",
                    fontsize=annot_fontsize,
                    color="black",
                )

    # N/A tiles for inactive cells
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

    ax.set_title("C2ST Accuracy", fontsize=title_fontsize)
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

    color = bayesgpt_cm_colors()

    design_config = {
        "1": ["v", "a", "z", "tau"],
        "u_1": ["v", "a", "tau"],
        "u_2": ["v", "a", "z"],
        "u_1:u_2": ["v", "a"],
    }

    intrinsic_params = design_config["1"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]

    num_categories = 3
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

    # Simulate two approximators: pred_a is well-calibrated, pred_b has a slight shift
    true = np.random.normal(0.0, 1.0, (batch_size, num_rows, num_cols))
    pred_a = true[:, None, :, :] + np.random.normal(0.0, 0.5, (batch_size, num_draws, num_rows, num_cols))
    pred_b = true[:, None, :, :] + np.random.normal(0.2, 0.6, (batch_size, num_draws, num_rows, num_cols))

    print("pred_a:", pred_a.shape)
    print("pred_b:", pred_b.shape)
    print("mask:", parameter_mask.shape)

    fig = adaptive_c2st(
        pred_a=pred_a,
        pred_b=pred_b,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=num_categories,
        parameter_mask=parameter_mask,
        variable_names=variable_names,
        intercept_color=color["intercept"],
        main_effect_color=color["main_effect"],
        interaction_color=color["interaction"],
        n_splits=5,
        random_state=0,
    )

    fig.savefig("test_adaptive_c2st.pdf", bbox_inches="tight")
    print("Saved test_adaptive_c2st.pdf")
