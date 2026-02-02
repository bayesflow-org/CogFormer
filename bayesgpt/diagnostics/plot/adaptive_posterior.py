import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bayesgpt.utils.plot_utils import bayesgpt_cm_colors
from bayesgpt.simulators.context_manager import ContextManager


from matplotlib.patches import Patch


def effect_type_from_unfolded_name(name: str) -> str:
    """
    Determine effect type from unfolded column name.

    Expected naming scheme from unfolded_names():
      - "1:<param>"                 -> intercept
      - "<u_k>|c<id>:<param>"        -> main
      - "<u_k:u_j>|c<id>:<param>"    -> interaction

    Returns
    -------
    effect_type : {"intercept", "main", "interaction"}
    """
    if not isinstance(name, str) or ":" not in name:
        return "intercept"

    row_label = name.split(":", 1)[0]  # "1" or "u_1|c1" or "u_1:u_2|c1"
    if row_label == "1":
        return "intercept"

    reg_key = row_label.split("|", 1)[0]  # "u_1" or "u_1:u_2"
    return "interaction" if ":" in reg_key else "main"


def pick_pair_color(x_name: str, y_name: str, colors: dict[str, str]) -> str:
    """
    Pick a single color for a (x,y) panel.
    Rule: highest-order present wins: interaction > main > intercept.
    """
    tx = effect_type_from_unfolded_name(x_name)
    ty = effect_type_from_unfolded_name(y_name)

    if tx == "interaction" or ty == "interaction":
        return colors["interaction"]
    if tx == "main" or ty == "main":
        return colors["main"]
    return colors["intercept"]

def unfolded_names(
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    col_labels: list[str] | None = None,
):
    """
    Names aligned with the unfolded parameter matrix.

    Parameters
    ----------
    col_labels : list[str] | None
        Optional display labels for columns (e.g. LaTeX r"$v$").
        If None, uses intrinsic_params.
    """
    regressor_keys = list(design_config.keys())

    num_regressors = len([k for k in regressor_keys if k != "1"])
    num_rows = 1 + num_regressors * (max_num_categories - 1)

    if col_labels is None:
        col_labels = intrinsic_params
    num_cols = len(col_labels)

    names = []
    for r in range(num_rows):
        if r == 0:
            row_label = "1"
        else:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            regressor_key = regressor_keys[regressor_id]
            row_label = f"{regressor_key}|c{category_id}"

        for c in range(num_cols):
            names.append(f"{row_label}:{col_labels[c]}")

    return names


def samples_to_unfolded_df(
    samples: np.ndarray | pd.DataFrame,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray,
    unfold: bool,
    col_labels: list[str] | None = None,
):
    """
    Convert samples into a DataFrame aligned with the unfolded parameter matrix.

    Expected sample shapes (pick one):
      - (N, R, C) posterior over matrix elements
      - (N, R*C)  already-flattened

    If unfold=False, drops masked-out columns.
    If unfold=True, keeps all columns, but masked-out columns are set to NaN.
    """
    names = unfolded_names(design_config, intrinsic_params, max_num_categories)

    if isinstance(samples, pd.DataFrame):
        # assume user already provided correct columns
        df = samples.copy()
        if df.shape[1] != len(names):
            raise ValueError("DataFrame has unexpected number of columns for unfolded layout.")
    else:
        x = np.asarray(samples)
        if x.ndim == 3:
            N, R, C = x.shape
            x2 = x.reshape(N, R * C)
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError("samples must be (N,R,C), (N,R*C), or a DataFrame.")

        if x2.shape[1] != len(names):
            raise ValueError(
                f"Expected {len(names)} columns for unfolded layout, got {x2.shape[1]}."
            )

        df = pd.DataFrame(x2, columns=names)

    # build patched column list from mask (row-major)
    mask_flat = parameter_mask.reshape(-1)
    patched_cols = [names[i] for i, m in enumerate(mask_flat) if float(m) != 1.0]
    active_cols = [names[i] for i, m in enumerate(mask_flat) if float(m) == 1.0]

    if unfold:
        # keep all cols, but set patched columns to NaN so seaborn doesn't draw density/scatter
        df.loc[:, patched_cols] = np.nan
        return df, active_cols, patched_cols

    # only keep active (in-scope) columns
    return df.loc[:, active_cols], active_cols, patched_cols


def patch_pairgrid_axes(g: sns.PairGrid, patched_cols: list[str], patch_label: str = "N/A"):
    """
    Hide axes involving patched variables (either x or y is patched).
    Works only when unfold=True and we included patched columns.
    """
    vars_ = list(g.x_vars)
    patched = set(patched_cols)

    for i, yv in enumerate(vars_):
        for j, xv in enumerate(vars_):
            ax = g.axes[i, j]
            if (xv in patched) or (yv in patched):
                ax.set_facecolor("lightgray")
                ax.patch.set_alpha(0.15)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(
                    0.5, 0.5, patch_label,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=12, weight="bold", alpha=0.6
                )

def adaptive_posterior(
    samples: np.ndarray | pd.DataFrame,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    col_labels: list[str] | None = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    num_bins: int = 10,
    height: float = 2.5,
    unfold: bool = True,
    add_legend: bool = False,
):
    """
    Pairplot of posterior samples with adaptive "patching" consistent with adaptive_recovery:

    - unfold=False: only show elements in design_config scope (mask==1)
    - unfold=True: show unfolded layout, but hide patched elements (mask==0)

    Colors are inferred from the unfolded column labels:
      - "1:..."                -> intercept_color
      - "u_k|c*:..."            -> main_effect_color
      - "u_k:u_j|c*:..."        -> interaction_color
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

    df, active_cols, patched_cols = samples_to_unfolded_df(
        samples=samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=parameter_mask,
        unfold=unfold,
        col_labels=col_labels,
    )

    colors = {
        "intercept": intercept_color,
        "main": main_effect_color,
        "interaction": interaction_color,
    }

    g = sns.PairGrid(df, corner=False, height=height)

    def _diag_hist(x, **kwargs):
        x = x.dropna()
        if x.size < 2:
            return
        # Avoid kde warnings for constant vectors
        if np.nanstd(x.values) == 0.0:
            sns.histplot(x=x, bins=num_bins, color=colors[effect_type_from_unfolded_name(getattr(x, "name", ""))])
            return

        name = getattr(x, "name", "")
        c = colors[effect_type_from_unfolded_name(name)]
        sns.histplot(x=x, kde=True, bins=num_bins, color=c)

    def _lower_kde(x, y, **kwargs):
        # Paired dropna is essential (x and y can have different NaN patterns)
        tmp = pd.concat([x, y], axis=1)
        tmp.columns = ["x", "y"]
        tmp = tmp.dropna()

        # Not enough points → skip (prevents KDE warnings)
        if tmp.shape[0] < 10:
            return

        # Avoid singular KDE (constant / near-constant)
        if np.nanstd(tmp["x"].values) < 1e-12 or np.nanstd(tmp["y"].values) < 1e-12:
            return

        c = pick_pair_color(getattr(x, "name", ""), getattr(y, "name", ""), colors)
        sns.kdeplot(x=tmp["x"], y=tmp["y"], fill=True, color=c, alpha=0.35)

    def _upper_scatter(x, y, **kwargs):
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if tmp.shape[0] < 2:
            return

        c = pick_pair_color(getattr(x, "name", ""), getattr(y, "name", ""), colors)
        sns.scatterplot(x=tmp["x"], y=tmp["y"], linewidth=0, alpha=0.25, color=c)

    g.map_diag(_diag_hist)
    g.map_lower(_lower_kde)
    g.map_upper(_upper_scatter)

    if unfold and len(patched_cols) > 0:
        patch_pairgrid_axes(g, patched_cols=patched_cols, patch_label="N/A")

    if add_legend:
        handles = [
            Patch(facecolor=colors["intercept"], edgecolor="none", label="Intercept"),
            Patch(facecolor=colors["main"], edgecolor="none", label="Main effect"),
            Patch(facecolor=colors["interaction"], edgecolor="none", label="Interaction"),
        ]
        g.legend(handles=handles, loc="upper right", frameon=False)

    return g


# def adaptive_posterior(
#     samples: np.ndarray | pd.DataFrame,
#     design_config: dict = None,
#     intrinsic_params: list[str] = None,
#     variable_names: list[str] = None,
#     max_num_categories: int = None,
#     parameter_mask: np.ndarray = None,
#     intercept_color: str = "#4e2a84",
#     main_effect_color: str = "#6969ff",
#     interaction_color: str = "#ff6969",
#     label_fontsize: int = 14,
#     title_fontsize: int = 14,
#     legend_fontsize: int = 14,
#     num_bins: int = 10,
#     height: int = 2.5,
#     unfold: bool = True
# ):
#
#     if isinstance(samples, np.ndarray):
#         samples = pd.DataFrame(samples, columns=variable_names)
#     g = sns.PairGrid(samples, corner=False, height=height)
#
#     # diagonal: 1D hist
#     g.map_diag(sns.histplot, kde=True, bins=num_bins, color=intercept_color)
#
#     # lower triangle: 2D KDE
#     g.map_lower(sns.kdeplot, fill=True, color=intercept_color, alpha=0.4)
#
#     # upper triangle: scatter
#     g.map_upper(sns.scatterplot, linewidth=0, alpha=0.3, color=intercept_color)
#     return g


def create_labels(
    design_config: dict,
):
    labels = []
    for k, v in design_config.items():
        if k == "1":
            pass

    return labels


if __name__ == "__main__":
    debug = True
    colors = bayesgpt_cm_colors()

    intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

    design_config = {
        "1": intrinsic_params,  # intercept
        "u_1": ["v", "a", "tau", "s_v"],
        "u_2": ["v", "a", "tau"],
        "u_1:u_2": ["v", "a"],
    }
    num_params = len(variable_names)

    cm = ContextManager()

    parameter_mask = cm.build_parameter_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        keep_intercept=True,
    )

    num_draws = 500
    R, C = parameter_mask.shape
    unfold = True

    # posterior over the full unfolded matrix
    posterior_samples = np.random.normal(0.0, 1.0, (num_draws, R, C))
    print(posterior_samples.shape)

    g = adaptive_posterior(
        samples=posterior_samples,  # (N, R, C) or (N, R*C)
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        unfold=True,  # or False
    )
    g.savefig(f"posterior_pairplot{'_unfolded' if unfold else ''}.pdf")
    print("awesome")