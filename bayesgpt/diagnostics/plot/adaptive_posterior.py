import numpy as np
import pandas as pd
import seaborn as sns

from bayesgpt.utils.plot_utils import bayesgpt_cm_colors
from bayesgpt.simulators.context_manager import ContextManager


from matplotlib.patches import Patch


def effect_type_from_unfolded_name(name: str) -> str:
    """
    Determine effect type from unfolded column name.

    Expected naming scheme from unfolded_names():
      - "<param>"                          -> intercept
      - "<param> -- $<u_k>$ | $c_<id>$"   -> main
      - "<param> -- $<u_k:u_j>$ | $c_<id>$" -> interaction

    Returns
    -------
    effect_type : {"intercept", "main", "interaction"}
    """
    if not isinstance(name, str) or " -- " not in name:
        return "intercept"

    regressor_part = name.split(" -- ", 1)[1]
    if " | " in regressor_part:
        regressor_part = regressor_part.split(" | ")[0]
    regressor_clean = regressor_part.replace("$", "")
    return "interaction" if ":" in regressor_clean else "main"


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
            for c in range(num_cols):
                names.append(col_labels[c])
        else:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            regressor_key = regressor_keys[regressor_id]
            for c in range(num_cols):
                names.append(f"{col_labels[c]} -- ${regressor_key}$ | $c_{{{category_id}}}$")

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
    ...
    """
    names = unfolded_names(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        col_labels=col_labels,
    )

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


def patch_pairgrid_axes(g: sns.PairGrid, patched_cols: set[str], patch_label: str = "N/A"):
    """
    Patch cells corresponding to inactive parameters with a grey background.
    Works even when PairGrid was created with corner=True.
    """
    for i, row_name in enumerate(g.y_vars):
        for j, col_name in enumerate(g.x_vars):

            ax = g.axes[i, j]

            # <-- NEW: skip non-existent axes (happens when corner=True)
            if ax is None:
                continue

            if (row_name in patched_cols) or (col_name in patched_cols):
                ax.cla()
                ax.set_facecolor("lightgray")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5, 0.5, patch_label,
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=10, color="gray"
                )


def adaptive_posterior(
    samples: np.ndarray | pd.DataFrame,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    variable_names: list[str] = None,
    targets: np.ndarray = None,   # (still unused; consider removing)
    priors: np.ndarray = None,    # now USED if show_prior=True
    parameter_mask: np.ndarray = None,
    col_labels: list[str] | None = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    num_bins: int = 10,
    height: float = 2.5,
    unfold: bool = True,
    add_legend: bool = False,
    show_upper_scatter: bool = True,
    show_prior: bool = False,
    prior_color: str = "0.3",
    prior_alpha: float = 0.12,
):
    """
    Pairplot of posterior samples with adaptive "patching" consistent with adaptive_recovery.

    New options
    -----------
    show_upper_scatter : bool
        If False, omits the upper triangle entirely (diag + lower only).
    show_prior : bool
        If True and `priors` is provided, draws prior samples behind posterior.
    """
    if col_labels is None and variable_names is not None:
        col_labels = variable_names

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

    df_post, active_cols, patched_cols = samples_to_unfolded_df(
        samples=samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=parameter_mask,
        unfold=unfold,
        col_labels=col_labels,
    )

    df_prior = None
    if show_prior and priors is not None:
        df_prior, _, _ = samples_to_unfolded_df(
            samples=priors,
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

    # If we don't want upper triangle, let seaborn handle layout cleanly.
    g = sns.PairGrid(df_post, corner=(not show_upper_scatter), height=height)

    def _diag_hist(x, **kwargs):
        name = getattr(x, "name", "")
        c = colors[effect_type_from_unfolded_name(name)]

        x_post = x.dropna()
        if x_post.size < 2:
            return

        # Prior behind posterior (if available)
        if df_prior is not None and name in df_prior.columns:
            x_pr = df_prior[name].dropna()
            if x_pr.size >= 2:
                sns.histplot(x=x_pr, bins=num_bins, stat="density", color=prior_color, alpha=prior_alpha)

        # Avoid kde warnings for constant vectors
        if np.nanstd(x_post.values) == 0.0:
            sns.histplot(x=x_post, bins=num_bins, stat="density", color=c, alpha=0.9)
            return

        sns.histplot(x=x_post, kde=True, bins=num_bins, stat="density", color=c, alpha=0.9)

    def _lower_kde(x, y, **kwargs):
        xname = getattr(x, "name", "")
        yname = getattr(y, "name", "")

        # Posterior
        post = pd.concat([x, y], axis=1)
        post.columns = ["x", "y"]
        post = post.dropna()
        if post.shape[0] < 10:
            return
        if np.nanstd(post["x"].values) < 1e-12 or np.nanstd(post["y"].values) < 1e-12:
            return

        # Prior behind posterior (if available)
        if df_prior is not None and (xname in df_prior.columns) and (yname in df_prior.columns):
            pr = pd.DataFrame({"x": df_prior[xname], "y": df_prior[yname]}).dropna()
            if pr.shape[0] >= 10 and np.nanstd(pr["x"].values) >= 1e-12 and np.nanstd(pr["y"].values) >= 1e-12:
                sns.kdeplot(x=pr["x"], y=pr["y"], fill=True, color=prior_color, alpha=prior_alpha)

        c = pick_pair_color(xname, yname, colors)
        sns.kdeplot(x=post["x"], y=post["y"], fill=True, color=c, alpha=0.35)

    def _upper_scatter(x, y, **kwargs):
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if tmp.shape[0] < 2:
            return

        c = pick_pair_color(getattr(x, "name", ""), getattr(y, "name", ""), colors)
        sns.scatterplot(x=tmp["x"], y=tmp["y"], linewidth=0, alpha=0.25, color=c)

    g.map_diag(_diag_hist)
    g.map_lower(_lower_kde)

    if show_upper_scatter:
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
    prior_samples = np.random.normal(0.0, 5.0, (num_draws, R, C))
    posterior_samples = np.random.normal(0.0, 1.0, (num_draws, R, C))
    print(posterior_samples.shape)

    g = adaptive_posterior(
        samples=posterior_samples,  # (N, R, C) or (N, R*C)
        priors=prior_samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        unfold=True,  # or False
        show_prior=True,
        show_upper_scatter=False
    )
    g.savefig(f"posterior_pairplot{'_unfolded' if unfold else ''}.pdf")
    print("awesome")