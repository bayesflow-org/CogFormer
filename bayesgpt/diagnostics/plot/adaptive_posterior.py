import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bayesgpt.utils.plot_utils import bayesgpt_cm_colors
from bayesgpt.simulators.context_manager import ContextManager


def adaptive_posterior(
    samples: np.ndarray | pd.DataFrame,
    design_config: dict = None,
    intrinsic_params: list[str] = None,
    variable_names: list[str] = None,
    max_num_categories: int = None,
    parameter_mask: np.ndarray = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    label_fontsize: int = 14,
    title_fontsize: int = 14,
    legend_fontsize: int = 14,
    num_bins: int = 10,
    height: int = 2.5,
):

    if isinstance(samples, np.ndarray):
        samples = pd.DataFrame(samples, columns=variable_names)
    g = sns.PairGrid(samples, corner=False, height=height)

    # diagonal: 1D hist
    g.map_diag(sns.histplot, kde=True, bins=num_bins, color=intercept_color)

    # lower triangle: 2D KDE
    g.map_lower(sns.kdeplot, fill=True, color=intercept_color, alpha=0.4)

    # upper triangle: scatter
    g.map_upper(sns.scatterplot, linewidth=0.2, alpha=0.3, color=intercept_color)
    return g


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

    import pandas as pd

    rng = np.random.default_rng(0)

    # Fake parameter samples (N samples x D params)
    N = 500
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]
    num_params = len(variable_names)

    samples = rng.normal(size=(N, num_params))

    g = adaptive_posterior(samples, variable_names=variable_names, height=2.0)
    g.savefig("adaptive_posterior.pdf")
    print("awesome")