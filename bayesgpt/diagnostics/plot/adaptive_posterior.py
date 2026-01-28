import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bayesgpt.utils.plot_utils import bayesgpt_cm_colors
from bayesgpt.simulators.context_manager import ContextManager


def adaptive_posterior(
    samples: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    variable_names: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    label_fontsize: int = 14,
    title_fontsize: int = 14,
    legend_fontsize: int = 14,
    height: int = 2.5,
):
    fig = sns.PairGrid(samples, height=height)





    fig.tight_layout()
    return fig


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
