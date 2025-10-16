import numpy as np

from .matrices import visualize_matrix, visualize_matrices, visualize_design_configs


def visualize(
    self,
    batch: dict[str, np.ndarray],
    intrinsic_params: list[str]
):
    # Design configs (whole)
    fig_configs = visualize_design_configs(batch["design_configs"], intrinsic_params)

    # Design matrices (per batch)
    fig_design = visualize_matrices(batch["design_matrices"], title="Design Matrices")

    # Param masks/matrices (per batch)
    fig_masks = visualize_matrices(batch["param_masks"], title="Parameter Masks")
    fig_mats = visualize_matrices(batch["param_matrices"], title="Parameter Matrices")

    # Masks (whole)
    fig_reg = visualize_matrix(batch["regressor_masks"], title="Regressor Masks")
    fig_disc = visualize_matrix(batch["discrete_masks"], title="Discrete Masks")