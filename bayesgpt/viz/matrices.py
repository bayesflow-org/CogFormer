import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_matrix(
    mask: np.ndarray,
    fig_size: tuple[int, int],
    colormap: str | None = "viridis",
):
    if mask.ndim > 2:
        raise ValueError("Masks must be in 2D")
    elif mask.ndim == 1:
        mask = mask[:, np.newaxis]

    f, ax = plt.subplots(1, 1, figsize=fig_size)
    sns.heatmap(mask, cmap=colormap)

    return f

def visualize_matrices(
    masks: np.ndarray,
    fig_size: tuple[int, int],
    colormap: str | None = "viridis",
    num_cols: int = 6,
):
    if masks.ndim > 3:
        raise ValueError("Collection of masks must be in 3D")
    elif masks.ndim == 2:
        masks = masks[:, np.newaxis]

    num_masks = masks.shape[0]
    num_rows = num_masks // num_cols

    f, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)

    for i, ax in enumerate(axes.flat):
        mask = masks[i, :, :]
        sns.heatmap(mask, cmap=colormap, ax=ax)

    plt.tight_layout()
    return f
