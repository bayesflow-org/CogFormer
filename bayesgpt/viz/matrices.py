import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_matrix(
    mask: np.ndarray,
    fig_size: tuple[int, int] | None = None,
    colormap: str | None = "viridis",
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
):
    if mask.ndim > 2:
        raise ValueError("Masks must be in 2D")
    elif mask.ndim == 1:
        mask = mask[:, np.newaxis]

    cell_size = 0.5
    rows, cols = mask.shape
    fig_size = fig_size or (cols * cell_size, rows * cell_size)

    f, ax = plt.subplots(1, 1, figsize=fig_size)
    sns.heatmap(mask, cmap=colormap, cbar=False, ax=ax, xticklabels=False, yticklabels=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return f

def visualize_matrices(
    masks: np.ndarray,
    fig_size: tuple[int, int] | None = None,
    colormap: str | list[str] = "viridis",
    num_cols: int = 6,
    title: str = None
):
    if masks.ndim > 3:
        raise ValueError("Collection of masks must be in 3D")
    elif masks.ndim == 2:
        masks = masks[:, np.newaxis]

    num_masks = masks.shape[0]
    num_rows = (num_masks + num_cols - 1) // num_cols

    rows, cols = masks.shape[1], masks.shape[2]
    cell_size = 0.5
    fig_size = fig_size or (num_cols * cols * cell_size, num_rows * rows * cell_size)

    f, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    axes = np.atleast_1d(axes).flat

    for i, ax in enumerate(axes):
        if i < num_masks:
            cmap = colormap[i] if isinstance (colormap, list) else colormap
            sns.heatmap(masks[i], cmap=colormap, cbar=False, ax=ax, xticklabels=False, yticklabels=False)
        else:
            ax.set_visible(False)
        ax.set_title(f"{title} {i}")
    plt.tight_layout()
    return f


def visualize_design_configs(
    configs: list[dict[str, list[str]]],
    intrinsic_params: list[str],
    fig_size: tuple[int, int] | None = None,
    colormap: str = "viridis",
) -> plt.Figure:
    # Extract unique regressors across configs
    all_regressors = sorted(set(k for c in configs for k in c))
    num_batches = len(configs)
    num_reg = len(all_regressors)
    num_params = len(intrinsic_params)
    matrix = np.zeros((num_batches, num_reg * num_params))

    for b, config in enumerate(configs):
        for r_idx, reg in enumerate(all_regressors):
            if reg in config:
                for p in config[reg]:
                    p_idx = intrinsic_params.index(p)
                    matrix[b, r_idx * num_params + p_idx] = 1

    cell_size = 0.5  # Inches per cell, consistent with other functions
    fig_size = fig_size or (num_reg * num_params * cell_size, num_batches * cell_size)  # Modified: Auto-size

    f, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(matrix, cmap=colormap, ax=ax, xticklabels=False, yticklabels=False, cbar=False)
    ax.set_title("Design Configs")
    return f
