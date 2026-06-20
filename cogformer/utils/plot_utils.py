import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


def make_quadratic(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "black"):
    """
    Utility to make subplots quadratic to avoid visual illusions
    in, e.g., recovery plot.
    """

    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())

    span = upper - lower
    if span == 0:
        span = 1.0
    eps = span * 0.1

    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))

    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color=color,
        alpha=0.7,
        linestyle="dashed",
    )

def credible_interval(x: np.ndarray, prob: float = 0.95, axis: int = None, **kwargs) -> np.ndarray:
    """
    Compute credible interval from samples using quantiles.

    Parameters
    ----------
    x : array_like
        Input array of samples from a posterior distribution or bootstrap samples.
    prob : float, default 0.95
        Coverage probability of the credible interval (between 0 and 1).
        For example, 0.95 gives a 95% credible interval.
    axis : Sequence[int]
        Axis or axes along which the credible interval is computed.
        Default is None (flatten array).

    Returns
    -------
    a numpy array of shape (2, ...) with the first dimension indicating the
    lower and upper bounds of the credible interval.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate posterior samples
    >>> samples = np.random.normal(size=(10, 1000, 3))

    >>> # Different coverage probabilities
    >>> credible_interval(samples, prob=0.5, axis=1)  # 50% CI
    >>> credible_interval(samples, prob=0.99, axis=1)  # 99% CI
    """

    # Input validation
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be between 0 and 1, got {prob}")

    # Calculate tail probabilities
    alpha = 1 - prob
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Compute quantiles
    return np.quantile(x, q=(lower_q, upper_q), axis=axis, **kwargs)

def _add_mask_legend(
    fig: plt.Figure,
    parameter_masks: list,
    colors: list,
    labels: list,
    num_rows: int,
    num_cols: int,
    label_fontsize: int = 11,
    pad: float = 0.012,
    gap: float = 0.025,
    pixel_alphas: list[float] | None = None,
    thumb_scale: float = 1.0,
):
    """
    Add a row of parameter mask thumbnails as a visual legend at the bottom
    of a figure. Active cells are drawn in the config's color; inactive cells
    in light gray. Pixels are forced square. Config labels appear below each
    thumbnail.

    Parameters
    ----------
    fig : plt.Figure
    parameter_masks : list of np.ndarray
        One binary mask per config, shape (num_rows_k, num_cols).
    colors : list
        One matplotlib color per config.
    labels : list of str
        One label per config, displayed below the thumbnail.
    num_rows : int
        Global max num_rows (used to pad shorter masks).
    num_cols : int
        Number of columns (intrinsic parameters).
    label_fontsize : int
    pad : float
        Vertical padding above and below thumbnails, in figure coords.
    gap : float
        Horizontal gap between thumbnails, in figure coords.
    """
    figW, figH = fig.get_size_inches()
    n = len(parameter_masks)

    # Fixed thumbnail width: n=12 fills ~full width (0.95)
    thumb_w_fig  = (0.95 - 11 * gap) / 12 * thumb_scale
    avail_w_fig  = n * thumb_w_fig + (n - 1) * gap
    total_gaps   = gap * (n - 1)
    # Centre the thumbnail strip horizontally
    left_margin  = (1.0 - avail_w_fig) / 2
    thumb_w_in   = thumb_w_fig * figW
    thumb_h_in   = thumb_w_in * num_rows / num_cols   # square pixels
    thumb_h_fig  = thumb_h_in / figH

    # Rough label height in figure coords (1 line of text)
    label_h_fig = (label_fontsize * 2.2 / 72) / figH

    strip_h_fig = pad + label_h_fig + thumb_h_fig + pad

    # Re-layout main subplots leaving the bottom strip free
    fig.tight_layout(rect=[0, strip_h_fig, 1, 1])

    # Thin dividing line between main plots and thumbnail strip
    line = Line2D(
        [0.02, 0.98], [strip_h_fig, strip_h_fig],
        transform=fig.transFigure,
        color="lightgray", linewidth=0.7, zorder=10,
    )
    fig.add_artist(line)

    for k in range(n):
        mask_k = np.array(parameter_masks[k])
        # Pad to global num_rows
        if mask_k.shape[0] < num_rows:
            mask_k = np.vstack([
                mask_k,
                np.zeros((num_rows - mask_k.shape[0], num_cols))
            ])

        # Build RGBA image: active = config color, inactive = light gray
        img = np.ones((num_rows, num_cols, 4))
        img[..., :3] = 0.92   # inactive gray
        dark_gray = np.array([0.25, 0.25, 0.25])
        config_rgb = np.array(mcolors.to_rgb(colors[k]))
        if pixel_alphas is not None:
            alpha_k = pixel_alphas[k]
            if np.ndim(alpha_k) == 0:
                # Scalar: uniform fade for all active pixels
                t = float(np.clip(alpha_k, 0.0, 1.0))
                cell_rgb = t * config_rgb + (1 - t) * dark_gray
                img[mask_k == 1.0, :3] = cell_rgb
                img[mask_k == 1.0,  3] = 1.0
            else:
                # 2D array: per-cell fade
                alpha_arr = np.clip(np.array(alpha_k), 0.0, 1.0)
                active_rows, active_cols = np.where(mask_k == 1.0)
                for ar, ac in zip(active_rows, active_cols):
                    t = float(alpha_arr[ar, ac])
                    img[ar, ac, :3] = t * config_rgb + (1 - t) * dark_gray
                    img[ar, ac,  3] = 1.0
        else:
            img[mask_k == 1.0, :3] = config_rgb
            img[mask_k == 1.0,  3] = 1.0

        x0 = left_margin + k * (thumb_w_fig + gap)
        y0 = pad + label_h_fig

        ax_t = fig.add_axes([x0, y0, thumb_w_fig, thumb_h_fig])
        ax_t.imshow(img, aspect="equal", interpolation="nearest")

        # Major ticks off, minor ticks at pixel boundaries for gridlines
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        ax_t.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
        ax_t.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
        ax_t.grid(which="minor", color="white", linewidth=0.4, alpha=0.8)
        ax_t.tick_params(which="minor", length=0)

        for sp in ax_t.spines.values():
            sp.set_visible(False)
        ax_t.set_xlabel(labels[k], fontsize=label_fontsize, labelpad=3, fontweight="bold")

    return strip_h_fig


def hex_code():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def bf_colors():
    colors = {
        # BF
        "intercept": "#284734",
        "main_effect": "#275D38",
        "interaction": "#9CAF88",
    }
    return colors

def cogformer_cm_colors():
    colors = {
        # CogFormer-CM
        "intercept": "#15435F",
        "main_effect": "#007396",
        "interaction": "#3EB1C8",
    }
    return colors

def cogformer_vi_colors():
    colors = {
        # CogFormer-VI
        "intercept": "#4E2A84",
        "main_effect": "#6969FF",
        "interaction": "#47b5ff",
    }

    return colors

def cogformer_mf_colors():
    colors = {
        # CogFormer model family (magenta/red hue)
        "intercept": "#59315F",
        "main_effect": "#EC008C",
        "interaction": "#FF6969",
    }

    return colors

def cogformer_mc_colors():
    colors = {
        # CogFormer model class (purple/blue hue)
        "intercept": "#4E2A84",
        "main_effect": "#6969FF",
        "interaction": "#47b5ff",
    }

    return colors

def interpolate_palette(colors_dict, n):
    """Return n colors by interpolating intercept → main_effect → interaction."""
    anchors = [colors_dict["intercept"], colors_dict["main_effect"], colors_dict["interaction"]]
    cmap = mcolors.LinearSegmentedColormap.from_list("palette", anchors)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]

def staedtler_fineliner():
    colors = {
        "bf": bf_colors(),
        "vi": cogformer_vi_colors(),
        "mf": cogformer_mf_colors(),
        "cm": cogformer_cm_colors()
    }
    return colors
