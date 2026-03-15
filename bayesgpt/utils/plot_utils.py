import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
    label_fontsize: int = 8,
    pad: float = 0.012,
    gap: float = 0.018,
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

    left_margin  = 0.05
    right_margin = 0.05
    avail_w_fig  = 0.65  # smaller than full width so thumbnails are compact
    total_gaps   = gap * (n - 1)
    thumb_w_fig  = (avail_w_fig - total_gaps) / n
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
        rgba_k = np.array(mcolors.to_rgba(colors[k]))
        img[mask_k == 1.0] = rgba_k

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
        ax_t.set_xlabel(labels[k], fontsize=label_fontsize, labelpad=3)


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

def bayesgpt_cm_colors():
    colors = {
        # BayesGPT-CM
        "intercept": "#15435F",
        "main_effect": "#007396",
        "interaction": "#3EB1C8",
    }
    return colors

def bayesgpt_vi_colors():
    colors = {
        # BayesGPT-VI
        "intercept": "#4E2A84",
        "main_effect": "#6969FF",
        "interaction": "#47b5ff",
    }

    return colors

def bayesgpt_fm_colors():
    colors = {
        # BayesGPT-FM
        "intercept": "#59315F",
        "main_effect": "#EC008C",
        "interaction": "#FF6969",
    }

    return colors

def staedtler_fineliner():
    colors = {
        "bf": bf_colors(),
        "vi": bayesgpt_vi_colors(),
        "fm": bayesgpt_fm_colors(),
        "cm": bayesgpt_cm_colors()
    }
    return colors
