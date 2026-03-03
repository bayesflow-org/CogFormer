import numpy as np
import pandas as pd
from scipy.stats import binom

from bayesgpt.simulators.context_manager import ContextManager


# ---------------------------------------------------------------------------
# Log-gamma helpers (adapted from BayesFlow)
# ---------------------------------------------------------------------------

def _gamma_discrepancy(ranks: np.ndarray, num_post_draws: int) -> float:
    """Gamma discrepancy statistic for a single parameter's rank distribution."""
    num_ranks = len(ranks)
    R_i = np.array([np.sum(ranks < i) for i in range(1, num_post_draws + 2)])
    z_i = np.arange(1, num_post_draws + 2) / (num_post_draws + 1)
    bin_1 = binom.cdf(R_i, num_ranks, z_i)
    bin_2 = 1 - binom.cdf(R_i - 1, num_ranks, z_i)
    return float(2 * np.min(np.minimum(bin_1, bin_2)))


def _gamma_null_distribution(
    num_ranks: int,
    num_post_draws: int,
    num_null_draws: int,
) -> np.ndarray:
    """Null distribution of gamma under uniformity of ranks."""
    z_i = np.arange(1, num_post_draws + 2) / (num_post_draws + 1)
    gamma = np.empty(num_null_draws)
    for i in range(num_null_draws):
        u = np.random.uniform(size=num_ranks)
        F_z = np.mean(u[:, None] < z_i, axis=0)
        bin_1 = binom.cdf(num_ranks * F_z, num_ranks, z_i)
        bin_2 = 1 - binom.cdf(num_ranks * F_z - 1, num_ranks, z_i)
        gamma[i] = 2 * np.min(np.minimum(bin_1, bin_2))
    return gamma


# ---------------------------------------------------------------------------
# Per-parameter metric computations
# ---------------------------------------------------------------------------

def _rmse(
    estimates: np.ndarray,
    targets: np.ndarray,
    normalize: str | None,
    aggregation,
) -> float:
    """
    Parameters
    ----------
    estimates : (batch_size, num_draws)
    targets   : (batch_size,)
    """
    rmse = np.sqrt(np.mean((estimates - targets[:, None]) ** 2, axis=0))  # (num_draws,)

    if normalize is None or normalize is False:
        pass
    elif normalize == "mean":
        denom = np.abs(np.mean(targets))
        rmse /= denom if denom > 0 else 1.0
    elif normalize == "median":
        denom = np.abs(np.median(targets))
        rmse /= denom if denom > 0 else 1.0
    elif normalize == "range":
        denom = targets.max() - targets.min()
        rmse /= denom if denom > 0 else 1.0
    elif normalize == "std":
        denom = np.std(targets, ddof=0)
        rmse /= denom if denom > 0 else 1.0
    elif normalize == "iqr":
        q75, q25 = np.percentile(targets, [75, 25])
        denom = q75 - q25
        rmse /= denom if denom > 0 else 1.0
    else:
        raise ValueError(f"Unknown normalization mode: {normalize!r}")

    return float(aggregation(rmse))


def _contraction(
    estimates: np.ndarray,
    targets: np.ndarray,
    aggregation,
) -> float:
    """
    Parameters
    ----------
    estimates : (batch_size, num_draws)
    targets   : (batch_size,)
    """
    prior_var = targets.var(ddof=1)
    if prior_var == 0:
        return 0.0
    post_vars = estimates.var(axis=1, ddof=1)  # (batch_size,)
    contraction = np.clip(1 - post_vars / prior_var, 0, 1)
    return float(aggregation(contraction))


def _calibration_error(
    estimates: np.ndarray,
    targets: np.ndarray,
    alphas: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    aggregation,
) -> float:
    """
    Parameters
    ----------
    estimates : (batch_size, num_draws)
    targets   : (batch_size,)
    alphas    : (resolution,)
    lowers    : (resolution,)  lower quantile bounds
    uppers    : (resolution,)  upper quantile bounds
    """
    quantiles = np.quantile(estimates, np.stack([lowers, uppers]), axis=1)  # (2, resolution, batch_size)
    lower_bounds, upper_bounds = quantiles[0], quantiles[1]  # (resolution, batch_size)
    inlier = (lower_bounds <= targets[None, :]) & (upper_bounds >= targets[None, :])
    alpha_pred = np.mean(inlier, axis=1)  # (resolution,)
    abs_errors = np.abs(alpha_pred - alphas)
    return float(aggregation(abs_errors))


def _log_gamma(
    estimates: np.ndarray,
    targets: np.ndarray,
    null_quantile: float,
) -> float:
    """
    Parameters
    ----------
    estimates    : (batch_size, num_draws)
    targets      : (batch_size,)
    null_quantile: pre-computed quantile of the null distribution
    """
    num_draws = estimates.shape[1]
    ranks = np.sum(estimates < targets[:, None], axis=1)  # (batch_size,)
    gamma = _gamma_discrepancy(ranks, num_post_draws=num_draws)
    return float(np.log(gamma / null_quantile))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adaptive_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    normalize: str | None = "range",
    aggregation=np.median,
    resolution: int = 20,
    min_quantile: float = 0.005,
    max_quantile: float = 0.995,
    num_null_draws: int = 1000,
    log_gamma_quantile: float = 0.05,
) -> pd.DataFrame:
    """
    Compute adaptive diagnostic metrics for all active parameters in the
    unfolded design-config layout.

    Metrics computed per active parameter
    --------------------------------------
    - (N)RMSE              : (Normalised) Root Mean Squared Error
    - Posterior Contraction: Reduction in variance from prior to posterior
    - Calibration Error    : Mean absolute deviation from ideal coverage
    - Log Gamma            : Log-ratio of gamma discrepancy to null quantile

    Parameters
    ----------
    true : np.ndarray of shape (batch_size, num_rows, num_cols)
        Ground-truth parameter values.
    pred : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
        Posterior samples.
    design_config : dict
        Mapping from regressor keys (e.g. "1", "u_1", "u_1:u_2") to lists of
        intrinsic parameter names that are active for that regressor.
    intrinsic_params : list[str]
        Ordered list of all intrinsic parameter names (column order).
    max_num_categories : int
        Maximum number of categories, determines the row layout.
    parameter_mask : np.ndarray of shape (num_rows, num_cols), optional
        Binary mask (1 = active, 0 = N/A). Built automatically if not provided.
    variable_names : list[str], optional
        Display labels for columns (e.g. LaTeX strings like r"$v$").
        Falls back to ``intrinsic_params`` if not given.
    normalize : str or None, optional (default "range")
        RMSE normalisation: "range", "std", "iqr", "mean", "median", or None.
    aggregation : callable, optional (default np.median)
        Function used to aggregate metric values across datasets/draws.
    resolution : int, optional (default 20)
        Number of alpha levels for the calibration error computation.
    min_quantile, max_quantile : float
        Quantile range for calibration error (default 0.005–0.995).
    num_null_draws : int, optional (default 1000)
        Number of draws for the log-gamma null distribution.
    log_gamma_quantile : float, optional (default 0.05)
        Quantile of the null distribution used as the log-gamma threshold.

    Returns
    -------
    pd.DataFrame
        Rows = active parameters (labeled consistently with
        ``adaptive_posterior``), columns = metric names.
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

    batch_size, num_draws, num_rows, num_cols = pred.shape

    # Determine RMSE column name
    rmse_col = "RMSE" if normalize is None else "NRMSE"

    # Pre-compute shared quantities for calibration error
    alphas = np.linspace(min_quantile, max_quantile, resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    # Pre-compute null distribution for log gamma (shared across all parameters)
    null_dist = _gamma_null_distribution(batch_size, num_draws, num_null_draws)
    null_quantile = np.quantile(null_dist, log_gamma_quantile)

    records = {}

    for r in range(num_rows):
        # Build label (same convention as adaptive_posterior.py)
        if r == 0:
            row_suffix = None
        else:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            regressor_key = regressor_keys[regressor_id]
            row_suffix = f"${regressor_key}$ | $c_{{{category_id}}}$"

        for c in range(num_cols):
            if parameter_mask[r, c] != 1.0:
                continue

            param_label = (
                col_labels[c]
                if row_suffix is None
                else f"{col_labels[c]} -- {row_suffix}"
            )

            estimates = pred[:, :, r, c]  # (batch_size, num_draws)
            targets = true[:, r, c]       # (batch_size,)

            records[param_label] = {
                rmse_col: _rmse(estimates, targets, normalize, aggregation),
                "Posterior Contraction": _contraction(estimates, targets, aggregation),
                "Calibration Error": _calibration_error(
                    estimates, targets, alphas, lowers, uppers, aggregation
                ),
                "Log Gamma": _log_gamma(estimates, targets, null_quantile),
            }

    return pd.DataFrame.from_dict(records, orient="index")
