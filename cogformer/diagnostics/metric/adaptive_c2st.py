import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from cogformer.simulators.context_manager import ContextManager


def compute_c2st_accuracy(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_splits: int = 5,
    hidden_layer_sizes: tuple | str = "auto",
    max_iter: int = 1000,
    patience: int = 5,
    random_state: int = 0,
    standardize: bool = True,
) -> float:
    """
    C2ST accuracy for two sets of samples via stratified k-fold cross-validation.

    Trains a small MLP classifier to distinguish samples from two distributions.
    An accuracy of 0.5 indicates the distributions are indistinguishable; 1.0
    indicates perfect separability.

    Parameters
    ----------
    samples_a : np.ndarray of shape (n,) or (n, d)
        Samples from the first distribution (e.g. CogFormer posterior).
    samples_b : np.ndarray of shape (m,) or (m, d)
        Samples from the second distribution (e.g. BayesFlow posterior).
    n_splits : int, optional, default: 5
        Number of stratified cross-validation folds.
    hidden_layer_sizes : tuple or "auto", optional, default: "auto"
        Hidden layer sizes for the MLP classifier. If "auto", sets two hidden
        layers of width ``2 ** ceil(log2(10 * num_dims))``.
    max_iter : int, optional, default: 1000
        Maximum training iterations per fold.
    patience : int, optional, default: 5
        Early-stopping patience (number of iterations with no improvement).
    random_state : int, optional, default: 0
        Random seed for reproducibility.
    standardize : bool, optional, default: True
        If True, standardize both sets using samples_a mean and std.

    Returns
    -------
    float
        Mean cross-validated classification accuracy.
    """
    if samples_a.ndim == 1:
        samples_a = samples_a[:, None]
    if samples_b.ndim == 1:
        samples_b = samples_b[:, None]

    num_dims = samples_a.shape[1]

    if hidden_layer_sizes == "auto":
        width = 2 ** int(np.ceil(np.log2(10 * num_dims)))
        hidden_layer_sizes = (width, width)

    if standardize:
        mean = np.mean(samples_a, axis=0)
        std = np.std(samples_a, axis=0) + 1e-8
        samples_a = (samples_a - mean) / std
        samples_b = (samples_b - mean) / std

    X = np.concatenate([samples_a, samples_b], axis=0)
    y = np.concatenate([np.zeros(len(samples_a)), np.ones(len(samples_b))])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=patience,
            validation_fraction=1.0 / n_splits,
            random_state=random_state,
        )
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[val_idx], y[val_idx]))

    return float(np.mean(scores))


def compute_joint_c2st(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    n_splits: int = 5,
    hidden_layer_sizes: tuple | str = "auto",
    max_iter: int = 1000,
    patience: int = 5,
    random_state: int = 0,
) -> float:
    """
    Global joint C2ST accuracy over all active parameters simultaneously.

    Pools posterior samples from all datasets and all active parameter dimensions
    into a single classifier, testing whether the full joint posterior matches
    between the two approximators.

    Parameters
    ----------
    pred_a : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
    pred_b : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
    design_config : dict
    intrinsic_params : list[str]
    max_num_categories : int
    parameter_mask : np.ndarray of shape (num_rows, num_cols), optional
    n_splits : int, optional, default: 5
    hidden_layer_sizes : tuple or "auto", optional, default: "auto"
    max_iter : int, optional, default: 1000
    patience : int, optional, default: 5
    random_state : int, optional, default: 0

    Returns
    -------
    float
        Single C2ST accuracy value for the joint distribution over all active parameters.
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

    active = parameter_mask == 1.0
    num_active = int(active.sum())

    # (batch, draws, num_active) → (batch*draws, num_active)
    samples_a = pred_a[:, :, active].reshape(-1, num_active)
    samples_b = pred_b[:, :, active].reshape(-1, num_active)

    return compute_c2st_accuracy(
        samples_a=samples_a,
        samples_b=samples_b,
        n_splits=n_splits,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        patience=patience,
        random_state=random_state,
    )


def adaptive_c2st(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    n_splits: int = 5,
    hidden_layer_sizes: tuple | str = "auto",
    max_iter: int = 1000,
    patience: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute C2ST accuracy for all active parameters in the adaptive grid.

    For each active cell (r, c), pools posterior samples across all datasets
    from both approximators and computes the C2ST accuracy.

    Parameters
    ----------
    pred_a : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
        Posterior samples from approximator A (e.g. CogFormer).
    pred_b : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
        Posterior samples from approximator B (e.g. BayesFlow).
    design_config : dict
        Mapping from regressor keys to lists of active parameter names.
    intrinsic_params : list[str]
        Ordered list of intrinsic parameter names (column order).
    max_num_categories : int
        Maximum number of categories, determines the row layout.
    parameter_mask : np.ndarray of shape (num_rows, num_cols), optional
        Binary mask (1 = active). Built from ContextManager if not provided.
    variable_names : list[str], optional
        Display labels for columns. Falls back to intrinsic_params if not given.
    n_splits : int, optional, default: 5
        Number of stratified cross-validation folds.
    hidden_layer_sizes : tuple or "auto", optional, default: "auto"
        Hidden layer sizes for the MLP classifier.
    max_iter : int, optional, default: 1000
        Maximum training iterations per fold.
    patience : int, optional, default: 5
        Early-stopping patience.
    random_state : int, optional, default: 0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Rows = active parameters, column = "C2ST Accuracy".
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
    _, _, num_rows, num_cols = pred_a.shape

    records = {}

    for r in range(num_rows):
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

            accuracy = compute_c2st_accuracy(
                samples_a=pred_a[:, :, r, c].ravel(),
                samples_b=pred_b[:, :, r, c].ravel(),
                n_splits=n_splits,
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                patience=patience,
                random_state=random_state,
            )
            records[param_label] = {"C2ST Accuracy": accuracy}

    return pd.DataFrame.from_dict(records, orient="index")
