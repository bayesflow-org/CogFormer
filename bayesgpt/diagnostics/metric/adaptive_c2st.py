import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from bayesgpt.simulators.context_manager import ContextManager


def compute_c2st_accuracy(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_splits: int = 5,
    hidden_layer_sizes: tuple = (32,),
    max_iter: int = 200,
    random_state: int = 0,
) -> float:
    """
    C2ST accuracy for two sets of 1D samples via stratified k-fold cross-validation.

    Trains a small MLP classifier to distinguish samples from two distributions.
    An accuracy of 0.5 indicates the distributions are indistinguishable; 1.0
    indicates perfect separability.

    Parameters
    ----------
    samples_a : np.ndarray of shape (n,)
        Samples from the first distribution (e.g. BayesGPT posterior).
    samples_b : np.ndarray of shape (m,)
        Samples from the second distribution (e.g. BayesFlow posterior).
    n_splits : int, optional, default: 5
        Number of stratified cross-validation folds.
    hidden_layer_sizes : tuple, optional, default: (32,)
        Hidden layer sizes for the MLP classifier.
    max_iter : int, optional, default: 200
        Maximum training iterations per fold.
    random_state : int, optional, default: 0
        Random seed for reproducibility.

    Returns
    -------
    float
        Mean cross-validated classification accuracy in [0.5, 1.0].
    """
    X = np.concatenate([samples_a, samples_b]).reshape(-1, 1)
    y = np.concatenate([np.zeros(len(samples_a)), np.ones(len(samples_b))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
        )
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[val_idx], y[val_idx]))

    return float(np.mean(scores))


def adaptive_c2st(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    n_splits: int = 5,
    hidden_layer_sizes: tuple = (32,),
    max_iter: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute C2ST accuracy for all active parameters in the adaptive grid.

    For each active cell (r, c), pools posterior samples across all datasets
    from both approximators and computes the C2ST accuracy.

    Parameters
    ----------
    pred_a : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
        Posterior samples from approximator A (e.g. BayesGPT).
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
    hidden_layer_sizes : tuple, optional, default: (32,)
        Hidden layer sizes for the MLP classifier.
    max_iter : int, optional, default: 200
        Maximum training iterations per fold.
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
                random_state=random_state,
            )
            records[param_label] = {"C2ST Accuracy": accuracy}

    return pd.DataFrame.from_dict(records, orient="index")
