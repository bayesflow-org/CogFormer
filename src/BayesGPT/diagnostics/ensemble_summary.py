import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_ensemble_summary(results, keys=None, max_lines=10):
    """
    Grid-based summary of simulator outputs using Seaborn and matplotlib.

    Simulator types are arranged in rows, and output keys in columns.
    Scalars are plotted as histograms; sequences as overlayed line plots.
    Cells are left blank when a simulator does not return that key.

    Parameters
    ----------
    results : dict
        Dictionary returned by `.run()` method of an EnsembleSimulator.
    keys : list of str, optional
        Keys to plot. If None, collects all unique keys across simulators.
    max_lines : int
        Max number of sequences to overlay in sequence-type plots.
    """
    simulators = list(results.keys())
    if not simulators:
        return

    # Collect all unique keys if not specified
    if keys is None:
        all_keys = set()
        for sim in simulators:
            for out in results[sim]:
                all_keys.update(out.keys())
        keys = sorted(all_keys)

    keys = keys or []
    nrows, ncols = len(simulators), len(keys)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for row, model_name in enumerate(simulators):
        outputs = results[model_name]
        if not outputs:
            for col in range(ncols):
                axes[row][col].set_visible(False)
            continue

        for col, key in enumerate(keys):
            ax = axes[row][col]
            values = [o[key] for o in outputs if key in o]

            if not values:
                ax.set_visible(False)
                continue

            # Scalar values (e.g., RT, choice)
            if all(isinstance(v, (int, float, np.number)) for v in values):
                df = pd.DataFrame({key: values})
                sns.histplot(df, x=key, bins=30, kde=True, ax=ax, element="step")

            # Sequence values (e.g., trajectory)
            elif all(isinstance(v, (list, np.ndarray)) for v in values):
                for traj in values[:max_lines]:
                    ax.plot(traj, alpha=0.6)
                ax.set_xlabel("Step")

            # Unsupported
            else:
                ax.set_visible(False)
                continue

            sns.despine(ax=ax)

            # Add labels only on leftmost and bottom plots
            if col == 0:
                ax.set_ylabel(model_name)
            else:
                ax.set_ylabel("")

            if row == nrows - 1:
                ax.set_xlabel(key)
            else:
                ax.set_xlabel("")

            ax.set_title("")

    plt.tight_layout()
    plt.show()
