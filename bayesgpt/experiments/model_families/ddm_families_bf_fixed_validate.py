import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import logging
import numpy as np
import pandas as pd
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from bayesgpt.simulators.model_family import NestedModelFamily
from bayesgpt.simulators.benchmarks.ddms.ddm import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors2
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from bayesgpt.utils.plot_utils import bf_colors


class DDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self):
        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_priors2(),
            mask_randomizer_kwargs=dict(
                free_intrinsics=["v", "a", "tau"],
                fixed_intrinsics=["s_v", "s_tau"],
                fixed_values={"s_v": 0, "s_tau": 0},
            )
    )

    def sample(self, batch_size, num_obs=500, flatten_param_outputs=False, **kwargs):

        if isinstance(batch_size, tuple):
             batch_size = batch_size[0]

        sample_kwargs = {
            'min_num_regressors': 0,
            "max_num_regressors": 0,
            "max_num_categories": 0,
            "fixed_config": True
        }

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=flatten_param_outputs,
            link_fun=ddm_link_fun(),
            **sample_kwargs,
            **kwargs
        )

        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        param_matrices = samples["param_matrices"]
        mask = samples["param_masks"]
        active_idx = mask[0].astype(bool)
        params = param_matrices[:, active_idx]

        # Special treatment for BF:
        # Trim away zeros from non-regressed params
        params = params[:, params[0] != 0]

        return {"rts": rts, "choices": choices, "params": params}

def main(num_samples=200, case="fixed"):
    # Define simulator
    ddm_family_simulator = DDMModelFamilyBF()

    # define checkpoint filepath
    checkpoint_path = f"./bayesgpt/experiments/checkpoints/ddm_families_bf_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)

    # Make directories
    param_names = [
        r"$v$", r"$a$", r"$\tau$"
    ]

    data_dir = Path("./bayesgpt/experiments/data")
    figures_dir = Path(f"./bayesgpt/experiments/figures/bf/{case}")
    evals_dir = Path("./bayesgpt/experiments/evaluations")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    # Generate validation samples
    val_sims = ddm_family_simulator.sample(num_samples)
    post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

    # Save some of them
    rts = val_sims["rts"][:10]
    choices = val_sims["choices"][:10]
    params = post_draws["params"][:10]

    np.savez(
        data_dir / f"ddm_families_bf_{case}_data.npz",
        rts=rts, choices=choices, params=params
    )
    logging.info(f"Saved data to {data_dir}")

    # Compute and save metric evaluations
    rmse = bf.diagnostics.metrics.root_mean_squared_error(
        estimates=post_draws, targets=val_sims, variable_names=param_names
    )

    log_gamma = bf.diagnostics.metrics.calibration_log_gamma(
        estimates=post_draws, targets=val_sims, variable_names=param_names
    )

    calibration_errors = bf.diagnostics.metrics.calibration_error(
        estimates=post_draws, targets=val_sims, variable_names=param_names
    )

    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=post_draws, targets=val_sims, variable_names=param_names
    )

    metrics = pd.DataFrame(
        {
            rmse["metric_name"]: rmse["values"],
            log_gamma["metric_name"]: log_gamma["values"],
            calibration_errors["metric_name"]: calibration_errors["values"],
            contraction["metric_name"]: contraction["values"],
        }
    )

    metrics.to_csv(evals_dir / f"ddm_families_bf_{case}_evaluations.csv", sep=";")
    logging.info("Metric evaluation is now finished.")

    recovery = bf.diagnostics.recovery(
        estimates=post_draws,
        targets=val_sims,
        variable_names=param_names,
        figsize=(9, 3),
        label_fontsize=14,
        num_row=1,
        num_col=3
    )
    recovery_path = figures_dir / f"ddm_families_bf_{case}_recovery.pdf"
    recovery.savefig(recovery_path)
    plt.close(recovery)
    logging.info(f"Saved recovery plot to {recovery_path}")

    for i in range(10):
        posterior = bf.diagnostics.plots.pairs_posterior(
            estimates=post_draws,
            targets=val_sims,
            priors=val_sims,
            dataset_id=i,
            variable_names=param_names
        )
        posterior_path = figures_dir / f"ddm_families_bf_{case}_posterior{i}.pdf"
        posterior.savefig(posterior_path)
        logging.info(f"Saved posterior pairplot to {posterior_path}")

if __name__ == "__main__":
    main()
