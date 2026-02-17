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
from bayesgpt.diagnostics.plot.adaptive_recovery import adaptive_recovery
from bayesgpt.diagnostics.plot.adaptive_posterior import adaptive_posterior
from bayesgpt.utils.plot_utils import bf_colors


class DDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self):
        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_priors2(),
            mask_randomizer_kwargs=dict(
                free_intrinsics=["v", "a", "tau", "s_v", "s_tau"],
                fixed_intrinsics=[],
                fixed_values={}
            )
    )

    def sample(self, batch_size, num_obs=500, flatten_param_outputs=True, **kwargs):
        intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]

        design_config = {
            "1": intrinsic_params,
            "u_1": ["v", "a", "tau", "s_v"],
            "u_2": ["v", "a", "tau"],
            "u_1:u_2": ["v", "a"]
        }

        if isinstance(batch_size, tuple):
             batch_size = batch_size[0]

        sample_kwargs = {
            "design_config": design_config,
            "max_num_regressors": 2,
            "max_num_categories": 2,
        }

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=flatten_param_outputs,
            link_fun=ddm_link_fun(),
            **sample_kwargs,
            **kwargs
        )

        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        param_matrices = samples["param_matrices"]
        param_masks = samples["param_masks"]

        return {
            "design_matrices": design_matrices,
            "rts": rts,
            "choices": choices,
            "params": param_matrices,
            "masks": param_masks
        }


def test(case="interaction"):
    ddm_family_simulator = DDMModelFamilyBF()

    # define checkpoint filepath
    checkpoint_path = f"./bayesgpt/experiments/checkpoints/ddm_families_bf_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)
    print("Loaded model")

    val_sims = ddm_family_simulator.sample(100)
    conditions = {
        "rts": val_sims["rts"],
        "choices": val_sims["choices"],
        "design_matrices": val_sims["design_matrices"],
    }
    targets = val_sims["params"]
    post_draws = approximator.sample(conditions=conditions, num_samples=100)
    estimates = post_draws["params"]
    print(targets.shape, estimates.shape)

    masks = val_sims['masks']
    active_idx = masks[0].astype(bool)
    true_params = targets[:, active_idx]
    pred_params = estimates[:, :, active_idx]
    print(true_params.shape, pred_params.shape)

def main(num_samples=200, case="interaction", data_path=None):
    # Define simulator
    ddm_family_simulator = DDMModelFamilyBF()

    # define checkpoint filepath
    checkpoint_path = f"./bayesgpt/experiments/checkpoints/ddm_families_bf_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)

    # Make directories
    param_names = [
        r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
        r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$",
        r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, \tau}$",
        r"$u_1:u_{2, v}$", r"$u_1:u_{2, a}$",
    ]

    data_dir = Path("./bayesgpt/experiments/data")
    figures_dir = Path(f"./bayesgpt/experiments/figures/bf/{case}")
    evals_dir = Path("./bayesgpt/experiments/evaluations")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    # Generate validation samples
    val_sims = ddm_family_simulator.sample(num_samples)
    conditions = {
        "rts": val_sims["rts"],
        "choices": val_sims["choices"],
        "design_matrices": val_sims["design_matrices"]
    }
    targets = val_sims["params"]
    post_draws = approximator.sample(conditions=conditions, num_samples=num_samples)
    estimates = post_draws["params"]

    masks = val_sims['masks']
    active_idx = masks[0].astype(bool)
    true_params = targets[:, active_idx]
    pred_params = estimates[:, :, active_idx]

    colors = bf_colors()

    for i in range(10):
        posterior = bf.diagnostics.plots.pairs_posterior(
            estimates=pred_params,
            targets=true_params,
            dataset_id=i,
            variable_names=param_names,
            post_color=colors["intercept"],
            place_legend_below=True
        )
        posterior_path = figures_dir / f"ddm_families_bf_{case}_posterior{i}.pdf"
        posterior.savefig(posterior_path)
        logging.info(f"Saved posterior pairplot to {posterior_path}")

    # Save some of them
    rts = val_sims["rts"][:10]
    choices = val_sims["choices"][:10]
    design_matrices = val_sims["design_matrices"][:10]
    param_masks = val_sims["masks"][:10]
    params = post_draws["params"][:10]

    np.savez(
        data_dir / f"ddm_families_bf_{case}_data.npz",
        rts=rts, choices=choices, params=params, design_matrices=design_matrices, param_masks=param_masks
    )
    logging.info(f"Saved data to {data_dir}")

    # Compute and save metric evaluations
    rmse = bf.diagnostics.metrics.root_mean_squared_error(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )

    log_gamma = bf.diagnostics.metrics.calibration_log_gamma(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )

    calibration_errors = bf.diagnostics.metrics.calibration_error(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )

    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=pred_params, targets=true_params, variable_names=param_names
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
        estimates=pred_params,
        targets=true_params,
        variable_names=param_names,
        figsize=(15, 12),
        label_fontsize=14,
        num_row=4,
        num_col=5,
        color=colors["intercept"],
    )
    recovery_path = figures_dir / f"ddm_families_bf_{case}_recovery.pdf"
    recovery.savefig(recovery_path)
    plt.close(recovery)
    logging.info(f"Saved recovery plot to {recovery_path}")


if __name__ == '__main__':
    debug = False
    if debug:
        test()
    else:
        main()
