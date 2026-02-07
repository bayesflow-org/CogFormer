import os
os.environ["KERAS_BACKEND"] = "torch"

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


class DDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self):
        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_priors2(),
            regressed_params=["v", "a"],
            mask_randomizer_kwargs=dict(
                free_intrinsics=["v", "a", "tau", "s_v", "s_tau"],
                fixed_intrinsics=[],
                fixed_values={}
            )
    )

    def sample(self, batch_size, num_obs=500, flatten_param_outputs=True, check=False, **kwargs):
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

        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        params = samples["param_matrices"]

        params = params[:, params[0] != 0]

        # Special treatment for BF:
        # Trim away zeros from non-regressed params
        if check:
            f = self.model_family.check_regressed_priors(
                design_config=design_config,
                num_obs=num_obs,
                link_fun=ddm_link_fun()
            )

        return {"rts": rts, "choices": choices, "params": params}

def main(batch_size=200, num_samples=200, case="interaction"):
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
    figures_dir = Path("./bayesgpt/experiments/figures")
    evals_dir = Path("./bayesgpt/experiments/evaluations")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    # Generate validation samples
    val_sims = ddm_family_simulator.sample(batch_size=batch_size)
    post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

    # Save some of them
    rts = val_sims["rts"][:15]
    choices = val_sims["choices"][:15]
    params = post_draws["params"][:15]

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
        figsize=(15, 9),
        label_fontsize=14,
        num_row=3,
        num_col=5
    )
    recovery_path = figures_dir / f"ddm_families_bf_{case}_recovery.pdf"
    recovery.savefig(recovery_path)
    plt.close(recovery)
    logging.info(f"Saved recovery plot to {recovery_path}")

    for i in range(5):
        posterior = bf.diagnostics.plots.pairs_posterior(
            estimates=post_draws,
            targets=val_sims,
            dataset_id=i,
            variable_names=param_names
        )
        posterior_path = figures_dir / f"ddm_families_bf_{case}_posterior_{i}.pdf"
        posterior.savefig(posterior_path)
        logging.info(f"Saved posterior pairplot {i} to {posterior_path}")


if __name__ == '__main__':
    main()
