import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import logging
import numpy as np
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from simulators.model_family import NestedModelFamily
from simulators.benchmarks.ddms.ddm import DDM
from simulators.benchmarks.ddms.ddm_priors import ddm_baseline_priors


class DDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self):
        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_baseline_priors(),
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
            **sample_kwargs,
            **kwargs
        )

        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        params = samples["param_matrices"]

        # Special treatment for BF:
        # Trim away zeros from non-regressed params
        params = params[:, params[0] != 0]

        return {"rts": rts, "choices": choices, "params": params}

def main():
    # Define simulator
    ddm_family_simulator = DDMModelFamilyBF()

    # define checkpoint filepath
    checkpoint_path = "./experiments/checkpoints/ddm_families_bf_fixed/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)

    param_names = [r"$v$", r"$a$", r"$\tau$"]
    figures_dir = Path("./experiments/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("./experiments/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    val_sims = ddm_family_simulator.sample(300)
    post_draws = approximator.sample(conditions=val_sims, num_samples=300)

    rts = val_sims["rts"][:10]
    choices = val_sims["choices"][:10]
    params = post_draws["params"][:10]

    np.savez(
        data_dir / "ddm_families_bf_fixed_data.npz",
        rts=rts, choices=choices, params=params
    )
    logging.info(f"Saved data to {data_dir}")

    recovery = bf.diagnostics.recovery(
        estimates=post_draws,
        targets=val_sims,
        variable_names=param_names,
        figsize=(3 * len(param_names), 3),
        label_fontsize=14
    )
    recovery_path = figures_dir / "ddm_families_bf_fixed_recovery.pdf"
    recovery.savefig(recovery_path)
    plt.close(recovery)
    logging.info(f"Saved recovery plot to {recovery_path}")

    posterior = bf.diagnostics.plots.pairs_posterior(
        estimates = post_draws,
        targets = val_sims,
        dataset_id = 0,
        variable_names = param_names
    )

    posterior_path = figures_dir / "ddm_families_bf_fixed_posterior.pdf"
    posterior.savefig(posterior_path)
    logging.info(f"Saved posterior pairplot to {posterior_path}")

if __name__ == "__main__":
    main()
