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


CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 0,
        "max_num_regressors": 0,
        "max_num_categories": 0,
        "fixed_config": False,
        "design_config": None,
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "param_names": [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
    },
    "fixed": {
        "free_intrinsics": ["v", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"],
        "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0,
        "max_num_regressors": 0,
        "max_num_categories": 0,
        "fixed_config": True,
        "design_config": None,
        "flatten_param_outputs": False,
        "squeeze_outputs": True,
        "param_names": [r"$v$", r"$a$", r"$\tau$"],
    },
    "regressed": {
        "free_intrinsics": ["v", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": True,
        "design_config": None,
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$",
            r"$u_{2, v}$", r"$u_{2, a}$",
        ],
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"],
        "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": True,
        "design_config": None,
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$a$", r"$\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$",
            r"$u_{2, v}$", r"$u_{2, a}$",
        ],
    },
    "interaction": {
        "free_intrinsics": ["v", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": False,
        "design_config": {
            "1": ["v", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "a", "tau", "s_v"],
            "u_2": ["v", "a", "tau"],
            "u_1:u_2": ["v", "a"],
        },
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$",
            r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, \tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, a}$",
        ],
    },
}


class DDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self, case: str):
        if case not in CASE_CONFIGS:
            raise ValueError(f"Unknown case: {case}. Must be one of {list(CASE_CONFIGS.keys())}")

        self.case = case
        self.config = CASE_CONFIGS[case]

        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_priors2(),
            regressed_params=self.config["regressed_params"],
            mask_randomizer_kwargs=dict(
                free_intrinsics=self.config["free_intrinsics"],
                fixed_intrinsics=self.config["fixed_intrinsics"],
                fixed_values=self.config["fixed_values"],
            )
        )

    def sample(self, batch_size, num_obs=500, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        sample_kwargs = {
            "min_num_regressors": self.config["min_num_regressors"],
            "max_num_regressors": self.config["max_num_regressors"],
            "max_num_categories": self.config["max_num_categories"],
        }

        if self.config["fixed_config"]:
            sample_kwargs["fixed_config"] = True

        if self.config["design_config"] is not None:
            sample_kwargs["design_config"] = self.config["design_config"]

        flatten_param_outputs = self.config["flatten_param_outputs"]

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

        if self.config["squeeze_outputs"]:
            param_matrices = param_matrices.squeeze(axis=1)
            param_masks = param_masks.squeeze(axis=1)

        return {
            "design_matrices": design_matrices,
            "rts": rts,
            "choices": choices,
            "params": param_matrices,
            "masks": param_masks,
        }


def test(case: str):
    ddm_family_simulator = DDMModelFamilyBF(case=case)

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
    print(f"Targets shape: {targets.shape}, Estimates shape: {estimates.shape}")

    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)
    true_params = targets[:, active_idx]
    pred_params = estimates[:, :, active_idx]
    print(f"Active true_params shape: {true_params.shape}, pred_params shape: {pred_params.shape}")


def main(case: str, batch_size: int = 200, num_samples: int = 200):
    config = CASE_CONFIGS[case]
    param_names = config["param_names"]

    ddm_family_simulator = DDMModelFamilyBF(case=case)

    checkpoint_path = f"./bayesgpt/experiments/checkpoints/ddm_families_bf_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)

    data_dir = Path("./bayesgpt/experiments/data")
    figures_dir = Path(f"./bayesgpt/experiments/figures/bf/{case}")
    evals_dir = Path("./bayesgpt/experiments/evaluations")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    val_sims = ddm_family_simulator.sample(batch_size)
    conditions = {
        "rts": val_sims["rts"],
        "choices": val_sims["choices"],
        "design_matrices": val_sims["design_matrices"],
    }
    targets = val_sims["params"]
    post_draws = approximator.sample(conditions=conditions, num_samples=num_samples)
    estimates = post_draws["params"]

    masks = val_sims["masks"]
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
            place_legend_below=True,
        )
        posterior_path = figures_dir / f"ddm_families_bf_{case}_posterior{i}.pdf"
        posterior.savefig(posterior_path)
        plt.close(posterior)
        logging.info(f"Saved posterior pairplot to {posterior_path}")

    rts = val_sims["rts"]
    choices = val_sims["choices"]
    design_matrices = val_sims["design_matrices"]
    param_masks = val_sims["masks"]

    true_set = val_sims["params"]
    pred_set = post_draws["params"]

    np.savez(
        data_dir / f"ddm_{case}_data.npz",
        rts=rts,
        choices=choices,
        true_set=true_set,
        pred_set=pred_set,
        design_matrices=design_matrices,
        param_masks=param_masks,
    )
    logging.info(f"Saved data to {data_dir}")

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

    coverage = bf.diagnostics.plots.coverage(
        estimates=pred_params,
        targets=true_params,
        variable_names=param_names,
        difference=True,
        figsize=(20, 16),
        label_fontsize=14,
        legend_fontsize=10,
        num_row=4,
        num_col=5,
        color=colors["intercept"],
    )
    coverage_path = figures_dir / f"ddm_families_bf_{case}_coverage.pdf"
    coverage.savefig(coverage_path)
    plt.close(coverage)
    logging.info(f"Saved coverage plot to {coverage_path}")

    calibration_ecdf = bf.diagnostics.calibration_ecdf(
        estimates=pred_params,
        targets=true_params,
        variable_names=param_names,
        difference=True,
        figsize=(20, 16),
        label_fontsize=14,
        legend_fontsize=10,
        num_row=4,
        num_col=5,
        rank_ecdf_color=colors["intercept"],
    )
    ecdf_path = figures_dir / f"ddm_families_bf_{case}_ecdf.pdf"
    calibration_ecdf.savefig(ecdf_path)
    plt.close(calibration_ecdf)
    logging.info(f"Saved ECDF plot to {ecdf_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDM Model Family Validation")
    parser.add_argument(
        "--case",
        type=str,
        default="intercept_only",
        choices=list(CASE_CONFIGS.keys()),
        help="Validation case to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of posterior samples",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode",
    )
    args = parser.parse_args()

    if args.test:
        test(case=args.case)
    else:
        main(case=args.case, batch_size=args.batch_size, num_samples=args.num_samples)

  # Structure:
  # - CASE_CONFIGS dictionary holds all configuration for the 5 cases: intercept_only, fixed, regressed, fixed_regressed, and interaction
  # - DDMModelFamilyBF class takes a case argument and configures itself accordingly
  # - test() and main() functions both accept a case parameter
  #
  # Usage:
  # # Run validation for a specific case
  # python ddm_family_bf_validate.py --case intercept_only
  # python ddm_family_bf_validate.py --case fixed
  # python ddm_family_bf_validate.py --case regressed
  # python ddm_family_bf_validate.py --case fixed_regressed
  # python ddm_family_bf_validate.py --case interaction
  #
  # # With custom batch size and number of samples
  # python ddm_family_bf_validate.py --case fixed --batch_size 100 --num_samples 100
  #
  # # Run in test mode
  # python ddm_family_bf_validate.py --case fixed --test
