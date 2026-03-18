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
from bayesgpt.simulators.benchmarks.cdms.cdm import CDM
from bayesgpt.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from bayesgpt.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from bayesgpt.diagnostics.plot.adaptive_recovery import adaptive_recovery
from bayesgpt.diagnostics.plot.adaptive_coverage import adaptive_coverage
from bayesgpt.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from bayesgpt.diagnostics.plot.adaptive_posterior import adaptive_posterior
from bayesgpt.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from bayesgpt.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from bayesgpt.utils.plot_utils import bf_colors


CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
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
        "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "expected_num_active": None,  # varies: random subset of 6 intrinsics
    },
    "fixed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
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
        "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$"],
        "expected_num_active": 4,  # v, v_theta, a, tau
    },
    "regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
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
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$",
            r"$u_{2, v}$", r"$u_{2, a}$",
        ],
        "expected_num_active": 10,  # 6 intercepts + 2 u_1 + 2 u_2
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
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
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$",
            r"$u_{2, v}$", r"$u_{2, a}$",
        ],
        "expected_num_active": 8,  # 4 intercepts (v, v_theta, a, tau) + 2 u_1 + 2 u_2
    },
    "interaction": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": False,
        "design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v"],
            "u_2": ["v", "v_theta", "a", "tau"],
            "u_1:u_2": ["v", "v_theta", "a"],
        },
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, v_\theta}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$",
            r"$u_{2, v}$", r"$u_{2, v_\theta}$", r"$u_{2, a}$", r"$u_{2, \tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_\theta}$", r"$u_1:u_{2, a}$",
        ],
        "expected_num_active": 18,  # 6 + 5 + 4 + 3 from design_config
    },
    "full": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": True,
        "design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        },
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, v_\theta}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{1, s_\tau}$",
            r"$u_{2, v}$", r"$u_{2, v_\theta}$", r"$u_{2, a}$", r"$u_{2, \tau}$", r"$u_{2, s_v}$", r"$u_{2, s_\tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_\theta}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, \tau}$", r"$u_1:u_{2, s_v}$", r"$u_1:u_{2, s_\tau}$",
        ],
        "expected_num_active": 24,  # 6 * 4 columns
    },
}


def get_benchmark_design_configs():
    free_params = ["v", "v_theta", "a", "tau"]
    fixed_params = ["s_v", "s_tau"]
    intrinsic_params = free_params + fixed_params

    intercept_only = {"1": intrinsic_params, "u_1": [], "u_2": [], "u_1:u_2": []}
    regressed = {"1": intrinsic_params, "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []}
    fixed = {"1": free_params, "u_1": [], "u_2": [], "u_1:u_2": []}
    fixed_regressed = {"1": free_params, "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []}
    interaction = {
        "1": intrinsic_params,
        "u_1": ["v", "v_theta", "a", "tau", "s_v"],
        "u_2": ["v", "v_theta", "a", "tau"],
        "u_1:u_2": ["v", "v_theta", "a"],
    }

    full = {
        "1": intrinsic_params,
        "u_1": intrinsic_params,
        "u_2": intrinsic_params,
        "u_1:u_2": intrinsic_params,
    }

    names = ["intercept_only", "regressed", "fixed", "fixed_regressed", "interaction", "full"]
    configs = [intercept_only, regressed, fixed, fixed_regressed, interaction, full]
    return {name: config for name, config in zip(names, configs)}


def reshape_bf_to_gpt(bf_samples, design_config, intrinsic_params):
    *leading, num_active = bf_samples.shape
    num_rows = len(design_config)
    num_cols = len(intrinsic_params)
    col_idx = {p: j for j, p in enumerate(intrinsic_params)}
    result = np.zeros((*leading, num_rows, num_cols))
    flat_pos = 0
    for row_i, active_params in enumerate(design_config.values()):
        ordered = [p for p in intrinsic_params if p in active_params]
        for p in ordered:
            result[..., row_i, col_idx[p]] = bf_samples[..., flat_pos]
            flat_pos += 1
    assert flat_pos == num_active, (
        f"Expected {num_active} active params but mapped {flat_pos}."
    )
    return result


class CDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self, case: str):
        if case not in CASE_CONFIGS:
            raise ValueError(f"Unknown case: {case}. Must be one of {list(CASE_CONFIGS.keys())}")

        self.case = case
        self.config = CASE_CONFIGS[case]

        self.model_family = NestedModelFamily(
            model=CDM(),
            prior_fun=cdm_priors(),
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
            link_fun=cdm_link_fun(),
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
    config = CASE_CONFIGS[case]
    cdm_family_simulator = CDMModelFamilyBF(case=case)

    # --- Simulator checks (before loading checkpoint) ---
    samples = cdm_family_simulator.sample(4)
    masks = samples["masks"]
    params = samples["params"]

    num_active = int(masks[0].astype(bool).sum())
    expected = config["expected_num_active"]
    if expected is not None:
        status = "OK" if num_active == expected else f"FAIL (expected {expected})"
        print(f"  [Check] Active params: {num_active}  →  {status}")
        assert num_active == expected, f"Active param count: got {num_active}, expected {expected}"
    else:
        print(f"  [Check] Active params: {num_active}  →  (variable, no assertion)")

    if config["fixed_config"] or config["design_config"] is not None:
        consistent = np.all(masks == masks[0])
        print(f"  [Check] Mask consistent across batch: {'OK' if consistent else 'FAIL'}")
        assert consistent, "Mask differs across batch items for a fixed-config case"

    zero_where_masked = np.allclose(params * (1 - masks), 0.0)
    print(f"  [Check] Params zero where masked: {'OK' if zero_where_masked else 'FAIL'}")
    assert zero_where_masked, "Non-zero param values found at masked-out positions"

    # --- Approximator checks ---
    checkpoint_path = f"./bayesgpt/experiments/checkpoints/cdm_families_bf_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)
    print("Loaded model")

    val_sims = cdm_family_simulator.sample(100)
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


def main(case: str, batch_size: int = 200, num_samples: int = 200, skip_posteriors: bool = False, skip_log_gamma: bool = True):
    config = CASE_CONFIGS[case]
    param_names = config["param_names"]

    cdm_family_simulator = CDMModelFamilyBF(case=case)

    checkpoint_path = f"./bayesgpt/experiments/checkpoints/cdm_families_bf_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)

    data_dir = Path("./bayesgpt/experiments/data")
    figures_dir = Path(f"./bayesgpt/experiments/figures/bf/cdm/{case}")
    evals_dir = Path("./bayesgpt/experiments/evaluations")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    val_sims = cdm_family_simulator.sample(batch_size)
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
    if len(param_names) == len(active_idx):
        param_names = [name for name, active in zip(param_names, active_idx) if active]

    rts = val_sims["rts"]
    choices = val_sims["choices"]
    design_matrices = val_sims["design_matrices"]
    param_masks = val_sims["masks"]

    true_set = val_sims["params"]
    pred_set = post_draws["params"]

    np.savez(
        data_dir / f"cdm_{case}_data.npz",
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
    calibration_errors = bf.diagnostics.metrics.calibration_error(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )
    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )

    metrics_dict = {
        rmse["metric_name"]: rmse["values"],
        calibration_errors["metric_name"]: calibration_errors["values"],
        contraction["metric_name"]: contraction["values"],
    }
    if not skip_log_gamma:
        log_gamma = bf.diagnostics.metrics.calibration_log_gamma(
            estimates=pred_params, targets=true_params, variable_names=param_names
        )
        metrics_dict[log_gamma["metric_name"]] = log_gamma["values"]

    metrics = pd.DataFrame(metrics_dict)
    metrics.to_csv(evals_dir / f"cdm_families_bf_{case}_evaluations.csv", sep=";")
    logging.info("Metric evaluation is now finished.")

    # --- Adaptive diagnostics ---
    intrinsic_params_all = ["v", "v_theta", "a", "tau", "s_v", "s_tau"]
    variable_names_all = [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]
    design_config = get_benchmark_design_configs()[case]
    adaptive_colors = bf_colors()

    true_grid = reshape_bf_to_gpt(true_params, design_config, intrinsic_params_all)
    pred_grid = reshape_bf_to_gpt(pred_params, design_config, intrinsic_params_all)
    params_mask = reshape_bf_to_gpt(
        np.ones((1, true_params.shape[-1])), design_config, intrinsic_params_all
    )[0]

    recovery_fig = adaptive_recovery(
        true=true_grid,
        pred=pred_grid,
        design_config=design_config,
        intrinsic_params=intrinsic_params_all,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"],
        main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
    )
    recovery_fig.savefig(figures_dir / f"cdm_family_{case}_bf_recovery.pdf", bbox_inches="tight")
    plt.close(recovery_fig)
    logging.info(f"Saved adaptive recovery to {figures_dir}")

    coverage_fig = adaptive_coverage(
        true=true_grid,
        pred=pred_grid,
        design_config=design_config,
        intrinsic_params=intrinsic_params_all,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"],
        main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
    )
    coverage_fig.savefig(figures_dir / f"cdm_family_{case}_bf_coverage.pdf", bbox_inches="tight")
    plt.close(coverage_fig)
    logging.info(f"Saved adaptive coverage to {figures_dir}")

    ecdf_fig = adaptive_ecdf(
        true=true_grid,
        pred=pred_grid,
        design_config=design_config,
        intrinsic_params=intrinsic_params_all,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"],
        main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
        difference=True,
    )
    ecdf_fig.savefig(figures_dir / f"cdm_family_{case}_bf_ecdf.pdf", bbox_inches="tight")
    plt.close(ecdf_fig)
    logging.info(f"Saved adaptive ECDF to {figures_dir}")

    metrics_fig = plot_adaptive_metrics(
        true=true_grid,
        pred=pred_grid,
        design_config=design_config,
        intrinsic_params=intrinsic_params_all,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"],
        main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
        skip_log_gamma=skip_log_gamma,
    )
    metrics_fig.savefig(figures_dir / f"cdm_family_{case}_bf_metrics.pdf", bbox_inches="tight")
    plt.close(metrics_fig)
    logging.info(f"Saved adaptive metrics to {figures_dir}")

    metrics_df = compute_adaptive_metrics(
        true=true_grid,
        pred=pred_grid,
        design_config=design_config,
        intrinsic_params=intrinsic_params_all,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=variable_names_all,
        skip_log_gamma=skip_log_gamma,
    )
    metrics_df.to_csv(figures_dir / f"cdm_family_{case}_bf_metrics.csv")
    logging.info(f"Saved adaptive metrics CSV to {figures_dir}")

    if not skip_posteriors:
        for i in range(10):
            posterior_fig = adaptive_posterior(
                samples=pred_grid[i],
                design_config=design_config,
                intrinsic_params=intrinsic_params_all,
                max_num_categories=2,
                unfold=False,
                intercept_color=adaptive_colors["intercept"],
                main_effect_color=adaptive_colors["main_effect"],
                interaction_color=adaptive_colors["interaction"],
            )
            posterior_fig.savefig(
                figures_dir / f"cdm_family_{case}_bf_posterior_{i}.pdf", bbox_inches="tight"
            )
            plt.close(posterior_fig.fig)
        logging.info(f"Saved adaptive posteriors to {figures_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CDM Model Family Validation")
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
    parser.add_argument(
        "--skip_posteriors",
        action="store_true",
        help="Skip posterior pairplots",
    )
    parser.add_argument(
        "--skip_log_gamma",
        action="store_true",
        default=True,
        help="Skip log gamma metric (slow, skipped by default)",
    )
    parser.add_argument(
        "--include_log_gamma",
        dest="skip_log_gamma",
        action="store_false",
        help="Include log gamma metric",
    )
    args = parser.parse_args()

    if args.test:
        test(case=args.case)
    else:
        main(case=args.case, batch_size=args.batch_size, num_samples=args.num_samples, skip_posteriors=args.skip_posteriors, skip_log_gamma=args.skip_log_gamma)