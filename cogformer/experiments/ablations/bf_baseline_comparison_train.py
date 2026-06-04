import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import pandas as pd
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from cogformer.simulators.model_family import NestedModelFamily
from cogformer.simulators.benchmarks.ddms.ddm import DDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import bf_colors


INTRINSIC_PARAMS = ["v", "a", "z", "tau", "s_v", "s_tau"]

DESIGN_CONFIG = {
    "1":       ["v", "a", "z", "tau", "s_v", "s_tau"],
    "u_1":     ["v", "a", "z", "tau", "s_v"],
    "u_2":     ["v", "a", "z", "tau"],
    "u_1:u_2": ["v", "a", "z"],
}

PARAM_NAMES = [
    r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$",
    r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$",
    r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$", r"$u_{2, \tau}$",
    r"$u_1:u_{2, v}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, z}$",
]

NETWORK_MAP = {
    "coupling_flow":    bf.networks.CouplingFlow,
    "diffusion_model":  bf.networks.DiffusionModel,
    "flow_matching":    bf.networks.FlowMatching,
    "stable_consistency": bf.networks.StableConsistencyModel,
}


class DDMInteractionSimulator(bf.simulators.Simulator):

    def __init__(self):
        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_priors(),
            mask_randomizer_kwargs=dict(
                free_intrinsics=INTRINSIC_PARAMS,
                fixed_intrinsics=[],
                fixed_values={},
            ),
        )

    def sample(self, batch_size, num_obs=500, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            min_num_regressors=2,
            max_num_regressors=2,
            max_num_categories=2,
            flatten_param_outputs=True,
            design_config=DESIGN_CONFIG,
            link_fun=ddm_link_fun(),
        )

        return {
            "design_matrices": samples["design_matrices"],
            "rts":             samples["sim_data"]["rts"],
            "choices":         samples["sim_data"]["choices"],
            "params":          samples["param_matrices"],
            "masks":           samples["param_masks"],
        }


def reshape_to_grid(flat_samples, design_config, intrinsic_params):
    *leading, num_active = flat_samples.shape
    num_rows = len(design_config)
    num_cols = len(intrinsic_params)
    col_idx = {p: j for j, p in enumerate(intrinsic_params)}
    result = np.zeros((*leading, num_rows, num_cols))
    flat_pos = 0
    for row_i, active_params in enumerate(design_config.values()):
        ordered = [p for p in intrinsic_params if p in active_params]
        for p in ordered:
            result[..., row_i, col_idx[p]] = flat_samples[..., flat_pos]
            flat_pos += 1
    assert flat_pos == num_active
    return result


def train(network_name: str, epochs: int, steps_per_epoch: int, batch_size: int):
    simulator = DDMInteractionSimulator()

    adapter = (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )

    summary_net = bf.networks.SetTransformer(
        summary_dim=32,
        seed_dim=64,
        num_heads=(4, 4, 4, 4),
        mlp_depths=(1, 1, 1, 1),
        embed_dims=(128, 128, 128, 128),
        mlp_widths=(128, 128, 128, 128),
        num_seeds=4,
    )
    inference_net = NETWORK_MAP[network_name]()

    checkpoint_path = f"./cogformer/experiments/checkpoints/ddm_baseline_{network_name}"

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=checkpoint_path,
    )

    workflow.fit_online(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
    )


def evaluate(network_name: str, batch_size: int, num_samples: int):
    simulator = DDMInteractionSimulator()

    checkpoint_path = f"./cogformer/experiments/checkpoints/ddm_baseline_{network_name}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)

    data_dir = Path("./cogformer/experiments/ablations/baselines_data")
    figures_dir = Path(f"./cogformer/experiments/ablations/baselines_figures/{network_name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    val_sims = simulator.sample(batch_size)
    conditions = {
        "rts":             val_sims["rts"],
        "choices":         val_sims["choices"],
        "design_matrices": val_sims["design_matrices"],
    }
    post_draws = approximator.sample(conditions=conditions, num_samples=num_samples)

    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)
    true_params = val_sims["params"][:, active_idx]
    pred_params = post_draws["params"][:, :, active_idx]
    active_names = [n for n, a in zip(PARAM_NAMES, active_idx) if a]

    rmse = bf.diagnostics.metrics.root_mean_squared_error(
        estimates=pred_params, targets=true_params, variable_names=active_names
    )
    cal_error = bf.diagnostics.metrics.calibration_error(
        estimates=pred_params, targets=true_params, variable_names=active_names
    )
    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=pred_params, targets=true_params, variable_names=active_names
    )

    metrics_df = pd.DataFrame({
        rmse["metric_name"]:      rmse["values"],
        cal_error["metric_name"]: cal_error["values"],
        contraction["metric_name"]: contraction["values"],
    })
    metrics_df.to_csv(data_dir / f"ddm_interaction_{network_name}_metrics.csv", sep=";")

    np.savez(
        data_dir / f"ddm_interaction_{network_name}_pred.npz",
        true_params=val_sims["params"],
        pred_params=post_draws["params"],
        masks=masks,
    )

    colors = bf_colors()
    params_mask = reshape_to_grid(
        np.ones((1, true_params.shape[-1])), DESIGN_CONFIG, INTRINSIC_PARAMS
    )[0]
    true_grid = reshape_to_grid(true_params, DESIGN_CONFIG, INTRINSIC_PARAMS)
    pred_grid = reshape_to_grid(pred_params, DESIGN_CONFIG, INTRINSIC_PARAMS)

    recovery_fig = adaptive_recovery(
        true=true_grid,
        pred=pred_grid,
        design_config=DESIGN_CONFIG,
        intrinsic_params=INTRINSIC_PARAMS,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=[r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
    )
    recovery_fig.savefig(figures_dir / "ddm_interaction_recovery.pdf", bbox_inches="tight")
    plt.close(recovery_fig)

    coverage_fig = adaptive_coverage(
        true=true_grid,
        pred=pred_grid,
        design_config=DESIGN_CONFIG,
        intrinsic_params=INTRINSIC_PARAMS,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=[r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
    )
    coverage_fig.savefig(figures_dir / "ddm_interaction_coverage.pdf", bbox_inches="tight")
    plt.close(coverage_fig)

    ecdf_fig = adaptive_ecdf(
        true=true_grid,
        pred=pred_grid,
        design_config=DESIGN_CONFIG,
        intrinsic_params=INTRINSIC_PARAMS,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=[r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
        difference=True,
    )
    ecdf_fig.savefig(figures_dir / "ddm_interaction_ecdf.pdf", bbox_inches="tight")
    plt.close(ecdf_fig)

    metrics_fig = plot_adaptive_metrics(
        true=true_grid,
        pred=pred_grid,
        design_config=DESIGN_CONFIG,
        intrinsic_params=INTRINSIC_PARAMS,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=[r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
    )
    metrics_fig.savefig(figures_dir / "ddm_interaction_metrics.pdf", bbox_inches="tight")
    plt.close(metrics_fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BayesFlow baseline comparison on DDM interaction case")
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        choices=list(NETWORK_MAP.keys()),
        help="Inference network to train",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only evaluate from checkpoint")
    args = parser.parse_args()

    if not args.eval_only:
        train(
            network_name=args.network,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
        )

    evaluate(
        network_name=args.network,
        batch_size=args.val_batch_size,
        num_samples=args.num_samples,
    )
