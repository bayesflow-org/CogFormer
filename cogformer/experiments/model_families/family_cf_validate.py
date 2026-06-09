import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulators import NestedModelFamily
from cogformer.simulators.benchmarks.ddms.ddm import DDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.simulators.benchmarks.rdms.rdm import RDM
from cogformer.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from cogformer.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
from cogformer.simulators.benchmarks.cdms.cdm import CDM
from cogformer.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from cogformer.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors


FAMILY_REGISTRY = {
    "ddm": {
        "name": "DDM",
        "model_cls": DDM,
        "prior_fun": ddm_priors,
        "link_fun": ddm_link_fun,
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir_default": "./cogformer/experiments/figures/fm/ddm/",
        "pred_stem": "ddm_family",
        "fig_stem": "ddm_family",
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "a", "z", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "a", "z", "tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
            "full":           {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"]},
        },
    },
    "rdm": {
        "name": "RDM",
        "model_cls": RDM,
        "prior_fun": rdm_priors,
        "link_fun": rdm_link_fun,
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir_default": "./cogformer/experiments/figures/fm/rdm/",
        "pred_stem": "rdm_families",
        "fig_stem": "rdm_family",
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "v_diff", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
            "full":           {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"]},
        },
    },
    "cdm": {
        "name": "CDM",
        "model_cls": CDM,
        "prior_fun": cdm_priors,
        "link_fun": cdm_link_fun,
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir_default": "./cogformer/experiments/figures/fm/cdm/",
        "pred_stem": "cdm_families",
        "fig_stem": "cdm_family",
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "v_theta", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "v_theta", "a", "tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v"], "u_2": ["v", "v_theta", "a", "tau"], "u_1:u_2": ["v", "v_theta", "a"]},
            "full":           {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"]},
        },
    },
}


def load_validation_data(data_path: Path):
    dataset = np.load(data_path, allow_pickle=True)
    if "true_set" not in dataset.files:
        raise ValueError("BF npz is missing 'true_params'. Re-save from bf pipeline with true_set included.")
    return {
        "design_matrices": dataset["design_matrices"],
        "sim_data": {"rts": dataset["rts"], "choices": dataset["choices"]},
        "param_matrices": dataset["true_set"],
        "param_masks": dataset["param_masks"],
    }


def infer_free_fixed_intrinsics(design_config, all_intrinsics, default_fixed_values):
    used = {p for plist in design_config.values() for p in plist}
    free_intrinsics = [p for p in all_intrinsics if p in used]
    fixed_intrinsics = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed_intrinsics if p in default_fixed_values}
    return free_intrinsics, fixed_intrinsics, fixed_values


def build_encoder_input_dim(max_num_regressors, max_num_categories, keep_intercept):
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    return max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_family", type=str, required=True, choices=list(FAMILY_REGISTRY.keys()))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None, help="Output directory (defaults per family)")
    p.add_argument("--pred_dir", type=str, default="./cogformer/experiments/data/")
    p.add_argument("--data_dir", type=str, default="./cogformer/experiments/data/")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--include_full", action="store_true", default=False)
    p.add_argument("--skip_posteriors", action="store_true", default=False)
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--layer_dropout", type=float, default=0.05)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reg = FAMILY_REGISTRY[args.model_family]
    intrinsic_params = reg["intrinsic_params"]
    variable_names = reg["variable_names"]
    default_fixed_values = reg["default_fixed_values"]
    outdir_root = args.outdir if args.outdir is not None else reg["outdir_default"]
    pred_stem = reg["pred_stem"]
    fig_stem = reg["fig_stem"]
    fam_lower = reg["name"].lower()

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = NestedModelFamily(
        model=reg["model_cls"](),
        name=reg["name"],
        prior_fun=reg["prior_fun"](),
        mask_randomizer_kwargs={
            "free_intrinsics": intrinsic_params,
            "fixed_intrinsics": [],
            "fixed_values": {},
        },
    )
    adapter = Adapter()

    encoder_input_dim = build_encoder_input_dim(
        max_num_regressors=args.max_num_regressors,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )

    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()

    colors = cogformer_fm_colors()
    benchmark = reg["benchmark_design_configs"]
    if not args.include_full:
        benchmark = {k: v for k, v in benchmark.items() if k != "full"}

    for cfg_name, design_config in benchmark.items():
        print(f"Validating case {cfg_name}")
        outdir = Path(outdir_root + cfg_name)
        outdir.mkdir(parents=True, exist_ok=True)

        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config, intrinsic_params, default_fixed_values
        )

        val_params_kwargs = {"free_intrinsics": free_intr, "fixed_intrinsics": fixed_intr, "fixed_values": fixed_vals}
        val_sample_config = {"mask_randomizer_kwargs": val_params_kwargs, "min_num_regressors": 0, "fixed_config": True}

        if args.data_dir is not None:
            data_path = Path(args.data_dir) / f"{fam_lower}_{cfg_name}_data.npz"
            test_samples = load_validation_data(data_path=data_path)

            design_matrices = test_samples["design_matrices"]
            num_rows = len(list(benchmark[cfg_name].keys()))
            num_params = len(intrinsic_params)

            if design_matrices.shape[-1] == 1:
                batch_size, num_obs = design_matrices.shape[:2]
                dm = np.zeros((batch_size, num_obs, num_rows))
                dm[:, :, 0] = design_matrices.squeeze(axis=-1)
                test_samples["design_matrices"] = dm

            if test_samples["param_matrices"].shape[-1] < num_rows * num_params:
                batch_size = test_samples["param_matrices"].shape[0]
                pmat = np.zeros((batch_size, num_rows * num_params))
                pmask = np.zeros((batch_size, num_rows * num_params))
                pmat[:, :test_samples["param_matrices"].shape[-1]] = test_samples["param_matrices"]
                pmask[:, :test_samples["param_masks"].shape[-1]] = test_samples["param_masks"]
                test_samples["param_matrices"] = pmat
                test_samples["param_masks"] = pmask

            file_batch = test_samples["sim_data"]["rts"].shape[0]
            assert file_batch == args.batch_size, f"Batch size mismatch: file={file_batch}, args={args.batch_size}"
            test_samples = test_samples | model_family_config
        else:
            test_samples = model_family.batch_sample(
                **model_family_config,
                **val_sample_config,
                batch_size=args.batch_size,
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=reg["link_fun"]()
            )

        adapted = adapter.adapt(test_samples, intrinsic_params=model_family.intrinsic_params)
        print(adapted["input_data"].shape, adapted["param_indices"].shape, adapted["param_masks"].shape, adapted["regressor_indices"].shape)

        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_cols = len(intrinsic_params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        pred_set = cogformer.sample(
            adapted["input_data"], adapted["param_indices"],
            adapted["regressor_indices"], adapted["param_masks"],
            steps=args.num_sample_steps, num_samples=args.num_samples,
        )
        pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_cols)
        print(true_set.shape, pred_set.shape)

        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]

        pred_dir = Path(args.pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"{pred_stem}_{cfg_name}_cf_pred.npz"
        np.savez(pred_path, pred_set=pred_set, true_set=true_set, params_mask=params_mask)
        logging.info(f"[saved] {pred_path}")

        fig = adaptive_recovery(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params, max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask, variable_names=variable_names,
            intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        fig.savefig(outdir / f"{fig_stem}_{cfg_name}_fm_mixed_recovery.pdf", bbox_inches="tight")
        plt.close(fig)

        coverage = adaptive_coverage(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params, max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask, variable_names=variable_names,
            intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        coverage.savefig(outdir / f"{fig_stem}_{cfg_name}_fm_mixed_coverage.pdf", bbox_inches="tight")
        plt.close(coverage)

        ecdf = adaptive_ecdf(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params, max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask, variable_names=variable_names,
            intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"], difference=True,
        )
        ecdf.savefig(outdir / f"{fig_stem}_{cfg_name}_fm_mixed_ecdf.pdf", bbox_inches="tight")
        plt.close(ecdf)

        metrics_fig = plot_adaptive_metrics(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params, max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask, variable_names=variable_names,
            intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"], skip_log_gamma=args.skip_log_gamma,
        )
        metrics_fig.savefig(outdir / f"{fig_stem}_{cfg_name}_fm_mixed_metrics.pdf", bbox_inches="tight")
        plt.close(metrics_fig)

        metrics_df = compute_adaptive_metrics(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params, max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask, variable_names=variable_names, skip_log_gamma=args.skip_log_gamma,
        )
        metrics_df.to_csv(outdir / f"{fig_stem}_{cfg_name}_fm_mixed_metrics.csv")
        logging.info(f"[saved] {outdir / fig_stem}_{cfg_name}_fm_mixed_metrics.csv")

        if not args.skip_posteriors:
            for i in range(10):
                posterior = adaptive_posterior(
                    samples=pred_set[i], design_config=design_config, intrinsic_params=intrinsic_params,
                    max_num_categories=args.max_num_categories, unfold=False,
                    intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
                    interaction_color=colors["interaction"]
                )
                posterior.savefig(outdir / f"{fig_stem}_{cfg_name}_fm_mixed_posterior_{i}.pdf", bbox_inches="tight")
                plt.close(posterior.fig)

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
