import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulators import NestedModelFamily
from cogformer.simulators.benchmarks.cdms.cdm import CDM
from cogformer.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from cogformer.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf.cogformer import CogFormer
from cogformer.simulators.context_manager import ContextManager
from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery
from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf
from cogformer.utils.plot_utils import cogformer_fm_colors


def get_benchmark_design_configs():
    free_params = ["v", "v_theta", "a", "tau"]
    fixed_params = ["s_v", "s_tau"]
    intrinsic_params = free_params + fixed_params

    return {
        "intercept_only": {
            "1": intrinsic_params, "u_1": [], "u_2": [], "u_1:u_2": [],
        },
        "regressed": {
            "1": intrinsic_params,
            "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": [],
        },
        "fixed": {
            "1": free_params, "u_1": [], "u_2": [], "u_1:u_2": [],
        },
        "fixed_regressed": {
            "1": free_params,
            "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": [],
        },
        "interaction": {
            "1": intrinsic_params,
            "u_1": ["v", "v_theta", "a", "tau", "s_v"],
            "u_2": ["v", "v_theta", "a", "tau"],
            "u_1:u_2": ["v", "v_theta", "a"],
        },
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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./cogformer/experiments/figures/fm/cdm/ensemble/")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--num_random_configs", type=int, default=8)
    # Architecture (must match checkpoint)
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

    intrinsic_params = ["v", "v_theta", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]
    default_fixed_values = {"s_v": 0.0, "s_tau": 0.0}
    free_params = ["v", "v_theta", "a", "tau"]
    fixed_params = ["s_v", "s_tau"]
    n_cols = len(intrinsic_params)

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = NestedModelFamily(
        model=CDM(), name="CDM", prior_fun=cdm_priors(),
        mask_randomizer_kwargs={"free_intrinsics": intrinsic_params, "fixed_intrinsics": [], "fixed_values": {}},
    )
    adapter = Adapter()
    cm = ContextManager()

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, args.keep_intercept
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    benchmark = get_benchmark_design_configs()

    case_labels = {
        "intercept_only": "Intercept Only",
        "regressed":       "Regressed",
        "fixed":           "Fixed Variability",
        "fixed_regressed": "Fixed + Regressed",
        "interaction":     "With Interaction",
    }

    # ── Benchmark ensemble (5 standard cases) ─────────────────────────────────
    true_list, pred_list, masks, design_configs, labels = [], [], [], [], []

    for cfg_name, design_config in benchmark.items():
        logging.info(f"Sampling benchmark case: {cfg_name}")
        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config, intrinsic_params, default_fixed_values
        )
        samples = model_family.batch_sample(
            **model_family_config,
            mask_randomizer_kwargs={"free_intrinsics": free_intr, "fixed_intrinsics": fixed_intr, "fixed_values": fixed_vals},
            min_num_regressors=0,
            fixed_config=True,
            batch_size=args.batch_size,
            flatten_param_outputs=True,
            design_config=design_config,
            link_fun=cdm_link_fun(),
        )
        adapted = adapter.adapt(samples, intrinsic_params=model_family.intrinsic_params)
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        pred_set = cogformer.sample(
            adapted["input_data"], adapted["param_indices"],
            adapted["regressor_indices"], adapted["param_masks"],
            steps=args.num_sample_steps, num_samples=args.num_samples,
        )
        pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_cols)

        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]

        true_list.append(true_set)
        pred_list.append(pred_set)
        masks.append(params_mask)
        design_configs.append(design_config)
        labels.append(case_labels[cfg_name])

    for plot_fn, plot_name in [
        (ensemble_recovery, "recovery"),
        (ensemble_coverage, "coverage"),
        (ensemble_ecdf, "ecdf"),
    ]:
        fig = plot_fn(
            true_list=true_list, pred_list=pred_list,
            design_configs=design_configs, intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_masks=masks, variable_names=variable_names, labels=labels,
        )
        fig_path = outdir / f"cdm_ensemble_benchmark_{plot_name}.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {fig_path}")

    # ── Random configs ensemble ────────────────────────────────────────────────
    random_design_configs = [
        cm.build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=args.max_num_regressors,
            free_intrinsics=free_params,
            fixed_intrinsics=fixed_params,
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
        )
        for _ in range(args.num_random_configs)
    ]
    random_labels = [f"Random {i + 1}" for i in range(args.num_random_configs)]
    random_true_list, random_pred_list, random_masks = [], [], []

    for rand_dc in random_design_configs:
        free_intr_r, fixed_intr_r, fixed_vals_r = infer_free_fixed_intrinsics(
            rand_dc, intrinsic_params, default_fixed_values
        )
        rand_samples = model_family.batch_sample(
            **model_family_config,
            mask_randomizer_kwargs={"free_intrinsics": free_intr_r, "fixed_intrinsics": fixed_intr_r, "fixed_values": fixed_vals_r},
            min_num_regressors=0, fixed_config=True,
            batch_size=args.batch_size, flatten_param_outputs=True,
            design_config=rand_dc, link_fun=cdm_link_fun(),
        )
        rand_adapted = adapter.adapt(rand_samples, intrinsic_params=model_family.intrinsic_params)
        for k, v in rand_adapted.items():
            if torch.is_tensor(v):
                rand_adapted[k] = v.to(device)

        rand_true = rand_adapted["param_matrices"].detach().cpu().numpy()
        n_rows_r = rand_true.shape[1] // n_cols
        rand_true = rand_true.reshape(args.batch_size, n_rows_r, n_cols)

        rand_pred = cogformer.sample(
            rand_adapted["input_data"], rand_adapted["param_indices"],
            rand_adapted["regressor_indices"], rand_adapted["param_masks"],
            steps=args.num_sample_steps, num_samples=args.num_samples,
        )
        rand_pred = rand_pred.reshape(args.batch_size, args.num_samples, n_rows_r, n_cols)

        rand_mask = rand_adapted["param_masks"].detach().cpu().numpy()
        rand_mask = rand_mask.reshape((args.batch_size, n_rows_r, n_cols))[0]

        random_true_list.append(rand_true)
        random_pred_list.append(rand_pred)
        random_masks.append(rand_mask)

    for plot_fn, plot_name in [
        (ensemble_recovery, "recovery"),
        (ensemble_coverage, "coverage"),
        (ensemble_ecdf, "ecdf"),
    ]:
        fig = plot_fn(
            true_list=random_true_list, pred_list=random_pred_list,
            design_configs=random_design_configs, intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_masks=random_masks, variable_names=variable_names,
            labels=random_labels, n_colors=args.num_random_configs,
        )
        fig_path = outdir / f"cdm_ensemble_random_{plot_name}.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {fig_path}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()