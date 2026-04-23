import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulators import NestedModelFamily, ModelClass
from cogformer.simulators.benchmarks import DDM, RDM, CDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from cogformer.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
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
from cogformer.utils.plot_utils import cogformer_mc_colors


def load_bf_validation_data(
    data_path: Path,
    design_config: dict,
    intrinsic_params: list,
    model_id: int,
    batch_size: int,
    max_dm_cols: int | None = None,
) -> dict:
    """Load BF-generated validation data into full (batch, n_rows * n_local) param space.

    BF stores params in two formats depending on the case:
      - Intercept-only cases: (batch, n_local) — only intercept row params
      - Cases with regression rows: (batch, n_rows * n_local) — full grid already

    max_dm_cols: if provided, zero-pads design matrices to this width (needed for
    model class encoder which expects fixed-width input across all cases).
    """
    dataset = np.load(data_path, allow_pickle=True)
    rts = dataset["rts"]
    choices = dataset["choices"]
    design_matrices = dataset["design_matrices"]
    flat_params = dataset["true_set"]
    flat_masks = dataset["param_masks"]

    if max_dm_cols is not None and design_matrices.shape[-1] < max_dm_cols:
        pad_width = max_dm_cols - design_matrices.shape[-1]
        design_matrices = np.pad(design_matrices, ((0, 0), (0, 0), (0, pad_width)))

    n_local = len(intrinsic_params)
    n_rows = len(design_config)
    full_size = n_rows * n_local

    if flat_params.shape[1] == full_size:
        # BF stored as full (n_rows * n_local) grid — use directly
        pm_full = flat_params.copy()
        pmask_full = flat_masks.copy()
    else:
        # BF stored only intercept-row params — pad remaining rows with zeros
        pm_full = np.zeros((batch_size, full_size), dtype=flat_params.dtype)
        pmask_full = np.zeros((batch_size, full_size), dtype=flat_masks.dtype)
        pm_full[:, :flat_params.shape[1]] = flat_params
        pmask_full[:, :flat_masks.shape[1]] = flat_masks

    return {
        "sim_data": {"rts": rts, "choices": choices},
        "design_matrices": design_matrices,
        "param_matrices": pm_full,
        "param_masks": pmask_full,
        "model_ids": np.full(batch_size, model_id, dtype=np.int64),
    }


# Benchmark design configs per model
def get_benchmark_design_configs():
    return {
        "DDM": {
            "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
            "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
            "benchmarks": {
                "intercept_only": {
                    "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
                    "u_1": [], "u_2": [], "u_1:u_2": [],
                },
                "regressed": {
                    "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
                    "u_1": ["v", "a", "z"],
                    "u_2": ["v", "a", "z"],
                    "u_1:u_2": [],
                },
                "fixed": {
                    "1": ["v", "a", "z", "tau"],
                    "u_1": [], "u_2": [], "u_1:u_2": [],
                },
                "fixed_regressed": {
                    "1": ["v", "a", "z", "tau"],
                    "u_1": ["v", "a", "z"],
                    "u_2": ["v", "a", "z"],
                    "u_1:u_2": [],
                },
                "interaction": {
                    "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
                    "u_1": ["v", "a", "z", "tau", "s_v"],
                    "u_2": ["v", "a", "z", "tau"],
                    "u_1:u_2": ["v", "a", "z"],
                },
                "full": {
                    "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
                    "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
                    "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
                    "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
                },
            },
        },
        "RDM": {
            "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
            "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
            "benchmarks": {
                "intercept_only": {
                    "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                    "u_1": [], "u_2": [], "u_1:u_2": [],
                },
                "regressed": {
                    "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                    "u_1": ["v_diff", "a"],
                    "u_2": ["v_diff", "a"],
                    "u_1:u_2": [],
                },
                "fixed": {
                    "1": ["v", "v_diff", "a", "tau"],
                    "u_1": [], "u_2": [], "u_1:u_2": [],
                },
                "fixed_regressed": {
                    "1": ["v", "v_diff", "a", "tau"],
                    "u_1": ["v_diff", "a"],
                    "u_2": ["v_diff", "a"],
                    "u_1:u_2": [],
                },
                "interaction": {
                    "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                    "u_1": ["v", "v_diff", "a", "tau", "s_v"],
                    "u_2": ["v", "v_diff", "a", "tau"],
                    "u_1:u_2": ["v", "v_diff", "a"],
                },
                "full": {
                    "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                    "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                    "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                    "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                },
            },
        },
        "CDM": {
            "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
            "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
            "benchmarks": {
                "intercept_only": {
                    "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                    "u_1": [], "u_2": [], "u_1:u_2": [],
                },
                "regressed": {
                    "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                    "u_1": ["v_theta", "a"],
                    "u_2": ["v_theta", "a"],
                    "u_1:u_2": [],
                },
                "fixed": {
                    "1": ["v", "v_theta", "a", "tau"],
                    "u_1": [], "u_2": [], "u_1:u_2": [],
                },
                "fixed_regressed": {
                    "1": ["v", "v_theta", "a", "tau"],
                    "u_1": ["v_theta", "a"],
                    "u_2": ["v_theta", "a"],
                    "u_1:u_2": [],
                },
                "interaction": {
                    "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                    "u_1": ["v", "v_theta", "a", "tau", "s_v"],
                    "u_2": ["v", "v_theta", "a", "tau"],
                    "u_1:u_2": ["v", "v_theta", "a"],
                },
                "full": {
                    "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                    "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                    "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                    "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                },
            },
        },
    }


def infer_mask_randomizer_kwargs(
    design_config: dict[str, list[str]],
    all_intrinsics: list[str],
    default_fixed_values: dict[str, float],
) -> dict:
    used = {p for params in design_config.values() for p in params}
    free_intrinsics = [p for p in all_intrinsics if p in used]
    fixed_intrinsics = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed_intrinsics if p in default_fixed_values}
    return {"free_intrinsics": free_intrinsics, "fixed_intrinsics": fixed_intrinsics, "fixed_values": fixed_values}


def build_encoder_input_dim(max_num_regressors: int, max_num_categories: int, keep_intercept: bool) -> int:
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    return max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./cogformer/experiments/figures/fm/model_class/")
    p.add_argument("--pred_dir", type=str, default="./cogformer/experiments/data/model_class/")
    p.add_argument("--data_dir", type=str, default="./cogformer/experiments/data/", help="Directory with BayesFlow validation data")

    # Validation settings
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--include_full", action="store_true", default=False, help="Include the 'full' benchmark case (skipped by default)")
    p.add_argument("--skip_posteriors", action="store_true", default=False, help="Skip posterior pairplots")
    p.add_argument("--skip_log_gamma", action="store_true", default=True, help="Skip log gamma metric (slow, skipped by default)")
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false", help="Include log gamma metric")

    # Architecture (must match training)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_dropout", type=float, default=0.0)
    p.add_argument("--model_embed_dim", type=int, default=8)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild ModelClass to access model_registry and max_num_params
    model_families = {
        "DDM": NestedModelFamily(model=DDM(), name="DDM", prior_fun=ddm_priors()),
        "RDM": NestedModelFamily(model=RDM(), name="RDM", prior_fun=rdm_priors()),
        "CDM": NestedModelFamily(model=CDM(), name="CDM", prior_fun=cdm_priors()),
    }
    link_funs = {
        "DDM": ddm_link_fun(),
        "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(),
    }
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
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
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=model_class.num_models,
        model_embed_dim=args.model_embed_dim,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()

    colors = cogformer_mc_colors()
    all_model_configs = get_benchmark_design_configs()
    if not args.include_full:
        for mc in all_model_configs.values():
            mc["benchmarks"] = {k: v for k, v in mc["benchmarks"].items() if k != "full"}
    max_num_params = model_class.max_num_params

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
    }

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_cfg in all_model_configs.items():
        mf = model_class.model_families[model_name]
        model_id = model_class.model_registry[model_name]
        link_fun = model_class.link_funs[model_name]

        intrinsic_params = model_cfg["intrinsic_params"]
        variable_names = model_cfg["variable_names"]
        default_fixed_values = model_cfg["default_fixed_values"]
        n_cols = len(intrinsic_params)

        for cfg_name, design_config in model_cfg["benchmarks"].items():
            print(f"Validating {model_name} / {cfg_name}")

            outdir = Path(args.outdir) / model_name.lower() / cfg_name
            outdir.mkdir(parents=True, exist_ok=True)

            mask_kwargs = infer_mask_randomizer_kwargs(
                design_config=design_config,
                all_intrinsics=intrinsic_params,
                default_fixed_values=default_fixed_values,
            )

            if args.data_dir is not None:
                data_path = Path(args.data_dir) / f"{model_name.lower()}_{cfg_name}_data.npz"
                max_total_regressors = args.max_num_regressors * (args.max_num_regressors + 1) // 2
                max_dm_cols = max_total_regressors * (args.max_num_categories - 1) + (1 if args.keep_intercept else 0)
                test_samples = load_bf_validation_data(
                    data_path=data_path,
                    design_config=design_config,
                    intrinsic_params=intrinsic_params,
                    model_id=model_id,
                    batch_size=args.batch_size,
                    max_dm_cols=max_dm_cols,
                )
                test_samples["max_num_regressors"] = args.max_num_regressors
                test_samples["max_num_categories"] = args.max_num_categories
            else:
                test_samples = mf.batch_sample(
                    design_config=design_config,
                    batch_size=args.batch_size,
                    num_obs=args.num_obs,
                    mask_randomizer_kwargs=mask_kwargs,
                    max_num_regressors=args.max_num_regressors,
                    max_num_categories=args.max_num_categories,
                    keep_intercept=args.keep_intercept,
                    add_interaction=args.add_interaction,
                    flatten_param_outputs=True,
                    fixed_config=True,
                    link_fun=link_fun,
                )
                test_samples["model_ids"] = np.full(args.batch_size, model_id, dtype=np.int64)

            # Lift local param positions to global space before adapting
            test_samples["param_matrices"], test_samples["param_masks"] = \
                model_class.lift_to_global_space(
                    model_name,
                    test_samples["param_matrices"],
                    test_samples["param_masks"],
                )

            adapted = adapter.adapt(
                test_samples,
                intrinsic_params=[],
                num_params=max_num_params,
            )

            pred_set = cogformer.sample(
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                steps=args.num_sample_steps,
                num_samples=args.num_samples,
                model_ids=adapted["model_ids"],
            )

            global_indices = model_class.local_to_global[model_name]

            true_set = adapted["param_matrices"].detach().cpu().numpy()
            n_rows = true_set.shape[1] // max_num_params
            true_set = true_set.reshape(args.batch_size, n_rows, max_num_params)[:, :, global_indices]
            pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, max_num_params)[:, :, :, global_indices]

            params_mask = adapted["param_masks"].detach().cpu().numpy()
            params_mask = params_mask.reshape(args.batch_size, n_rows, max_num_params)[0][:, global_indices]

            # Save predictions
            tag = f"{model_name.lower()}_{cfg_name}"
            np.savez(
                pred_dir / f"{tag}_cf_pred.npz",
                pred_set=pred_set,
                true_set=true_set,
                params_mask=params_mask,
            )
            logging.info(f"[saved] {pred_dir / f'{tag}_cf_pred.npz'}")

            # Recovery
            fig = adaptive_recovery(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                parameter_mask=params_mask,
                variable_names=variable_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            fig_path = outdir / f"{tag}_fm_recovery.pdf"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            logging.info(f"[saved] {fig_path}")

            # Coverage
            coverage_fig = adaptive_coverage(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                parameter_mask=params_mask,
                variable_names=variable_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            coverage_path = outdir / f"{tag}_fm_coverage.pdf"
            coverage_fig.savefig(coverage_path, bbox_inches="tight")
            plt.close(coverage_fig)
            logging.info(f"[saved] {coverage_path}")

            # ECDF
            ecdf_fig = adaptive_ecdf(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                parameter_mask=params_mask,
                variable_names=variable_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
                difference=True,
            )
            ecdf_path = outdir / f"{tag}_fm_ecdf.pdf"
            ecdf_fig.savefig(ecdf_path, bbox_inches="tight")
            plt.close(ecdf_fig)
            logging.info(f"[saved] {ecdf_path}")

            # Metrics figure
            metrics_fig = plot_adaptive_metrics(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                parameter_mask=params_mask,
                variable_names=variable_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
                skip_log_gamma=args.skip_log_gamma,
            )
            metrics_fig_path = outdir / f"{tag}_fm_metrics.pdf"
            metrics_fig.savefig(metrics_fig_path, bbox_inches="tight")
            plt.close(metrics_fig)
            logging.info(f"[saved] {metrics_fig_path}")

            # Metrics CSV
            metrics_df = compute_adaptive_metrics(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                parameter_mask=params_mask,
                variable_names=variable_names,
                skip_log_gamma=args.skip_log_gamma,
            )
            metrics_csv_path = outdir / f"{tag}_fm_metrics.csv"
            metrics_df.to_csv(metrics_csv_path)
            logging.info(f"[saved] {metrics_csv_path}")

            # Posterior plots for first 10 datasets
            if not args.skip_posteriors:
                for i in range(10):
                    posterior_fig = adaptive_posterior(
                        samples=pred_set[i],
                        design_config=design_config,
                        intrinsic_params=intrinsic_params,
                        max_num_categories=args.max_num_categories,
                        unfold=False,
                        intercept_color=colors["intercept"],
                        main_effect_color=colors["main_effect"],
                        interaction_color=colors["interaction"],
                    )
                    posterior_path = outdir / f"{tag}_fm_posterior_{i}.pdf"
                    posterior_fig.savefig(posterior_path, bbox_inches="tight")
                    plt.close(posterior_fig.fig)

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
