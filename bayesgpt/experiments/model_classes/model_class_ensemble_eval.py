import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from bayesgpt.simulators import NestedModelFamily, ModelClass
from bayesgpt.simulators.benchmarks import DDM, RDM, CDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from bayesgpt.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from bayesgpt.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
from bayesgpt.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from bayesgpt.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.adapters import Adapter
from bayesgpt.networks.transformers.gpt.bayesgpt import BayesGPT
from bayesgpt.diagnostics.plot.ensemble_recovery import ensemble_recovery
from bayesgpt.diagnostics.plot.ensemble_coverage import ensemble_coverage
from bayesgpt.diagnostics.plot.ensemble_ecdf import ensemble_ecdf
from bayesgpt.diagnostics.plot.ensemble_metrics import ensemble_metrics


# Global parameter space variable names (ordered to match ModelClass.all_params)
# ["v", "a", "z", "tau", "s_v", "s_tau", "v_diff", "v_theta"]
ALL_PARAMS_VARIABLE_NAMES = [
    r"$v$", r"$a$", r"$z$", r"$\tau$",
    r"$s_v$", r"$s_\tau$",
    r"$v_{\mathrm{diff}}$", r"$v_\theta$",
]

DEFAULT_FIXED_VALUES = {
    "DDM": {"s_v": 0.0, "s_tau": 0.0},
    "RDM": {"s_v": 0.0, "s_tau": 0.0},
    "CDM": {"s_v": 0.0, "s_tau": 0.0},
}


def infer_free_fixed_intrinsics(design_config, all_intrinsics, default_fixed_values):
    used = {p for params in design_config.values() for p in params}
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
    p.add_argument("--outdir", type=str,
                   default="./bayesgpt/experiments/figures/fm/model_class/ensemble/")
    p.add_argument("--n_configs", type=int, default=12, help="Total random configs (4 per family by default; use 8 for 2-3 per family)")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--free_prob", type=float, default=0.5)
    p.add_argument("--metrics", action="store_true", default=False)
    # Architecture (must match checkpoint)
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
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_families = {
        "DDM": NestedModelFamily(model=DDM(), name="DDM", prior_fun=ddm_priors()),
        "RDM": NestedModelFamily(model=RDM(), name="RDM", prior_fun=rdm_priors()),
        "CDM": NestedModelFamily(model=CDM(), name="CDM", prior_fun=cdm_priors()),
    }
    link_funs = {"DDM": ddm_link_fun(), "RDM": rdm_link_fun(), "CDM": cdm_link_fun()}
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()
    cm = ContextManager()

    max_num_params = model_class.max_num_params  # 8

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, args.keep_intercept
    )
    bayesgpt = BayesGPT(
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

    state = torch.load(Path(args.checkpoint), map_location=device)
    bayesgpt.load_state_dict(state)
    bayesgpt.eval()

    # Build balanced model-name list: floor(n_configs / n_families) per family,
    # distributing any remainder to the first families.
    model_names_ordered = list(model_families.keys())
    n_families = len(model_names_ordered)
    base, remainder = divmod(args.n_configs, n_families)
    sampled_model_names = []
    for j, name in enumerate(model_names_ordered):
        sampled_model_names.extend([name] * (base + (1 if j < remainder else 0)))
    rng.shuffle(sampled_model_names)

    true_list, pred_list, masks = [], [], []
    design_configs_out, labels = [], []
    colors = sns.husl_palette(args.n_configs, h=0.05, l=0.82)

    # Per-model counter for label numbering: "DDM | 1", "RDM | 1", "DDM | 2", ...
    model_counters = {name: 0 for name in model_names_ordered}

    for i, model_name in enumerate(sampled_model_names):
        model_counters[model_name] += 1
        label = f"{model_name} | {model_counters[model_name]}"
        logging.info(f"Config {i + 1}/{args.n_configs}: {label}")

        mf = model_class.model_families[model_name]
        model_id = model_class.model_registry[model_name]
        intrinsic_params_local = mf.intrinsic_params
        link_fun = link_funs[model_name]
        default_fixed = DEFAULT_FIXED_VALUES[model_name]

        dc = cm.build_random_design_config(
            intrinsic_params=intrinsic_params_local,
            num_regressors=args.max_num_regressors,
            free_intrinsics=[p for p in intrinsic_params_local if p not in default_fixed],
            fixed_intrinsics=list(default_fixed.keys()),
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
            free_prob=args.free_prob,
        )

        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            dc, intrinsic_params_local, default_fixed
        )

        samples = mf.batch_sample(
            design_config=dc,
            batch_size=args.batch_size,
            num_obs=args.num_obs,
            mask_randomizer_kwargs={
                "free_intrinsics": free_intr,
                "fixed_intrinsics": fixed_intr,
                "fixed_values": fixed_vals,
            },
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.max_num_categories,
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
            flatten_param_outputs=True,
            fixed_config=True,
            link_fun=link_fun,
        )
        samples["model_ids"] = np.full(args.batch_size, model_id, dtype=np.int64)

        # Lift local param positions to global space before adapting
        samples["param_matrices"], samples["param_masks"] = \
            model_class.lift_to_global_space(
                model_name,
                samples["param_matrices"],
                samples["param_masks"],
            )

        adapted = adapter.adapt(
            samples,
            intrinsic_params=[],
            num_params=max_num_params,
            device=device,
        )

        pred_set = bayesgpt.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=args.num_sample_steps,
            num_samples=args.num_samples,
            model_ids=adapted["model_ids"],
        )

        # Keep in global 8-param space — no local extraction
        n_rows = adapted["param_matrices"].shape[1] // max_num_params
        true_global = adapted["param_matrices"].detach().cpu().numpy()
        true_global = true_global.reshape(args.batch_size, n_rows, max_num_params)
        pred_global = pred_set.reshape(args.batch_size, args.num_samples, n_rows, max_num_params)
        mask_global = adapted["param_masks"].detach().cpu().numpy()
        mask_global = mask_global.reshape(args.batch_size, n_rows, max_num_params)[0]

        true_list.append(true_global)
        pred_list.append(pred_global)
        masks.append(mask_global)
        design_configs_out.append(dc)
        labels.append(label)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_params = model_class.all_params
    variable_names = ALL_PARAMS_VARIABLE_NAMES

    logging.info("Generating ensemble recovery plot...")
    fig = ensemble_recovery(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=all_params,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=variable_names,
        colors=colors,
        labels=labels,
    )
    p = outdir / "model_class_ensemble_recovery.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {p}")

    logging.info("Generating ensemble coverage plot...")
    fig = ensemble_coverage(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=all_params,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=variable_names,
        colors=colors,
        labels=labels,
    )
    p = outdir / "model_class_ensemble_coverage.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {p}")

    logging.info("Generating ensemble ECDF plot...")
    fig = ensemble_ecdf(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=all_params,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=variable_names,
        colors=colors,
        labels=labels,
    )
    p = outdir / "model_class_ensemble_ecdf.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {p}")

    if args.metrics:
        logging.info("Generating ensemble metrics heatmap...")
        fig = ensemble_metrics(
            true_list=true_list,
            pred_list=pred_list,
            design_configs=design_configs_out,
            intrinsic_params=all_params,
            max_num_categories=args.max_num_categories,
            parameter_masks=masks,
            variable_names=variable_names,
        )
        p = outdir / "model_class_ensemble_metrics.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {p}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
