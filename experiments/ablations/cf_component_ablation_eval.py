from cogformer.utils import paths
import torch
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

from cogformer.simulators import NestedModelFamily
from cogformer.simulators.benchmarks import DDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_mf_colors


DESIGN_CONFIGS = {
    "intercept_only": {
        "1":       ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1":     [],
        "u_2":     [],
        "u_1:u_2": [],
    },
    "fixed": {
        "1":       ["v", "a", "z", "tau"],
        "u_1":     [],
        "u_2":     [],
        "u_1:u_2": [],
    },
    "regressed": {
        "1":       ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1":     ["v", "a", "z"],
        "u_2":     ["v", "a", "z"],
        "u_1:u_2": [],
    },
    "fixed_regressed": {
        "1":       ["v", "a", "z", "tau"],
        "u_1":     ["v", "a", "z"],
        "u_2":     ["v", "a", "z"],
        "u_1:u_2": [],
    },
    "interaction": {
        "1":       ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1":     ["v", "a", "z", "tau", "s_v"],
        "u_2":     ["v", "a", "z", "tau"],
        "u_1:u_2": ["v", "a", "z"],
    },
}

INTRINSIC_PARAMS = ["v", "a", "z", "tau", "s_v", "s_tau"]
PARAM_NAMES = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

CONDITION_CONFIGS = {
    "baseline": {
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
        "use_film": True,
        "time_embedding_type": "fourier",
    },
    "no_sab": {
        "decoder_layer_design": "cross_attention",
        "decoder_layer_kwargs": {},
        "use_film": True,
        "time_embedding_type": "fourier",
    },
    "no_mab": {
        "decoder_layer_design": "self_attention",
        "decoder_layer_kwargs": {},
        "use_film": True,
        "time_embedding_type": "fourier",
    },
    "no_film": {
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
        "use_film": False,
        "time_embedding_type": "fourier",
    },
    "no_fourier": {
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
        "use_film": True,
        "time_embedding_type": "sinusoidal",
    },
}


def evaluate(
    condition: str,
    checkpoint: str,
    batch_size: int,
    num_samples: int,
    fm_sample_steps: int,
    num_layers: int,
    num_heads: int,
    proj_dim: int,
    num_seeds: int,
    seed_dim: int,
    time_embedding_dim: int,
    pos_embedding_dim: int,
    cases: list[str] | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cc = CONDITION_CONFIGS[condition]

    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    encoder_input_dim = max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)

    cf = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=num_layers,
        decoder_num_layers=num_layers,
        encoder_num_heads=num_heads,
        decoder_num_heads=num_heads,
        num_seeds=num_seeds,
        seed_dim=seed_dim,
        proj_dim=proj_dim,
        time_embedding_dim=time_embedding_dim,
        pos_embedding_dim=pos_embedding_dim,
        decoder_layer_design=cc["decoder_layer_design"],
        decoder_layer_kwargs=cc["decoder_layer_kwargs"],
        use_film=cc["use_film"],
        time_embedding_type=cc["time_embedding_type"],
    ).to(device)

    cf.load_state_dict(torch.load(checkpoint, map_location=device))
    cf.eval()

    val_params_kwargs = {
        "free_intrinsics": INTRINSIC_PARAMS,
        "fixed_intrinsics": [],
        "fixed_values": {},
    }

    model_family = NestedModelFamily(
        model=DDM(),
        prior_fun=ddm_priors(),
        mask_randomizer_kwargs=val_params_kwargs,
    )
    adapter = Adapter()
    colors = cogformer_mf_colors()

    stem = Path(checkpoint).stem

    fig_base = paths.figures_dir("ablations", "component")
    data_dir = paths.tables_dir("ablations", "component")
    data_dir.mkdir(parents=True, exist_ok=True)

    recovery_dir  = fig_base / "recovery";       recovery_dir.mkdir(parents=True, exist_ok=True)
    posterior_dir = fig_base / "test_posterior"; posterior_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir  = fig_base / "coverage";       coverage_dir.mkdir(parents=True, exist_ok=True)
    ecdf_dir      = fig_base / "ecdf";           ecdf_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir   = fig_base / "metrics";        metrics_dir.mkdir(parents=True, exist_ok=True)

    cases_to_run = cases if cases else list(DESIGN_CONFIGS.keys())
    for case, design_config in DESIGN_CONFIGS.items():
        if case not in cases_to_run:
            continue
        print(f"\n--- Evaluating condition={condition}, case={case} ---")

        test_samples = model_family.batch_sample(
            batch_size=batch_size,
            num_obs=500,
            min_num_regressors=2,
            max_num_regressors=max_num_regressors,
            max_num_categories=max_num_categories,
            keep_intercept=keep_intercept,
            flatten_param_outputs=True,
            design_config=design_config,
            link_fun=ddm_link_fun(),
            mask_randomizer_kwargs=val_params_kwargs,
            fixed_config=False,
        )
        adapted = adapter.adapt(test_samples, intrinsic_params=model_family.intrinsic_params)

        n_cols = len(INTRINSIC_PARAMS)
        n_rows = adapted["param_matrices"].shape[1] // n_cols
        true_set = adapted["param_matrices"].detach().cpu().numpy().reshape(batch_size, n_rows, n_cols)
        params_mask = adapted["param_masks"].detach().cpu().numpy().reshape(batch_size, n_rows, n_cols)[0]

        with torch.no_grad():
            pred_set = cf.sample(
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                steps=fm_sample_steps,
                num_samples=num_samples,
            )
        pred_set = pred_set.reshape(batch_size, num_samples, n_rows, n_cols)

        tag = f"{stem}_{case}"

        recovery_fig = adaptive_recovery(
            true_set, pred_set,
            design_config=design_config,
            intrinsic_params=INTRINSIC_PARAMS,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=PARAM_NAMES,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        recovery_fig.savefig(recovery_dir / f"{tag}.pdf", bbox_inches="tight")
        plt.close(recovery_fig)

        posterior_fig = adaptive_posterior(
            samples=pred_set[0],
            design_config=design_config,
            intrinsic_params=INTRINSIC_PARAMS,
            max_num_categories=max_num_categories,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            unfold=False,
        )
        posterior_fig.figure.savefig(posterior_dir / f"{tag}.pdf", bbox_inches="tight")
        plt.close(posterior_fig.figure)

        coverage_fig = adaptive_coverage(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=INTRINSIC_PARAMS,
            variable_names=PARAM_NAMES,
            max_num_categories=max_num_categories,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        coverage_fig.savefig(coverage_dir / f"{tag}.pdf", bbox_inches="tight")
        plt.close(coverage_fig)

        ecdf_fig = adaptive_ecdf(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=INTRINSIC_PARAMS,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=PARAM_NAMES,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            difference=True,
        )
        ecdf_fig.savefig(ecdf_dir / f"{tag}.pdf", bbox_inches="tight")
        plt.close(ecdf_fig)

        metrics_fig = plot_adaptive_metrics(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=INTRINSIC_PARAMS,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=PARAM_NAMES,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        metrics_fig.savefig(metrics_dir / f"{tag}.pdf", bbox_inches="tight")
        plt.close(metrics_fig)

        metrics_df = compute_adaptive_metrics(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=INTRINSIC_PARAMS,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=PARAM_NAMES,
        )
        metrics_df.to_csv(data_dir / f"{condition}_{case}_metrics.csv")
        print(f"Saved metrics to {data_dir / f'{condition}_{case}_metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CogFormer checkpoint for component ablation")
    parser.add_argument("--condition", type=str, required=True, choices=list(CONDITION_CONFIGS.keys()),
                        help="Condition name: baseline or one of the ablations")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint file")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--fm_sample_steps", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=64)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)
    parser.add_argument("--cases", nargs="+", default=None, choices=list(DESIGN_CONFIGS.keys()),
                        help="Which cases to evaluate (default: all)")
    args = parser.parse_args()

    evaluate(
        condition=args.condition,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        fm_sample_steps=args.fm_sample_steps,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        proj_dim=args.proj_dim,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        cases=args.cases,
    )
