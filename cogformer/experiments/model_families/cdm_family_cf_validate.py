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
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors



def load_validation_data(data_path: Path):
    dataset = np.load(data_path, allow_pickle=True)

    rts = dataset["rts"]
    choices = dataset["choices"]
    design_matrices = dataset["design_matrices"]
    param_masks = dataset["param_masks"]

    if "true_set" not in dataset.files:
        raise ValueError(
            "BF npz is missing 'true_params'. Re-save from bf pipeline with true_set included."
        )
    true_params = dataset["true_set"]

    test_samples = {
        "design_matrices": design_matrices,
        "sim_data": {"rts": rts, "choices": choices},
        "param_matrices": true_params,
        "param_masks": param_masks,
    }
    return test_samples


def get_benchmark_design_configs():
    free_params = ["v", "v_theta", "a", "tau"]
    fixed_params = ["s_v", "s_tau"]
    intrinsic_params = free_params + fixed_params

    intercept_only = {
        "1": intrinsic_params,
        "u_1": [],
        "u_2": [],
        "u_1:u_2": []
    }

    regressed = {
        "1": intrinsic_params,
        "u_1": ["v", "a"],
        "u_2": ["v", "a"],
        "u_1:u_2": []
    }

    fixed = {
        "1": free_params,
        "u_1": [],
        "u_2": [],
        "u_1:u_2": []
    }

    fixed_regressed = {
        "1": free_params,
        "u_1": ["v", "a"],
        "u_2": ["v", "a"],
        "u_1:u_2": []
    }

    interaction = {
        "1": intrinsic_params,
        "u_1": ["v", "v_theta", "a", "tau", "s_v"],
        "u_2": ["v", "v_theta", "a", "tau"],
        "u_1:u_2": ["v", "v_theta", "a"]
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


def infer_free_fixed_intrinsics(
    design_config: dict[str, list[str]],
    all_intrinsics: list[str],
    default_fixed_values: dict[str, float],
):
    used = set()
    for _, plist in design_config.items():
        used.update(plist)

    free_intrinsics = [p for p in all_intrinsics if p in used]
    fixed_intrinsics = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed_intrinsics if p in default_fixed_values}
    return free_intrinsics, fixed_intrinsics, fixed_values


def build_encoder_input_dim(max_num_regressors: int, max_num_categories: int, keep_intercept: bool) -> int:
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    return max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained CogFormer checkpoint")
    p.add_argument("--outdir", type=str, default="./cogformer/experiments/figures/fm/cdm/", help="Output directory")
    p.add_argument("--pred_dir", type=str, default="./cogformer/experiments/data/", help="Directory to save CF pred npz files")
    p.add_argument("--data_dir", type=str, default="./cogformer/experiments/data/", help="Directory with BayesFlow validation data")

    # Validation settings
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)

    # Inference mode
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--include_full", action="store_true", default=False, help="Include the 'full' benchmark case (skipped by default)")
    p.add_argument("--skip_posteriors", action="store_true", default=False, help="Skip posterior pairplots")
    p.add_argument("--skip_log_gamma", action="store_true", default=True, help="Skip log gamma metric (slow, skipped by default)")
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false", help="Include log gamma metric")

    # MUST match training architecture
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


def test(case="fixed"):
    intrinsic_params = ["v", "v_theta", "a", "tau", "s_v", "s_tau"]
    benchmarks = get_benchmark_design_configs()
    for k, v in benchmarks.items():
        print(k, v)

    data_path = Path(f"cogformer/experiments/data/cdm_{case}_data.npz")
    data = load_validation_data(data_path)

    design_matrices = data["design_matrices"]
    num_rows = len(list(benchmarks[case].keys()))
    num_params = len(intrinsic_params)

    if design_matrices.shape[-1] == 1:
        batch_size, num_obs = design_matrices.shape[:2]
        dm = np.zeros((batch_size, num_obs, num_rows))
        dm[:, :, 0] = design_matrices.squeeze(axis=-1)
        data["design_matrices"] = dm

        pmat = np.zeros((batch_size, num_rows * num_params))
        pmask = np.zeros((batch_size, num_rows * num_params))

        pmat[:, :num_params] = data["param_matrices"]
        pmask[:, :num_params] = data["param_masks"]
        data["param_matrices"] = pmat
        data["param_masks"] = pmask

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        elif isinstance(v, dict):
            for key, val in v.items():
                print(key, val.shape)


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    intrinsic_params = ["v", "v_theta", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = NestedModelFamily(
        model=CDM(),
        name="CDM",
        prior_fun=cdm_priors(),
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

    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()

    colors = cogformer_fm_colors()
    benchmark = get_benchmark_design_configs()
    if not args.include_full:
        benchmark = {k: v for k, v in benchmark.items() if k != "full"}

    default_fixed_values = {"s_v": 0.0, "s_tau": 0.0}

    for cfg_name, design_config in benchmark.items():
        print(f"Validating case {cfg_name}")
        outdir = Path(args.outdir + cfg_name)
        outdir.mkdir(parents=True, exist_ok=True)

        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config=design_config,
            all_intrinsics=intrinsic_params,
            default_fixed_values=default_fixed_values,
        )

        val_params_kwargs = {
            "free_intrinsics": free_intr,
            "fixed_intrinsics": fixed_intr,
            "fixed_values": fixed_vals,
        }

        val_sample_config = {
            "mask_randomizer_kwargs": val_params_kwargs,
            "min_num_regressors": 0,
            "fixed_config": True,
        }

        if args.data_dir is not None:
            data_path = Path(args.data_dir) / f"cdm_{cfg_name}_data.npz"
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
            assert file_batch == args.batch_size, (
                f"Batch size mismatch: file={file_batch}, args={args.batch_size}"
            )
            test_samples = test_samples | model_family_config
        else:
            test_samples = model_family.batch_sample(
                **model_family_config,
                **val_sample_config,
                batch_size=args.batch_size,
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=cdm_link_fun()
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
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=args.num_sample_steps,
            num_samples=args.num_samples,
        )
        pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_cols)
        print(true_set.shape, pred_set.shape)

        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]

        pred_dir = Path(args.pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"cdm_families_{cfg_name}_cf_pred.npz"
        np.savez(pred_path, pred_set=pred_set, true_set=true_set, params_mask=params_mask)
        logging.info(f"[saved] {pred_path}")

        fig = adaptive_recovery(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )

        figpath = outdir / f"cdm_family_{cfg_name}_fm_mixed_recovery.pdf"
        fig.savefig(figpath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {figpath}")

        coverage = adaptive_coverage(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )

        coverage_path = outdir / f"cdm_family_{cfg_name}_fm_mixed_coverage.pdf"
        coverage.savefig(coverage_path, bbox_inches="tight")
        plt.close(coverage)
        logging.info(f"[saved] {coverage_path}")

        ecdf = adaptive_ecdf(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            difference=True,
        )
        ecdf_path = outdir / f"cdm_family_{cfg_name}_fm_mixed_ecdf.pdf"
        ecdf.savefig(ecdf_path, bbox_inches="tight")
        plt.close(ecdf)
        logging.info(f"[saved] {ecdf_path}")

        metrics_fig = plot_adaptive_metrics(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            skip_log_gamma=args.skip_log_gamma,
        )
        metrics_fig_path = outdir / f"cdm_family_{cfg_name}_fm_mixed_metrics.pdf"
        metrics_fig.savefig(metrics_fig_path, bbox_inches="tight")
        plt.close(metrics_fig)
        logging.info(f"[saved] {metrics_fig_path}")

        metrics_df = compute_adaptive_metrics(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            skip_log_gamma=args.skip_log_gamma,
        )
        metrics_csv_path = outdir / f"cdm_family_{cfg_name}_fm_mixed_metrics.csv"
        metrics_df.to_csv(metrics_csv_path)
        logging.info(f"[saved] {metrics_csv_path}")

        if not args.skip_posteriors:
            for i in range(10):
                posterior = adaptive_posterior(
                    samples=pred_set[i],
                    design_config=design_config,
                    intrinsic_params=intrinsic_params,
                    max_num_categories=args.max_num_categories,
                    unfold=False,
                    intercept_color=colors["intercept"],
                    main_effect_color=colors["main_effect"],
                    interaction_color=colors["interaction"]
                )
                posterior_path = outdir / f"cdm_family_{cfg_name}_fm_mixed_posterior_{i}.pdf"
                posterior.savefig(posterior_path, bbox_inches="tight")
                plt.close(posterior.fig)

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    debug = False

    if not debug:
        main()
    else:
        test()

    # To use:
    # python -m cogformer.experiments.model_families.cdm_family_cf_validate \
    # --checkpoint cogformer/experiments/checkpoints/fm/cdm/cogformer_cdm_mixed_attn_....pt \
    # --batch_size 200 \
    # --num_sample_steps 200 \
    # --num_samples 200
