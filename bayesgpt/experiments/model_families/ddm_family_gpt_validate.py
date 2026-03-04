import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from bayesgpt.simulators import NestedModelFamily
from bayesgpt.simulators.benchmarks import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from bayesgpt.adapters import Adapter
from bayesgpt.networks.transformers.gpt.bayesgpt import BayesGPT
from bayesgpt.diagnostics.plot.adaptive_posterior import adaptive_posterior
from bayesgpt.diagnostics.plot.adaptive_recovery import adaptive_recovery
from bayesgpt.diagnostics.plot.adaptive_coverage import adaptive_coverage
from bayesgpt.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from bayesgpt.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from bayesgpt.utils.plot_utils import bayesgpt_fm_colors


def check_data_path(data_path: str | None, case: str) -> Path | None:
    if data_path is None:
        return None

    p = Path(data_path)

    # If a directory, use default naming convention
    if p.is_dir():
        return p / f"ddm_families_bf_{case}_data.npz"

    # If a template string
    if "{case}" in str(p):
        return Path(str(p).format(case=case))

    # If a single file, treat it as that case's data (no looping)
    if p.is_file() and p.suffix == ".npz":
        return p

    raise ValueError(f"Unrecognized data_path: {data_path}")


def load_validation_data(data_path: Path):
    dataset = np.load(data_path, allow_pickle=True)

    # Required pieces
    rts = dataset["rts"]
    choices = dataset["choices"]
    design_matrices = dataset["design_matrices"]
    param_masks = dataset["param_masks"]

    # True params (targets) must be present for recovery
    if "true_set" not in dataset.files:
        raise ValueError(
            "BF npz is missing 'true_params'. Re-save from bf pipeline with true_set included."
        )
    true_params = dataset["true_set"]

    # Construct a dict shaped like NestedModelFamily.batch_sample output
    test_samples = {
        "design_matrices": design_matrices,
        "sim_data": {"rts": rts, "choices": choices},
        "param_matrices": true_params,
        "param_masks": param_masks,
        # "meta": meta,
    }
    return test_samples

def get_benchmark_design_configs():
    free_params = ["v", "a", "tau"]
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
        "u_1": ["v", "a", "tau", "s_v"],
        "u_2": ["v", "a", "tau"],
        "u_1:u_2": ["v", "a"]
    }

    benchmarks = {}

    names = ["intercept_only", "regressed", "fixed", "fixed_regressed", "interaction"]
    configs = [intercept_only, regressed, fixed, fixed_regressed, interaction]

    for name, config in zip(names, configs):
        benchmarks[name] = config
    return benchmarks


def infer_free_fixed_intrinsics(
    design_config: dict[str, list[str]],
    all_intrinsics: list[str],
    default_fixed_values: dict[str, float],
):
    # If a param never appears in the design_config value-lists, treat it as fixed.
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
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained BayesGPT checkpoint")
    p.add_argument("--outdir", type=str, default="./bayesgpt/experiments/figures/fm/ddm/", help="Output directory")
    p.add_argument("--data_dir", type=str, default=None, help="Directory with BayesFlow validation data")

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

    return p.parse_args()


def test(case="fixed"):
    intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]
    benchmarks = get_benchmark_design_configs()
    for k, v in benchmarks.items():
        print(k, v)

    data_path = Path(f"bayesgpt/experiments/data/ddm_{case}_data.npz")
    data = load_validation_data(data_path)

    design_matrices = data["design_matrices"]
    num_rows = len(list(benchmarks[case].keys()))
    num_params = len(intrinsic_params)

    if design_matrices.shape[-1] == 1:
        batch_size, num_obs = design_matrices.shape[:2]
        dm = np.zeros((batch_size, num_obs, num_rows))
        dm[:,:,0] = design_matrices.squeeze(axis=-1)
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

    # Intrinsics / display names
    intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

    # Rebuild model-family exactly like training
    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = NestedModelFamily(
        model=DDM(),
        name="DDM",
        prior_fun=ddm_priors(),
        mask_randomizer_kwargs={  # default; per-config we’ll override
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
        decoder_layer_design="self_attention",
        decoder_layer_kwargs={"skip_first": True},
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    bayesgpt.load_state_dict(state)
    bayesgpt.eval()

    colors = bayesgpt_fm_colors()
    benchmark = get_benchmark_design_configs()

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

        # Per-config mask randomizer settings
        val_params_kwargs = {
            "free_intrinsics": free_intr,
            "fixed_intrinsics": fixed_intr,
            "fixed_values": fixed_vals,
        }

        val_sample_config = {
            "mask_randomizer_kwargs": val_params_kwargs,
            "min_num_regressors": 0,
            "fixed_config": True,  # make intent explicit
        }

        if args.data_dir is not None:
            data_path = Path(args.data_dir) / f"ddm_{cfg_name}_data.npz"
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
            # Simulate
            test_samples = model_family.batch_sample(
                **model_family_config,
                **val_sample_config,
                batch_size=args.batch_size,
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=ddm_link_fun()
            )

        # Adapt
        adapted = adapter.adapt(test_samples, intrinsic_params=model_family.intrinsic_params)
        print(adapted["input_data"].shape, adapted["param_indices"].shape, adapted["param_masks"].shape, adapted["regressor_indices"].shape)

        # Move tensors to device if adapter didn’t
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_cols = len(intrinsic_params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        pred_set = bayesgpt.sample(
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

        pred_path = outdir / f"ddm_families_{cfg_name}_gpt_pred.npz"
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

        figpath = outdir / f"ddm_families_{cfg_name}_fm_mixed_m_recovery.pdf"
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

        coverage_path = outdir / f"ddm_families_{cfg_name}_fm_mixed_m_coverage.pdf"
        coverage.savefig(coverage_path, bbox_inches="tight")
        plt.close(coverage)

        logging.info(f"[saved] {coverage_path}")

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
        )
        metrics_fig_path = outdir / f"ddm_families_{cfg_name}_fm_mixed_m_metrics.pdf"
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
        )
        metrics_csv_path = outdir / f"ddm_families_{cfg_name}_fm_mixed_m_metrics.csv"
        metrics_df.to_csv(metrics_csv_path)
        logging.info(f"[saved] {metrics_csv_path}")

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
            posterior_path = outdir / f"ddm_families_{cfg_name}_fm_mixed_m_posterior{i}.pdf"
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

    # To use
    # python -m ddm_families_gpt_fm_validate.py \
    # --checkpoint bayesgpt_fm_eps500_stp200_bse32_nls4_nhs8_nss10.pt \
    # --batch_size 200 \
    # --num_sample_steps 100 \
    # --num_samples 100
