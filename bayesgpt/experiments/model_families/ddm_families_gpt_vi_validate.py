import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from bayesgpt.simulators import NestedModelFamily
from bayesgpt.simulators.benchmarks import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors2
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from bayesgpt.adapters import Adapter
from bayesgpt.networks.transformers.gpt import BayesGPTv1
from bayesgpt.diagnostics.plot.adaptive_recovery import adaptive_recovery
from bayesgpt.utils.plot_utils import bayesgpt_vi_colors


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

    av_regressed = {
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

    names = ["intercept_only", "av_regressed", "fixed", "fixed_regressed", "interaction"]
    configs = [intercept_only, av_regressed, fixed, fixed_regressed, interaction]
    return list(zip(names, configs))


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
    # Same formula you used in training:
    # input_dim = total_regressors * (categories - 1) + (3 if keep_intercept else 2)
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    return max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained BayesGPT .pt checkpoint")
    p.add_argument("--outdir", type=str, default="./experiments/figures/benchmark_recovery", help="Output directory")

    # Validation settings
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)

    # Inference mode
    p.add_argument("--point_estimates", action="store_true", help="Use point estimates instead of posterior sampling")
    p.add_argument("--num_sample_steps", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=100)

    # MUST match training architecture
    p.add_argument("--encoder_num_layers", type=int, default=4)
    p.add_argument("--decoder_num_layers", type=int, default=4)
    p.add_argument("--encoder_num_heads", type=int, default=4)
    p.add_argument("--decoder_num_heads", type=int, default=4)
    p.add_argument("--num_seeds", type=int, default=8)
    p.add_argument("--seed_dim", type=int, default=32)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--layer_dropout", type=float, default=0.1)

    return p.parse_args()


@torch.no_grad()
def main(data_path=None):
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        prior_fun=ddm_priors2(),
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

    bayesgpt = BayesGPTv1(
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
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    bayesgpt.load_state_dict(state)
    bayesgpt.eval()

    colors = bayesgpt_vi_colors()
    benchmark = get_benchmark_design_configs()

    default_fixed_values = {"s_v": 0.0, "s_tau": 0.0}

    for cfg_name, design_config in benchmark:
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

        if data_path is not None:
            test_samples = np.load(data_path, allow_pickle=True)
        else:
            # Simulate
            test_samples = model_family.batch_sample(
                **model_family_config,
                **val_sample_config,
                batch_size=args.batch_size,
                link_fun=ddm_link_fun(),
                flatten_param_outputs=True,
                design_config=design_config,
            )

        # Adapt
        adapted = adapter.adapt(test_samples, intrinsic_params=model_family.intrinsic_params)

        # Move tensors to device if adapter didn’t
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        # Forward or sample
        mu, logvar = bayesgpt(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
        )

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_cols = len(intrinsic_params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        if args.point_estimates:
            pred_set = mu.detach().cpu().numpy()[:, :, 0]
            pred_set = pred_set.reshape(args.batch_size, n_rows, n_cols)
        else:
            mu = mu.detach().cpu().numpy()[:, :, 0]
            mu = mu.reshape(args.batch_size, n_rows, n_cols)
            logvar = logvar.detach().cpu().numpy()[:, :, 0]
            sigma = np.exp(0.5 * logvar)
            sigma = sigma.reshape(args.batch_size, n_rows, n_cols)
            pred_set = np.random.normal(
                loc=mu[:, None, :, :],
                scale=sigma[:, None, :, :],
                size=(args.batch_size, args.num_samples, n_rows, n_cols),
            )

            pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_cols)

        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]

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

        if args.point_estimates:
            figpath = outdir / f"ddm_benchmark_{cfg_name}_vi_pt_recovery.pdf"
        else:
            figpath = outdir / f"ddm_benchmark_{cfg_name}_vi_post_recovery.pdf"
        fig.savefig(figpath, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"[saved] {figpath}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

    # To use (change checkpoints accordingly)
    # 1) Point estimates
    # python -m ddm_families_gpt_vi_validate.py \
    # --checkpoint bayesgpt_vi_eps500_stp200_bse32_nls4_nhs8_nss10.pt \
    # --outdir ./experiments/figures/benchmark_recovery \
    # --batch_size 200 \
    # --point_estimates
    #
    # 2) Full posterior
    # python -m ddm_families_gpt_vi_validate.py \
    # --checkpoint bayesgpt_vi_eps500_stp200_bse32_nls4_nhs8_nss10.pt \
    # --batch_size 200 \
    # --num_sample_steps 100 \
    # --num_samples 100
