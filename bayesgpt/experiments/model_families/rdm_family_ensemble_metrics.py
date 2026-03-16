import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from bayesgpt.simulators import NestedModelFamily
from bayesgpt.simulators.benchmarks.rdms.rdm import RDM
from bayesgpt.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from bayesgpt.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.adapters import Adapter
from bayesgpt.networks.transformers.gpt.bayesgpt import BayesGPT
from bayesgpt.diagnostics.plot.ensemble_metrics import ensemble_metrics


def build_encoder_input_dim(max_num_regressors: int, max_num_categories: int, keep_intercept: bool) -> int:
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    return max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./bayesgpt/experiments/figures/fm/rdm/ensemble/")
    p.add_argument("--n_configs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
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

    intrinsic_params = ["v", "v_diff", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]
    default_fixed_values = {"s_v": 0.0, "s_tau": 0.0}

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = NestedModelFamily(
        model=RDM(),
        name="RDM",
        prior_fun=rdm_priors(),
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
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    bayesgpt.load_state_dict(state)
    bayesgpt.eval()

    cm = ContextManager()
    design_configs = [
        cm.build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=args.max_num_regressors,
            free_intrinsics=["v", "v_diff", "a", "tau"],
            fixed_intrinsics=["s_v", "s_tau"],
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
        )
        for _ in range(args.n_configs)
    ]

    n_cols = len(intrinsic_params)
    true_list = []
    pred_list = []
    masks = []

    for i, dc in enumerate(design_configs):
        logging.info(f"Config {i + 1}/{args.n_configs}")
        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config=dc,
            all_intrinsics=intrinsic_params,
            default_fixed_values=default_fixed_values,
        )
        samples = model_family.batch_sample(
            **model_family_config,
            mask_randomizer_kwargs={
                "free_intrinsics": free_intr,
                "fixed_intrinsics": fixed_intr,
                "fixed_values": fixed_vals,
            },
            min_num_regressors=0,
            fixed_config=True,
            batch_size=args.batch_size,
            flatten_param_outputs=True,
            design_config=dc,
            link_fun=rdm_link_fun(),
        )
        adapted = adapter.adapt(samples, intrinsic_params=model_family.intrinsic_params)
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true = adapted["param_matrices"].detach().cpu().numpy()
        n_rows = true.shape[1] // n_cols
        true = true.reshape(args.batch_size, n_rows, n_cols)

        pred = bayesgpt.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=args.num_sample_steps,
            num_samples=args.num_samples,
        )
        pred = pred.reshape(args.batch_size, args.num_samples, n_rows, n_cols)

        mask = adapted["param_masks"].detach().cpu().numpy()
        mask = mask.reshape((args.batch_size, n_rows, n_cols))[0]

        true_list.append(true)
        pred_list.append(pred)
        masks.append(mask)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = ensemble_metrics(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs,
        intrinsic_params=intrinsic_params,
        max_num_categories=model_family_config["max_num_categories"],
        parameter_masks=masks,
        variable_names=variable_names,
    )
    fig_path = outdir / "rdm_ensemble_metrics.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {fig_path}")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
