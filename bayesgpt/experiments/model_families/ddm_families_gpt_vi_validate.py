import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from simulators import NestedModelFamily
from simulators.benchmarks import DDM
from simulators.benchmarks.ddms.ddm_priors import ddm_baseline_priors
from adapters import Adapter
from networks.transformers.gpt import BayesGPTv1
from diagnostics.plot.matrix_recovery import matrix_recovery
from diagnostics.plot.correlation import correlation


def build_everything(device: torch.device):
    # --- Must match training ---
    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True
    num_obs = 500

    # input_dim = regressors * (categories - 1) + (3 if keep_intercept else 2)
    encoder_input_dim = max_num_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "num_obs": num_obs,
    }

    mask_randomizer_kwargs = {
        "free_intrinsics": ["v", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {}
    }

    bayesgpt_config = {
        "encoder_input_dim": encoder_input_dim,
        "encoder_num_layers": 8,
        "decoder_num_layers": 8,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "num_seeds": 40,
        "seed_dim": 128,
        "proj_dim": 64,
        "dropout": 0.1,
        "layer_dropout": 0.1,
    }

    model_family = NestedModelFamily(
        model=DDM(),
        name="DDM",
        prior_fun=ddm_baseline_priors(),
        mask_randomizer_kwargs=mask_randomizer_kwargs
    )
    adapter = Adapter()
    gpt = BayesGPTv1(**bayesgpt_config).to(device).eval()

    val_config = {
        "batch_size": 300,
        "sim_config": model_family_config,
        "free_params": mask_randomizer_kwargs["free_intrinsics"],
        "fixed_params": mask_randomizer_kwargs["fixed_intrinsics"],
    }

    return model_family, adapter, gpt, val_config


def run_recovery(model_family, adapter, gpt, val_config, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample validation batch (same as trainer.validate)
    test_samples = model_family.batch_sample(
        **val_config["sim_config"],
        batch_size=val_config["batch_size"],
        flatten_param_outputs=True
    )

    adapted = adapter.adapt(
        test_samples,
        intrinsic_params=model_family.intrinsic_params
    )

    with torch.inference_mode():
        mu, logvar = gpt(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
        )

    true_set = adapted["param_matrices"].detach().cpu().numpy()
    pred_set = mu.detach().cpu().numpy()[:, :, 0]  # matches your trainer

    # Must match your plotting reshape assumptions
    params = ["v", "a", "tau", "s_v", "s_tau"]
    n_cols = len(params)
    n_rows = true_set.shape[1] // n_cols

    true_set = true_set.reshape(val_config["batch_size"], n_rows, n_cols)
    pred_set = pred_set.reshape(val_config["batch_size"], n_rows, n_cols)

    recovery_fig = matrix_recovery(
        true_set, pred_set,
        free_params=val_config["free_params"],
        fixed_params=val_config["fixed_params"],
    )
    correlation_fig = correlation(
        true_set, pred_set,
        free_params=val_config["free_params"],
        fixed_params=val_config["fixed_params"],
    )

    recovery_path = out_dir / "recovery.pdf"
    corr_path = out_dir / "correlation.pdf"

    recovery_fig.savefig(recovery_path, bbox_inches="tight")
    correlation_fig.savefig(corr_path, bbox_inches="tight")
    plt.close(recovery_fig)
    plt.close(correlation_fig)

    print(f"Saved: {recovery_path}")
    print(f"Saved: {corr_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint (state_dict)")
    parser.add_argument("--out", type=str, default="./experiments/final_validation", help="Output directory for figures")
    parser.add_argument("--device", type=str, default=None, help='e.g. "cuda" or "cpu" (default: auto)')
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_family, adapter, gpt, val_config = build_everything(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    gpt.load_state_dict(ckpt, strict=True)
    gpt.eval()

    run_recovery(model_family, adapter, gpt, val_config, Path(args.out))


if __name__ == "__main__":
    main()
