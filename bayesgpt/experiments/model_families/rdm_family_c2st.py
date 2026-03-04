import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bayesgpt.diagnostics.metric.adaptive_c2st import adaptive_c2st as compute_adaptive_c2st
from bayesgpt.diagnostics.plot.adaptive_c2st import adaptive_c2st as plot_adaptive_c2st
from bayesgpt.utils.plot_utils import bayesgpt_fm_colors


def get_benchmark_design_configs():
    free_params = ["v", "v_diff", "a", "tau"]
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
        "u_1": ["v_diff", "a"],
        "u_2": ["v_diff", "a"],
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
        "u_1": ["v_diff", "a"],
        "u_2": ["v_diff", "a"],
        "u_1:u_2": []
    }

    interaction = {
        "1": intrinsic_params,
        "u_1": ["v", "v_diff", "a", "tau", "s_v"],
        "u_2": ["v", "v_diff", "a", "tau"],
        "u_1:u_2": ["v", "v_diff", "a"]
    }

    names = ["intercept_only", "regressed", "fixed", "fixed_regressed", "interaction"]
    configs = [intercept_only, regressed, fixed, fixed_regressed, interaction]
    return {name: config for name, config in zip(names, configs)}


def reshape_bf_to_gpt(
    bf_samples: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
) -> np.ndarray:
    """
    Reshape BayesFlow flat active-param samples to (..., num_rows, num_cols).

    BayesFlow stores only active parameters in row-major order: for each row
    in design_config, active params appear in intrinsic_params order. Inactive
    cells are filled with 0 to match BayesGPT's grid layout.

    Parameters
    ----------
    bf_samples : np.ndarray of shape (..., num_active_params)
        Active-parameter array from BayesFlow, already filtered by active_idx.
    design_config : dict
        Mapping from row names to lists of active param names, in row order.
    intrinsic_params : list[str]
        Ordered list of all intrinsic param names (defines columns).

    Returns
    -------
    np.ndarray of shape (..., num_rows, num_cols)
    """
    *leading, num_active = bf_samples.shape
    num_rows = len(design_config)
    num_cols = len(intrinsic_params)
    col_idx = {p: j for j, p in enumerate(intrinsic_params)}

    result = np.zeros((*leading, num_rows, num_cols))

    flat_pos = 0
    for row_i, active_params in enumerate(design_config.values()):
        ordered = [p for p in intrinsic_params if p in active_params]
        for p in ordered:
            result[..., row_i, col_idx[p]] = bf_samples[..., flat_pos]
            flat_pos += 1

    assert flat_pos == num_active, (
        f"Expected {num_active} active params but mapped {flat_pos}. "
        "Check that design_config and intrinsic_params match the BayesFlow model."
    )
    return result


def load_bf_data(data_path: Path):
    dataset = np.load(data_path, allow_pickle=True)
    return {
        "param_masks": dataset["param_masks"],
        "pred_set": dataset["pred_set"],
    }


def load_gpt_pred(pred_path: Path):
    dataset = np.load(pred_path, allow_pickle=True)
    return dataset["pred_set"], dataset["params_mask"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bf_data_dir", type=str, default="./bayesgpt/experiments/data",
                   help="Directory with BayesFlow validation data (.npz files)")
    p.add_argument("--gpt_pred_dir", type=str, required=True,
                   help="Directory with saved BayesGPT pred .npz files (output of gpt_validate)")
    p.add_argument("--outdir", type=str, default="./bayesgpt/experiments/figures/c2st/rdm/")
    p.add_argument("--max_num_categories", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()

    intrinsic_params = ["v", "v_diff", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

    colors = bayesgpt_fm_colors()
    benchmark = get_benchmark_design_configs()

    for cfg_name, design_config in benchmark.items():
        bf_data_path = Path(args.bf_data_dir) / f"rdm_{cfg_name}_data.npz"
        gpt_pred_path = Path(args.gpt_pred_dir) / cfg_name / f"rdm_families_{cfg_name}_gpt_pred.npz"

        if not bf_data_path.exists():
            logging.warning(f"No BF data for '{cfg_name}' at {bf_data_path}, skipping.")
            continue
        if not gpt_pred_path.exists():
            logging.warning(f"No GPT pred for '{cfg_name}' at {gpt_pred_path}, skipping.")
            continue

        outdir = Path(args.outdir) / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Running C2ST for case '{cfg_name}'")

        # Load GPT predictions (already in (batch, draws, num_rows, num_cols) format)
        gpt_pred_set, params_mask = load_gpt_pred(gpt_pred_path)

        # Load BF predictions and reshape to match GPT format
        bf_data = load_bf_data(bf_data_path)
        bf_masks = bf_data["param_masks"]
        bf_pred_set = bf_data["pred_set"]

        active_idx = bf_masks[0].astype(bool)
        bf_pred_active = bf_pred_set[:, :, active_idx]  # (batch, draws, num_active)
        bf_pred_reshaped = reshape_bf_to_gpt(bf_pred_active, design_config, intrinsic_params)

        logging.info(
            f"[{cfg_name}] GPT: {gpt_pred_set.shape}, BF: {bf_pred_reshaped.shape}"
        )

        # --- C2ST ---
        c2st_fig = plot_adaptive_c2st(
            pred_a=gpt_pred_set,
            pred_b=bf_pred_reshaped,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        c2st_fig_path = outdir / f"rdm_families_{cfg_name}_c2st.pdf"
        c2st_fig.savefig(c2st_fig_path, bbox_inches="tight")
        plt.close(c2st_fig)
        logging.info(f"[saved] {c2st_fig_path}")

        c2st_df = compute_adaptive_c2st(
            pred_a=gpt_pred_set,
            pred_b=bf_pred_reshaped,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask,
            variable_names=variable_names,
        )
        c2st_csv_path = outdir / f"rdm_families_{cfg_name}_c2st.csv"
        c2st_df.to_csv(c2st_csv_path)
        logging.info(f"[saved] {c2st_csv_path}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

    # Usage:
    # python -m bayesgpt.experiments.model_families.rdm_family_c2st \
    #   --bf_data_dir bayesgpt/experiments/data \
    #   --gpt_pred_dir bayesgpt/experiments/figures/fm/rdm \
    #   --outdir bayesgpt/experiments/figures/c2st/rdm
