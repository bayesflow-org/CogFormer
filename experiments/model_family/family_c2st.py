from cogformer.utils import paths
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cogformer.diagnostics.metric.adaptive_c2st import adaptive_c2st as compute_adaptive_c2st
from cogformer.diagnostics.metric.adaptive_c2st import compute_joint_c2st
from cogformer.diagnostics.plot.adaptive_c2st import adaptive_c2st as plot_adaptive_c2st
from cogformer.utils.plot_utils import cogformer_mf_colors


FAMILY_REGISTRY = {
    "ddm": {
        "name": "DDM",
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "outdir_default": str(paths.figures_dir("mf", "c2st", "ddm")),
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "a", "z", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "a", "z", "tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
        },
    },
    "rdm": {
        "name": "RDM",
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "outdir_default": str(paths.figures_dir("mf", "c2st", "rdm")),
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "v_diff", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
        },
    },
    "cdm": {
        "name": "CDM",
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "outdir_default": str(paths.figures_dir("mf", "c2st", "cdm")),
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "v_theta", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "v_theta", "a", "tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v"], "u_2": ["v", "v_theta", "a", "tau"], "u_1:u_2": ["v", "v_theta", "a"]},
        },
    },
}


def reshape_bf_to_gpt(bf_samples, design_config, intrinsic_params):
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
    assert flat_pos == num_active
    return result


def load_bf_data(data_path):
    dataset = np.load(data_path, allow_pickle=True)
    return {"param_masks": dataset["param_masks"], "pred_set": dataset["pred_set"]}


def load_cf_pred(pred_path):
    dataset = np.load(pred_path, allow_pickle=True)
    return dataset["pred_set"], dataset["params_mask"]


def run_comparison(pred_a, pred_b, design_config, intrinsic_params, variable_names,
                   colors, max_num_categories, params_mask, outdir, file_prefix):
    joint_score = compute_joint_c2st(
        pred_a=pred_a, pred_b=pred_b, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        parameter_mask=params_mask,
    )
    c2st_df = compute_adaptive_c2st(
        pred_a=pred_a, pred_b=pred_b, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        parameter_mask=params_mask, variable_names=variable_names,
    )
    mean_accuracy = float(c2st_df["C2ST Accuracy"].mean())

    fig = plot_adaptive_c2st(
        pred_a=pred_a, pred_b=pred_b, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        parameter_mask=params_mask, variable_names=variable_names,
        intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
        global_accuracy=mean_accuracy, joint_score=joint_score,
    )
    fig.savefig(outdir / f"{file_prefix}_c2st.pdf", bbox_inches="tight")
    plt.close(fig)

    c2st_df.loc["_mean_accuracy_"] = {"C2ST Accuracy": mean_accuracy}
    c2st_df.loc["_joint_c2st_"] = {"C2ST Accuracy": joint_score}
    c2st_df.to_csv(paths.metrics_mirror(outdir, make=True) / f"{file_prefix}_c2st.csv")
    logging.info(f"[saved] {outdir / file_prefix}_c2st.pdf  mean_acc={mean_accuracy:.3f}  joint={joint_score:.3f}")

    return mean_accuracy, joint_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mf", type=str, required=True, choices=list(FAMILY_REGISTRY.keys()))
    p.add_argument("--bf_data_dir", type=str, default=str(paths.data_dir("mf", "bf")))
    p.add_argument("--cf_pred_dir", type=str, default=str(paths.data_dir("mf", "cf")))
    p.add_argument("--mc_pred_dir", type=str, default=str(paths.data_dir("mc")),
                   help="Directory with CogFormer ModelClass pred files; if set, also runs Class vs Family C2ST")
    p.add_argument("--outdir", type=str, default=None, help="Output directory (defaults per family)")
    p.add_argument("--max_num_categories", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    reg = FAMILY_REGISTRY[args.mf]
    model = args.mf
    intrinsic_params = reg["intrinsic_params"]
    variable_names = reg["variable_names"]
    colors = cogformer_mf_colors()
    benchmark = reg["benchmark_design_configs"]
    outdir_root = args.outdir if args.outdir is not None else reg["outdir_default"]
    summary_rows = []

    for cfg_name, design_config in benchmark.items():
        bf_data_path = Path(args.bf_data_dir) / model.lower() / f"{model}_{cfg_name}_data.npz"
        cf_pred_path = Path(args.cf_pred_dir) / model.lower() / f"{model}_family_{cfg_name}_cf_pred.npz"

        if not bf_data_path.exists():
            logging.warning(f"No BF data for '{cfg_name}', skipping.")
            continue
        if not cf_pred_path.exists():
            logging.warning(f"No CF pred for '{cfg_name}', skipping.")
            continue

        outdir = Path(outdir_root) / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)

        cf_pred_set, params_mask = load_cf_pred(cf_pred_path)
        bf_data = load_bf_data(bf_data_path)
        active_idx = bf_data["param_masks"][0].astype(bool)
        bf_pred_reshaped = reshape_bf_to_gpt(
            bf_data["pred_set"][:, :, active_idx], design_config, intrinsic_params
        )

        mean_acc, joint = run_comparison(
            cf_pred_set, bf_pred_reshaped, design_config, intrinsic_params,
            variable_names, colors, args.max_num_categories, params_mask,
            outdir, f"{model}_family_{cfg_name}",
        )
        summary_rows.append({"model": reg["name"], "case": cfg_name, "scenario": "Family_vs_BF",
                              "mean_accuracy": mean_acc, "joint_c2st": joint})

        if args.mc_pred_dir is not None:
            mc_pred_path = Path(args.mc_pred_dir) / f"{model}_{cfg_name}_cf_pred.npz"
            if mc_pred_path.exists():
                mc_pred_set, _ = load_cf_pred(mc_pred_path)
                mean_acc_mc, joint_mc = run_comparison(
                    mc_pred_set, cf_pred_set, design_config, intrinsic_params,
                    variable_names, colors, args.max_num_categories, params_mask,
                    outdir, f"{model}_{cfg_name}_mc",
                )
                summary_rows.append({"model": reg["name"], "case": cfg_name, "scenario": "Class_vs_Family",
                                      "mean_accuracy": mean_acc_mc, "joint_c2st": joint_mc})
            else:
                logging.warning(f"No MC pred for '{cfg_name}' at {mc_pred_path}, skipping.")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = paths.metrics_mirror(Path(outdir_root), make=True) / f"{model}_c2st_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"[saved] {summary_path}")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
