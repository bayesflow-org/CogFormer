"""
CogFormer application to real DDM data (data_application_1.csv).

Dataset structure
-----------------
- 81 participants, ~108 trials each
- Binary 2x2 factorial design encoded in four conditions:
    Condition 1: stim_type=0, factual_truth=1
    Condition 2: stim_type=1, factual_truth=1
    Condition 3: stim_type=0, factual_truth=0
    Condition 4: stim_type=1, factual_truth=0
- `rt`   : decision response time (seconds)
- `resp` : binary response (0=lower boundary, 1=upper boundary)

Design config
-------------
Intercept (baseline DDM parameters) + main effects of stim_type and factual_truth
on the free DDM parameters + their interaction. Fixed parameters s_v and s_tau are
estimated only at the intercept level (no slopes).

Outputs
-------
- ddm_application_1_pred.npz       : posterior draws for all participants
- ddm_application_1_summary.csv    : per-participant posterior means/SDs
- ddm_application_1_posterior_pid*.pdf : individual posterior pairplots (first 10)
- ddm_application_1_group_params.pdf   : group-level parameter distributions
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf.cogformer import CogFormer
from cogformer.simulators.context_manager import ContextManager
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.utils.plot_utils import cogformer_fm_colors, credible_interval


def build_encoder_input_dim(
    max_num_regressors: int, max_num_categories: int, keep_intercept: bool
) -> int:
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    return max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run CogFormer inference on data_application_1.csv"
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained CogFormer checkpoint")
    p.add_argument("--data", type=str, default="data/data_application_1.csv",
                   help="Path to the CSV data file")
    p.add_argument("--outdir", type=str,
                   default="./cogformer/experiments/figures/fm/ddm/application_1/",
                   help="Output directory for figures and results")
    p.add_argument("--num_obs", type=int, default=500,
                   help="Number of trials per participant (bootstrap resampled to this N)")
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_posterior_plots", type=int, default=10,
                   help="Number of individual posterior pairplots to save")
    # Architecture (must match training)
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


def build_participant_batch(
    df: pd.DataFrame,
    participant_ids: list,
    design_config: dict,
    cm: ContextManager,
    num_obs: int,
    max_num_categories: int,
    keep_intercept: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (design_matrices, rts, choices) arrays for all participants.

    Each participant's trials are bootstrap-resampled to `num_obs` to match
    the observation count the model was trained on.

    Returns
    -------
    design_matrices : (batch, num_obs, num_dm_cols)
    rts             : (batch, num_obs, 1)
    choices         : (batch, num_obs, 1)
    """
    all_dm, all_rt, all_ch = [], [], []

    for pid in participant_ids:
        pdata = df[df["id"] == pid]
        n_trials = len(pdata)

        # Bootstrap resample to num_obs
        idx = rng.integers(0, n_trials, size=num_obs)
        pdata_s = pdata.iloc[idx]

        stim_type_vals    = pdata_s["stim_type"].values.astype(np.float32)
        factual_truth_vals = pdata_s["factual_truth"].values.astype(np.float32)

        dm = cm.build_design_matrix(
            design_config=design_config,
            num_obs=num_obs,
            context={
                "stim_type":   stim_type_vals,
                "factual_truth": factual_truth_vals,
            },
            keep_intercept=keep_intercept,
            max_num_categories=max_num_categories,
        )  # (num_obs, num_dm_cols)

        rt  = pdata_s["rt"].values.astype(np.float32)[:, None]   # (num_obs, 1)
        ch  = pdata_s["resp"].values.astype(np.float32)[:, None]  # (num_obs, 1)

        all_dm.append(dm)
        all_rt.append(rt)
        all_ch.append(ch)

    design_matrices = np.stack(all_dm, axis=0)  # (batch, num_obs, num_dm_cols)
    rts             = np.stack(all_rt, axis=0)  # (batch, num_obs, 1)
    choices         = np.stack(all_ch, axis=0)  # (batch, num_obs, 1)

    return design_matrices, rts, choices


def save_summary_csv(
    pred_set: np.ndarray,
    participant_ids: list,
    param_mask_2d: np.ndarray,
    intrinsic_params: list[str],
    design_config: dict,
    outdir: Path,
):
    """
    Save a CSV of per-participant posterior means and 90% CI bounds
    for every active parameter cell.
    """
    batch_size, num_samples, n_rows, n_cols = pred_set.shape
    rows = []

    regressor_keys = list(design_config.keys())

    for i, pid in enumerate(participant_ids):
        row = {"participant_id": pid}
        for r in range(n_rows):
            reg_key = regressor_keys[r] if r < len(regressor_keys) else f"row_{r}"
            for c in range(n_cols):
                if param_mask_2d[r, c] != 1.0:
                    continue
                param_name = intrinsic_params[c]
                label = f"{reg_key}__{param_name}"
                draws = pred_set[i, :, r, c]   # (num_samples,)
                row[f"{label}__mean"] = float(draws.mean())
                row[f"{label}__sd"]   = float(draws.std())
                lo, hi = np.quantile(draws, [0.05, 0.95])
                row[f"{label}__q05"]  = float(lo)
                row[f"{label}__q95"]  = float(hi)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = outdir / "ddm_application_1_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logging.info(f"[saved] {csv_path}")
    return summary_df


def plot_group_params(
    pred_set: np.ndarray,
    participant_ids: list,
    param_mask_2d: np.ndarray,
    intrinsic_params: list[str],
    design_config: dict,
    colors: dict,
    outdir: Path,
    label_fontsize: int = 13,
    tick_fontsize: int = 10,
):
    """
    Violin plot of per-participant posterior means for each active parameter cell.
    One panel per regressor row, one violin per intrinsic parameter.
    """
    batch_size, num_samples, n_rows, n_cols = pred_set.shape
    regressor_keys = list(design_config.keys())

    # Posterior means per participant per cell: (batch, n_rows, n_cols)
    post_means = pred_set.mean(axis=1)

    active_rows = [
        r for r in range(n_rows)
        if param_mask_2d[r].any()
    ]
    if not active_rows:
        return

    fig, axes = plt.subplots(
        1, len(active_rows),
        figsize=(4 * len(active_rows), 4),
        squeeze=False,
    )

    for ax_idx, r in enumerate(active_rows):
        ax = axes[0, ax_idx]
        reg_key = regressor_keys[r] if r < len(regressor_keys) else f"row_{r}"

        # Determine color from regressor type
        if r == 0:
            color = colors["intercept"]
        elif ":" in reg_key:
            color = colors["interaction"]
        else:
            color = colors["main_effect"]

        active_cols = [c for c in range(n_cols) if param_mask_2d[r, c] == 1.0]
        param_labels = [intrinsic_params[c] for c in active_cols]
        data_for_violin = [post_means[:, r, c] for c in active_cols]

        parts = ax.violinplot(
            data_for_violin,
            positions=range(len(active_cols)),
            showmedians=True,
            showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            if key in parts:
                parts[key].set_color(color)
                parts[key].set_linewidth(1.2)

        ax.set_xticks(range(len(active_cols)))
        ax.set_xticklabels(param_labels, fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_title(f"${reg_key}$" if reg_key != "1" else r"$1$ (intercept)",
                     fontsize=label_fontsize)
        ax.axhline(0.0, color="black", linestyle="dashed", linewidth=0.8, alpha=0.5)
        sns.despine(ax=ax)

    fig.suptitle("Group-level parameter distributions (posterior means per participant)",
                 fontsize=label_fontsize + 1)
    fig.tight_layout()

    fig_path = outdir / "ddm_application_1_group_params.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {fig_path}")


@torch.no_grad()
def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ── Parameter setup ────────────────────────────────────────────────────────
    intrinsic_params = ["v", "a", "z", "tau", "s_v", "s_tau"]
    variable_names   = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"]
    free_params      = ["v", "a", "z", "tau"]

    # Intercept covers all params; slopes only for free (non-noise) params
    design_config = {
        "1":                          intrinsic_params,
        "stim_type":                  free_params,
        "factual_truth":              free_params,
        "stim_type:factual_truth":    free_params,
    }

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    participant_ids = sorted(df["id"].unique().tolist())
    batch_size = len(participant_ids)
    logging.info(
        f"Loaded {len(df)} trials from {batch_size} participants "
        f"({df.groupby('id').size().min()}–{df.groupby('id').size().max()} trials each)"
    )

    # ── Build parameter mask ────────────────────────────────────────────────────
    cm = ContextManager()
    param_mask_2d = cm.build_parameter_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )  # (n_rows, n_cols)
    n_rows, n_cols = param_mask_2d.shape
    param_mask_flat = param_mask_2d.flatten()
    logging.info(f"Parameter matrix: {n_rows} rows × {n_cols} cols")

    # ── Build per-participant input arrays ─────────────────────────────────────
    design_matrices, rts, choices = build_participant_batch(
        df=df,
        participant_ids=participant_ids,
        design_config=design_config,
        cm=cm,
        num_obs=args.num_obs,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
        rng=rng,
    )
    logging.info(f"Input shapes — DM: {design_matrices.shape}, RT: {rts.shape}")

    # Dummy param_matrices (zeros; only param_masks is used during sampling)
    param_matrices = np.zeros((batch_size, n_rows * n_cols), dtype=np.float32)
    param_masks    = np.tile(param_mask_flat, (batch_size, 1)).astype(np.float32)

    samples_dict = {
        "design_matrices":    design_matrices,
        "sim_data":           {"rts": rts, "choices": choices},
        "param_matrices":     param_matrices,
        "param_masks":        param_masks,
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
    }

    adapted = Adapter.adapt(samples_dict, intrinsic_params=intrinsic_params)
    for k, v in adapted.items():
        if torch.is_tensor(v):
            adapted[k] = v.to(device)

    # ── Load model ─────────────────────────────────────────────────────────────
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

    state = torch.load(args.checkpoint, map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()
    logging.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Sample posterior ───────────────────────────────────────────────────────
    logging.info(
        f"Sampling {args.num_samples} posterior draws for {batch_size} participants..."
    )
    pred_set = cogformer.sample(
        adapted["input_data"],
        adapted["param_indices"],
        adapted["regressor_indices"],
        adapted["param_masks"],
        steps=args.num_sample_steps,
        num_samples=args.num_samples,
    )  # (batch, num_samples, n_rows * n_cols)
    pred_set = pred_set.reshape(batch_size, args.num_samples, n_rows, n_cols)
    logging.info(f"Posterior shape: {pred_set.shape}")

    # ── Save raw draws ─────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / "ddm_application_1_pred.npz"
    np.savez(
        npz_path,
        pred_set=pred_set,
        param_mask=param_mask_2d,
        participant_ids=np.array(participant_ids),
    )
    logging.info(f"[saved] {npz_path}")

    # ── Summary CSV ────────────────────────────────────────────────────────────
    save_summary_csv(
        pred_set=pred_set,
        participant_ids=participant_ids,
        param_mask_2d=param_mask_2d,
        intrinsic_params=intrinsic_params,
        design_config=design_config,
        outdir=outdir,
    )

    # ── Individual posterior pairplots ─────────────────────────────────────────
    colors = cogformer_fm_colors()
    n_plots = min(args.num_posterior_plots, batch_size)

    for i in range(n_plots):
        pid = participant_ids[i]
        fig = adaptive_posterior(
            samples=pred_set[i],
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            variable_names=variable_names,
            parameter_mask=param_mask_2d,
            unfold=False,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        fig_path = outdir / f"ddm_application_1_posterior_pid{pid}.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig.fig)
        logging.info(f"[saved] {fig_path}")

    # ── Group-level parameter distribution plot ────────────────────────────────
    plot_group_params(
        pred_set=pred_set,
        participant_ids=participant_ids,
        param_mask_2d=param_mask_2d,
        intrinsic_params=intrinsic_params,
        design_config=design_config,
        colors=colors,
        outdir=outdir,
    )

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
