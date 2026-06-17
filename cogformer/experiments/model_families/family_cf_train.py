import torch
import wandb
import logging
import argparse

from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from cogformer.simulators import NestedModelFamily
from cogformer.simulators.benchmarks.ddms.ddm import DDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.simulators.benchmarks.rdms.rdm import RDM
from cogformer.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from cogformer.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
from cogformer.simulators.benchmarks.cdms.cdm import CDM
from cogformer.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from cogformer.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors, interpolate_palette
from cogformer.utils.training_utils import Prefetcher
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_c2st import compute_joint_c2st


def infer_free_fixed_intrinsics(design_config, all_intrinsics, default_fixed_values):
    used = {p for plist in design_config.values() for p in plist}
    free_intrinsics = [p for p in all_intrinsics if p in used]
    fixed_intrinsics = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed_intrinsics if p in default_fixed_values}
    return free_intrinsics, fixed_intrinsics, fixed_values


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


FAMILY_REGISTRY = {
    "ddm": {
        "name": "DDM",
        "model_cls": DDM,
        "prior_fun": ddm_priors,
        "link_fun": ddm_link_fun,
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "free_intrinsics": ["v", "a", "z", "tau"],
        "val_interval": 1000,
        "checkpoint_subdir": "fm/ddm",
        "checkpoint_stem": "cogformer_mixed_attn",
        "fig_base": "fm",
        "wandb_project": "cogformer-fm-ddm",
        "wandb_tags": ["CogFormer", "ModelFamily", "Designer"],
        "val_scenarios": [
            ("interaction", {
                "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
                "u_1": ["v", "a", "z", "tau", "s_v"],
                "u_2": ["v", "a", "z", "tau"],
                "u_1:u_2": ["v", "a", "z"],
            }),
            ("fixed", {
                "1": ["v", "a", "z", "tau"],
                "u_1": [],
                "u_2": [],
                "u_1:u_2": [],
            }),
        ],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "bf_data_stem": "ddm",
        "benchmark_design_configs": {
            "intercept_only":  {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed":           {"1": ["v", "a", "z", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":       {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "a", "z", "tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "interaction":     {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
        },
    },
    "rdm": {
        "name": "RDM",
        "model_cls": RDM,
        "prior_fun": rdm_priors,
        "link_fun": rdm_link_fun,
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "free_intrinsics": ["v", "v_diff", "a", "tau"],
        "val_interval": 1000,
        "checkpoint_subdir": "fm/rdm",
        "checkpoint_stem": "cogformer_rdm_mixed_attn",
        "fig_base": "fm/rdm",
        "wandb_project": "cogformer-fm-rdm",
        "wandb_tags": ["CogFormer", "RDM", "ModelFamily", "Designer"],
        "val_scenarios": [
            ("interaction", {
                "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                "u_1": ["v", "v_diff", "a", "tau", "s_v"],
                "u_2": ["v", "v_diff", "a", "tau"],
                "u_1:u_2": ["v", "v_diff", "a"],
            }),
            ("fixed", {
                "1": ["v", "v_diff", "a", "tau"],
                "u_1": [],
                "u_2": [],
                "u_1:u_2": [],
            }),
        ],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "bf_data_stem": "rdm",
        "benchmark_design_configs": {
            "intercept_only":  {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed":           {"1": ["v", "v_diff", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":       {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "interaction":     {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
        },
    },
    "cdm": {
        "name": "CDM",
        "model_cls": CDM,
        "prior_fun": cdm_priors,
        "link_fun": cdm_link_fun,
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "val_interval": 1000,
        "checkpoint_subdir": "fm/cdm",
        "checkpoint_stem": "cogformer_cdm_mixed_attn",
        "fig_base": "fm/cdm",
        "wandb_project": "cogformer-fm-cdm",
        "wandb_tags": ["CogFormer", "CDM", "ModelFamily"],
        "val_scenarios": [
            ("interaction", {
                "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                "u_1": ["v", "v_theta", "a", "tau", "s_v"],
                "u_2": ["v", "v_theta", "a", "tau"],
                "u_1:u_2": ["v", "v_theta", "a"],
            }),
            ("fixed", {
                "1": ["v", "v_theta", "a", "tau"],
                "u_1": [],
                "u_2": [],
                "u_1:u_2": [],
            }),
        ],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "bf_data_stem": "cdm",
        "benchmark_design_configs": {
            "intercept_only":  {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed":           {"1": ["v", "v_theta", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":       {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "v_theta", "a", "tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "interaction":     {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v"], "u_2": ["v", "v_theta", "a", "tau"], "u_1:u_2": ["v", "v_theta", "a"]},
        },
    },
}


class CogFormerTrainer:
    def __init__(self, cf, model_family, adapter, reg, use_wandb=False, use_amp=False, num_gpus=1):
        super().__init__()
        if num_gpus > 1 and torch.cuda.is_available():
            device_ids = list(range(min(num_gpus, torch.cuda.device_count())))
            self.cf = torch.nn.DataParallel(cf, device_ids=device_ids)
        else:
            self.cf = cf
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.model_family = model_family
        self.adapter = adapter
        self.use_wandb = use_wandb
        self.link_fun = reg["link_fun"]
        self.intrinsic_params = reg["intrinsic_params"]
        self.param_names = reg["param_names"]
        self.val_interval = reg["val_interval"]
        self.checkpoint_subdir = reg["checkpoint_subdir"]
        self.fig_base = reg["fig_base"]
        self.val_scenarios = reg["val_scenarios"]
        self.fam_lower = reg["name"].lower()
        self.default_fixed_values = reg["default_fixed_values"]
        self.bf_data_stem = reg["bf_data_stem"]
        self.benchmark_design_configs = reg["benchmark_design_configs"]
        self.amortization_history = {
            case: {"steps": [], "joint_c2st": []}
            for case in reg["benchmark_design_configs"]
        }
        self.device = next(cf.parameters()).device

    def _make_sample_fn(self, config):
        def sample_fn():
            return self.model_family.batch_sample(
                **config["model_family_config"],
                prior_fun=self.model_family.prior_fun,
                batch_size=config["batch_size"],
                flatten_param_outputs=True,
                link_fun=self.link_fun()
            )
        return sample_fn

    def train(self, train_config, val_config, checkpoint_path="cogformer_fm.pt", fig_path="fig.pdf"):
        global_step = 0
        optimizer = AdamW(self.cf.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])
        prefetcher = Prefetcher(self._make_sample_fn(train_config))

        for epoch in range(train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {(epoch + 1)}/{train_config['epochs']}",
                miniters=100,
            )
            for _ in range(train_config["steps_per_epoch"]):
                samples = prefetcher.next()
                loss, current_lr = self.train_step(samples=samples, config=train_config, optimizer=optimizer, scheduler=scheduler)
                if self.use_wandb:
                    wandb.log({"train/loss": loss, "opt/lr": current_lr, "epoch": epoch + 1}, step=global_step)
                global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            if (epoch + 1) % self.val_interval == 0:
                # self.val_step(val_config, global_step, fig_path)
                self.amortization_gap_step(val_config, global_step)

            scheduler.step()
            pbar.close()

        prefetcher.shutdown()
        self.plot_amortization_gap()
        checkpoint_dir = Path(f"./cogformer/experiments/checkpoints/{self.checkpoint_subdir}/")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        module = self.cf.module if isinstance(self.cf, torch.nn.DataParallel) else self.cf
        torch.save(module.state_dict(), checkpoint_dir / checkpoint_path)

    def train_step(self, samples, config, optimizer, scheduler):
        adapted = self.adapter.adapt(samples, intrinsic_params=self.model_family.intrinsic_params)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred_velocity, target_velocity = self.cf(
                adapted["param_matrices"][..., None],
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
            )
            L = self.cf.compute_loss(pred_velocity, target_velocity, adapted["param_masks"])
        if self.scaler is not None:
            self.scaler.scale(L).backward()
            self.scaler.unscale_(optimizer)
            if config["gradient_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.cf.parameters(), config["gradient_clip_norm"])
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            L.backward()
            if config["gradient_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.cf.parameters(), config["gradient_clip_norm"])
            optimizer.step()
        return L.detach().item(), scheduler.get_last_lr()[0]

    def val_step(self, config, global_step, fig_path):
        params = self.intrinsic_params
        param_names = self.param_names
        colors = cogformer_fm_colors()
        max_num_categories = config["model_family_config"]["max_num_categories"]
        fig_base = Path(f"./cogformer/experiments/figures/{self.fig_base}")

        recovery_dir = fig_base / "recovery"
        posterior_dir = fig_base / "test_posterior"
        coverage_dir = fig_base / "coverage"
        metrics_dir = fig_base / "metrics"
        for d in [recovery_dir, posterior_dir, coverage_dir, metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.cf.eval()

        for tag, design_config in self.val_scenarios:
            test_samples = self.model_family.batch_sample(
                **config["model_family_config"],
                **config["val_sample_config"],
                batch_size=config["batch_size"],
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=self.link_fun(),
            )

            adapted = self.adapter.adapt(test_samples, intrinsic_params=self.model_family.intrinsic_params)

            pred_velocity, target_velocity = self.cf(
                adapted["param_matrices"][..., None],
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
            )

            true_set = adapted["param_matrices"].detach().cpu().numpy()
            params_mask = adapted["param_masks"].detach().cpu().numpy()

            n_cols = len(params)
            n_rows = true_set.shape[1] // n_cols
            true_set = true_set.reshape(config["batch_size"], n_rows, n_cols)

            fm_sample_steps = config["fm_sample_steps"]
            fm_num_samples = config["fm_num_samples"]
            pred_set = self.cf.sample(
                adapted["input_data"], adapted["param_indices"],
                adapted["regressor_indices"], adapted["param_masks"],
                steps=fm_sample_steps, num_samples=fm_num_samples,
            )
            pred_set = pred_set.reshape(config["batch_size"], fm_num_samples, n_rows, n_cols)

            params_mask = params_mask.reshape((config["batch_size"], n_rows, n_cols))[0]

            recovery = adaptive_recovery(
                true_set, pred_set, design_config=design_config, intrinsic_params=params,
                max_num_categories=max_num_categories, parameter_mask=params_mask, variable_names=param_names,
                intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out_recovery = recovery_dir / Path(fig_path).with_stem(f"{Path(fig_path).stem}_{tag}")
            recovery.savefig(out_recovery, bbox_inches="tight")

            posterior = adaptive_posterior(
                samples=pred_set[0], design_config=design_config, intrinsic_params=params,
                max_num_categories=max_num_categories, intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"], interaction_color=colors["interaction"], unfold=False,
            )
            out_posterior = posterior_dir / f"{self.fam_lower}_benchmark_test_posterior_{tag}.pdf"
            posterior.savefig(out_posterior, bbox_inches="tight")

            coverage = adaptive_coverage(
                true=true_set, pred=pred_set, design_config=design_config, intrinsic_params=params,
                variable_names=param_names, max_num_categories=max_num_categories,
                intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out_coverage = coverage_dir / f"{self.fam_lower}_benchmark_test_coverage_{tag}.pdf"
            coverage.savefig(out_coverage, bbox_inches="tight")

            metrics_fig = plot_adaptive_metrics(
                true=true_set, pred=pred_set, design_config=design_config, intrinsic_params=params,
                max_num_categories=max_num_categories, parameter_mask=params_mask, variable_names=param_names,
                intercept_color=colors["intercept"], main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out_metrics = metrics_dir / f"{self.fam_lower}_benchmark_test_metrics_{tag}.pdf"
            metrics_fig.savefig(out_metrics, bbox_inches="tight")

            if self.use_wandb:
                wandb.log(
                    {
                        f"val/recovery_{tag}": wandb.Image(recovery),
                        f"val/posterior_{tag}": wandb.Image(posterior.fig),
                        f"val/coverage_{tag}": wandb.Image(coverage),
                        f"val/metrics_{tag}": wandb.Image(metrics_fig),
                    },
                    step=global_step,
                )
                plt.close(recovery)
                plt.close(posterior.fig)
                plt.close(coverage)
                plt.close(metrics_fig)

        self.cf.train()

    @torch.no_grad()
    def amortization_gap_step(self, config, global_step):
        self.cf.eval()

        intrinsic_params = self.intrinsic_params
        max_num_categories = config["model_family_config"]["max_num_categories"]
        max_num_regressors = config["model_family_config"]["max_num_regressors"]
        keep_intercept = config["model_family_config"]["keep_intercept"]
        max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
        expected_dm_cols = max_total_regressors * (max_num_categories - 1) + (1 if keep_intercept else 0)
        num_obs = config["val_sample_config"]["num_obs"]
        log_dict = {}

        for case_name, design_config in self.benchmark_design_configs.items():
            bf_path = Path(f"./cogformer/experiments/data/{self.bf_data_stem}_{case_name}_data.npz")

            # Pair CogFormer with BayesFlow by conditioning both on the SAME datasets.
            # When the BayesFlow npz exists, reconstruct the adapter input from its stored
            # observations (rts/choices/design_matrices) so CogFormer answers the exact
            # questions BayesFlow answered. Only then is a per-dataset C2ST meaningful.
            if bf_path.exists():
                bf_data = np.load(bf_path, allow_pickle=True)
                dm = bf_data["design_matrices"]
                if dm.shape[-1] < expected_dm_cols:
                    dm = np.pad(dm, ((0,0),(0,0),(0, expected_dm_cols - dm.shape[-1])), constant_values=0.0)
                # BF stores 1-row grids for intercept_only/fixed; pad to full n_rows×n_cols
                full_grid = len(design_config) * len(intrinsic_params)
                true_set = bf_data["true_set"]
                param_masks = bf_data["param_masks"]
                if true_set.shape[1] < full_grid:
                    pad = full_grid - true_set.shape[1]
                    true_set = np.pad(true_set, ((0,0),(0,pad)), constant_values=0.0)
                    param_masks = np.pad(param_masks, ((0,0),(0,pad)), constant_values=0.0)
                test_samples = {
                    "design_matrices": dm,
                    "sim_data": {"rts": bf_data["rts"], "choices": bf_data["choices"]},
                    "param_masks": param_masks,
                    "param_matrices": true_set,
                    "max_num_regressors": max_num_regressors,
                    "max_num_categories": max_num_categories,
                }
            else:
                # No BayesFlow baseline for this case: fall back to fresh simulation and
                # report CogFormer's own metrics only (no C2ST without a paired baseline).
                bf_data = None
                free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
                    design_config, intrinsic_params, self.default_fixed_values
                )
                test_samples = self.model_family.batch_sample(
                    **config["model_family_config"],
                    mask_randomizer_kwargs={"free_intrinsics": free_intr, "fixed_intrinsics": fixed_intr, "fixed_values": fixed_vals},
                    min_num_regressors=0,
                    num_obs=num_obs,
                    fixed_config=True,
                    batch_size=config["batch_size"],
                    flatten_param_outputs=True,
                    design_config=design_config,
                    link_fun=self.link_fun(),
                )

            adapted = self.adapter.adapt(test_samples, intrinsic_params=intrinsic_params)
            for k, v in adapted.items():
                if torch.is_tensor(v):
                    adapted[k] = v.to(self.device)

            pred_set = self.cf.sample(
                adapted["input_data"], adapted["param_indices"],
                adapted["regressor_indices"], adapted["param_masks"],
                steps=config["fm_sample_steps"], num_samples=config["fm_num_samples"],
            )

            n_cols = len(intrinsic_params)
            true_set = adapted["param_matrices"].detach().cpu().numpy()
            batch_size = true_set.shape[0]
            n_rows = true_set.shape[1] // n_cols
            true_set = true_set.reshape(batch_size, n_rows, n_cols)
            pred_set = pred_set.reshape(batch_size, config["fm_num_samples"], n_rows, n_cols)
            params_mask = adapted["param_masks"].detach().cpu().numpy().reshape(batch_size, n_rows, n_cols)[0]

            metrics_df = compute_adaptive_metrics(
                true=true_set, pred=pred_set, design_config=design_config,
                intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
                parameter_mask=params_mask, variable_names=self.param_names,
                skip_log_gamma=True,
            )
            for metric in ["NRMSE", "Calibration Error", "Posterior Contraction"]:
                if metric in metrics_df.columns:
                    key = f"val/{case_name}/{metric.lower().replace(' ', '_')}"
                    log_dict[key] = float(metrics_df[metric].mean())

            if bf_data is not None:
                # BayesFlow stores the full flat grid; select active cells and scatter
                # them back onto the (n_rows, n_cols) grid, aligned with CogFormer.
                active_idx = bf_data["param_masks"][0].astype(bool)
                bf_pred_grid = reshape_bf_to_gpt(
                    bf_data["pred_set"][:, :, active_idx], design_config, intrinsic_params
                )
                # Per-posterior C2ST over the same 10 datasets used for pair plots.
                n_plot = min(10, pred_set.shape[0], bf_pred_grid.shape[0])
                c2st_scores = [
                    compute_joint_c2st(
                        pred_a=pred_set[i], pred_b=bf_pred_grid[i],
                        design_config=design_config, intrinsic_params=intrinsic_params,
                        max_num_categories=max_num_categories, parameter_mask=params_mask,
                    )
                    for i in range(n_plot)
                ]
                joint_score = float(np.mean(c2st_scores))
                log_dict[f"val/{case_name}/joint_c2st"] = joint_score
                self.amortization_history[case_name]["steps"].append(global_step)
                self.amortization_history[case_name]["joint_c2st"].append(joint_score)

        if log_dict and self.use_wandb:
            wandb.log(log_dict, step=global_step)

        self.cf.train()

    def plot_amortization_gap(self):
        import seaborn as sns
        import pandas as pd

        all_steps = [s for data in self.amortization_history.values() for s in data["steps"]]
        max_step = max(all_steps) if all_steps else 1

        cases = list(self.amortization_history.keys())
        palette = interpolate_palette(cogformer_fm_colors(), len(cases))
        regressed_idx = cases.index("regressed") if "regressed" in cases else len(cases) // 2
        agg_color = palette[regressed_idx]

        rows = []
        for case_name, data in self.amortization_history.items():
            if data["steps"]:
                norm_steps = np.asarray(data["steps"]) / max_step
                for step, score in zip(norm_steps, data["joint_c2st"]):
                    rows.append({"norm_step": step, "joint_c2st": score})
        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(data=df, x="norm_step", y="joint_c2st", ax=ax,
                     color=agg_color, errorbar="sd", marker="o", markersize=3,
                     label="mean ± SD (all cases)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance (0.5)")
        ax.set_xlabel("Normalized training step")
        ax.set_ylabel("Joint C2ST")
        ax.set_xlim(0.1, 1.1)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        fig_dir = Path(f"./cogformer/experiments/figures/{self.fig_base}")
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / f"{self.fam_lower}_amortization_gap.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        logging.info(f"[saved] {out_path}")
        if self.use_wandb:
            wandb.log({"val/amortization_gap_plot": wandb.Image(fig)})
        plt.close(fig)

    @staticmethod
    def finish():
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, required=True, choices=list(FAMILY_REGISTRY.keys()))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=64)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--fm_sample_steps", type=int, default=200)
    parser.add_argument("--fm_num_samples", type=int, default=200)
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--use_amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile for faster training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = FAMILY_REGISTRY[args.model_family]
    use_wandb = args.use_wandb

    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "add_interaction": True
    }

    train_params_kwargs = {
        "free_intrinsics": reg["free_intrinsics"],
        "fixed_intrinsics": [p for p in reg["intrinsic_params"] if p not in reg["free_intrinsics"]],
        "fixed_values": {p: 0.0 for p in reg["intrinsic_params"] if p not in reg["free_intrinsics"]},
    }

    val_params_kwargs = {
        "free_intrinsics": reg["intrinsic_params"],
        "fixed_intrinsics": [],
        "fixed_values": {}
    }

    train_sample_config = {
        "mask_randomizer_kwargs": train_params_kwargs,
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "fixed_config": False,
    }

    val_sample_config = {
        "mask_randomizer_kwargs": val_params_kwargs,
        "min_num_regressors": 2,
        "num_obs": args.num_obs,
        "fixed_config": False
    }

    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    encoder_input_dim = max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)

    train_config = {
        "epochs": args.epochs,
        "batch_size": args.train_batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.lr,
        "gradient_clip_norm": 5.0,
        "device": device,
        "model_family_config": model_family_config,
        "train_sample_config": train_sample_config,
    }

    val_config = {
        "fm_sample_steps": args.fm_sample_steps,
        "fm_num_samples": args.fm_num_samples,
        "batch_size": args.val_batch_size,
        "model_family_config": model_family_config,
        "val_sample_config": val_sample_config,
    }

    cogformer_config = {
        "encoder_input_dim": encoder_input_dim,
        "encoder_num_layers": args.encoder_num_layers,
        "decoder_num_layers": args.decoder_num_layers,
        "encoder_num_heads": args.encoder_num_heads,
        "decoder_num_heads": args.decoder_num_heads,
        "num_seeds": args.num_seeds,
        "seed_dim": args.seed_dim,
        "proj_dim": args.projection_dim,
        "dropout": args.dropout,
        "layer_dropout": args.layer_dropout,
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
    }

    logging.info(
        f"Training {reg['name']} with {cogformer_config['decoder_layer_design']} over "
        f"{train_config['epochs']} epochs, {train_config['steps_per_epoch']} steps per epoch, "
        f"and {train_config['batch_size']} batches per step."
    )

    if use_wandb:
        wandb.init(
            project=reg["wandb_project"],
            name=None,
            tags=reg["wandb_tags"],
            config={**train_config, **{"cogformer": cogformer_config}},
        )

    model_family = NestedModelFamily(
        model=reg["model_cls"](),
        name=reg["name"],
        prior_fun=reg["prior_fun"](),
        mask_randomizer_kwargs=train_params_kwargs
    )
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()
    if args.compile:
        cogformer = torch.compile(cogformer)

    trainer = CogFormerTrainer(
        model_family=model_family,
        adapter=adapter,
        cf=cogformer,
        reg=reg,
        use_wandb=use_wandb,
        use_amp=args.use_amp,
        num_gpus=args.num_gpus,
    )

    stem = reg["checkpoint_stem"]
    checkpoint_path = (
        f"{stem}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_o{val_sample_config['num_obs']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )

    fig_path = (
        f"{stem}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_o{val_sample_config['num_obs']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}"
        f"_test.pdf"
    )

    trainer.train(train_config=train_config, val_config=val_config, checkpoint_path=checkpoint_path, fig_path=fig_path)
    trainer.finish()
