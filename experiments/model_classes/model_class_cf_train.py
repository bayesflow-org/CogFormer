from cogformer.utils import paths
import torch
import wandb
import logging
import argparse
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt

from cogformer.simulators import NestedModelFamily, ModelClass
from cogformer.simulators.benchmarks import DDM, RDM, CDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from cogformer.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
from cogformer.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from cogformer.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.utils.plot_utils import cogformer_fm_colors, cogformer_mc_colors, interpolate_palette
from cogformer.utils.training_utils import Prefetcher
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_c2st import compute_joint_c2st

np.set_printoptions(suppress=True)


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


# Per-model metadata used in val_step (fixed design config + display info)
MODEL_CONFIGS = {
    "DDM": {
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "val_design_config": {
            "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "u_1": ["v", "a", "z", "tau", "s_v"],
            "u_2": ["v", "a", "z", "tau"],
            "u_1:u_2": ["v", "a", "z"],
        },
        "benchmark_design_configs": {
            "intercept_only":  {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed":           {"1": ["v", "a", "z", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":       {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "a", "z", "tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "interaction":     {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
        },
    },
    "RDM": {
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "val_design_config": {
            "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_diff", "a", "tau", "s_v"],
            "u_2": ["v", "v_diff", "a", "tau"],
            "u_1:u_2": ["v", "v_diff", "a"],
        },
        "benchmark_design_configs": {
            "intercept_only":  {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed":           {"1": ["v", "v_diff", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":       {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "interaction":     {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
        },
    },
    "CDM": {
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "val_design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v"],
            "u_2": ["v", "v_theta", "a", "tau"],
            "u_1:u_2": ["v", "v_theta", "a"],
        },
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
    def __init__(self, cf, model_class, adapter, use_wandb=False, use_amp=False, num_gpus=1):
        if num_gpus > 1 and torch.cuda.is_available():
            device_ids = list(range(min(num_gpus, torch.cuda.device_count())))
            self.cf = torch.nn.DataParallel(cf, device_ids=device_ids)
        else:
            self.cf = cf
        self.model_class = model_class
        self.adapter = adapter
        self.use_wandb = use_wandb
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.device = next(cf.parameters()).device
        self.amortization_history = {
            model_name: {
                case_name: {"steps": [], "joint_c2st": []}
                for case_name in MODEL_CONFIGS[model_name]["benchmark_design_configs"]
            }
            for model_name in MODEL_CONFIGS
        }

    def _make_sample_fn(self, config):
        def sample_fn():
            return self.model_class.batch_sample(
                **config["model_family_config"],
                **config["train_sample_config"],
                batch_size=config["batch_size"],
                flatten_param_outputs=True,
            )
        return sample_fn

    def train(self, train_config, val_config, checkpoint_path="cogformer_model_class_fm.pt", fig_path="fig"):
        global_step = 0
        optimizer = AdamW(self.cf.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])
        prefetcher = Prefetcher(self._make_sample_fn(train_config))

        for epoch in range(train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {epoch + 1}/{train_config['epochs']}",
                miniters=100,
            )
            for _ in range(train_config["steps_per_epoch"]):
                samples = prefetcher.next()
                loss, current_lr = self.train_step(
                    samples=samples,
                    config=train_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                if self.use_wandb:
                    wandb.log(
                        {"train/loss": loss, "opt/lr": current_lr, "epoch": epoch + 1},
                        step=global_step,
                    )
                global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            if (epoch + 1) % 1000 == 0:
                self.val_step(val_config, global_step, fig_path)
                self.amortization_gap_step(val_config, global_step)

            scheduler.step()
            pbar.close()

        prefetcher.shutdown()
        self.plot_amortization_gap()
        checkpoint_dir = paths.checkpoints_dir("model_class")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        module = self.cf.module if isinstance(self.cf, torch.nn.DataParallel) else self.cf
        torch.save(module.state_dict(), checkpoint_dir / checkpoint_path)
        logging.info(f"[saved] {checkpoint_dir / checkpoint_path}")

    def train_step(self, samples, config, optimizer, scheduler):
        adapted = self.adapter.adapt(
            samples,
            intrinsic_params=[],
            num_params=samples["max_num_params"],
        )

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred_velocity, target_velocity = self.cf(
                adapted["param_matrices"][..., None],
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                model_ids=adapted["model_ids"],
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

    @torch.no_grad()
    def val_step(self, config, global_step, fig_path):
        self.cf.eval()

        model_family_config = config["model_family_config"]
        max_num_params = self.model_class.max_num_params
        colors = cogformer_fm_colors()

        figures_dir = paths.figures_dir("model_class", "cf", "recovery")
        figures_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_cfg in MODEL_CONFIGS.items():
            mf = self.model_class.model_families[model_name]
            model_id = self.model_class.model_registry[model_name]
            link_fun = self.model_class.link_funs[model_name]

            intrinsic_params = model_cfg["intrinsic_params"]
            variable_names = model_cfg["variable_names"]
            design_config = model_cfg["val_design_config"]

            test_samples = mf.batch_sample(
                design_config=design_config,
                batch_size=config["batch_size"],
                num_obs=config["num_obs"],
                max_num_regressors=model_family_config["max_num_regressors"],
                max_num_categories=model_family_config["max_num_categories"],
                keep_intercept=model_family_config["keep_intercept"],
                flatten_param_outputs=True,
                link_fun=link_fun,
            )
            test_samples["model_ids"] = np.full(config["batch_size"], model_id, dtype=np.int64)

            # Lift local param positions to global space before adapting
            test_samples["param_matrices"], test_samples["param_masks"] = \
                self.model_class.lift_to_global_space(
                    model_name,
                    test_samples["param_matrices"],
                    test_samples["param_masks"],
                )

            adapted = self.adapter.adapt(
                test_samples,
                intrinsic_params=[],
                num_params=max_num_params,
            )

            pred_set = self.cf.sample(
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                steps=config["fm_sample_steps"],
                num_samples=config["fm_num_samples"],
                model_ids=adapted["model_ids"],
            )

            batch_size = config["batch_size"]
            global_indices = self.model_class.local_to_global[model_name]

            true_set = adapted["param_matrices"].detach().cpu().numpy()
            n_rows = true_set.shape[1] // max_num_params
            true_set = true_set.reshape(batch_size, n_rows, max_num_params)[:, :, global_indices]
            pred_set = pred_set.reshape(batch_size, config["fm_num_samples"], n_rows, max_num_params)[:, :, :, global_indices]

            params_mask = adapted["param_masks"].detach().cpu().numpy()
            params_mask = params_mask.reshape(batch_size, n_rows, max_num_params)[0][:, global_indices]

            recovery_fig = adaptive_recovery(
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

            fig_name = f"{model_name.lower()}_{fig_path}_recovery.pdf"
            recovery_fig.savefig(figures_dir / fig_name, bbox_inches="tight")
            logging.info(f"[saved] {figures_dir / fig_name}")

            if self.use_wandb:
                wandb.log({f"val/{model_name}/recovery": wandb.Image(recovery_fig)}, step=global_step)

            plt.close(recovery_fig)

        self.cf.train()

    @torch.no_grad()
    def amortization_gap_step(self, config, global_step):
        self.cf.eval()

        max_num_params = self.model_class.max_num_params
        model_family_config = config["model_family_config"]
        max_num_regressors = model_family_config["max_num_regressors"]
        max_num_categories = model_family_config["max_num_categories"]
        keep_intercept = model_family_config["keep_intercept"]
        max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
        expected_dm_cols = max_total_regressors * (max_num_categories - 1) + (1 if keep_intercept else 0)
        log_dict = {}

        for model_name, model_cfg in MODEL_CONFIGS.items():
            if model_name != "DDM":
                continue
            model_lower = model_name.lower()
            model_id = self.model_class.model_registry[model_name]
            intrinsic_params = model_cfg["intrinsic_params"]
            global_indices = self.model_class.local_to_global[model_name]

            for case_name, design_config in model_cfg["benchmark_design_configs"].items():
                bf_path = paths.data_dir("predictions", f"{model_lower}_{case_name}_data.npz")

                if bf_path.exists():
                    bf_data = np.load(bf_path, allow_pickle=True)
                    dm = bf_data["design_matrices"]
                    if dm.shape[-1] < expected_dm_cols:
                        dm = np.pad(dm, ((0,0),(0,0),(0, expected_dm_cols - dm.shape[-1])), constant_values=0.0)
                    full_grid = len(design_config) * len(intrinsic_params)
                    true_set = bf_data["true_set"]
                    bf_masks = bf_data["param_masks"]
                    if true_set.shape[1] < full_grid:
                        pad = full_grid - true_set.shape[1]
                        true_set = np.pad(true_set, ((0,0),(0,pad)), constant_values=0.0)
                        bf_masks = np.pad(bf_masks, ((0,0),(0,pad)), constant_values=0.0)
                    lifted_params, lifted_masks = self.model_class.lift_to_global_space(
                        model_name, true_set, bf_masks,
                    )
                    batch_size = lifted_params.shape[0]
                    test_samples = {
                        "design_matrices": dm,
                        "sim_data": {"rts": bf_data["rts"], "choices": bf_data["choices"]},
                        "param_masks": lifted_masks,
                        "param_matrices": lifted_params,
                        "max_num_regressors": max_num_regressors,
                        "max_num_categories": max_num_categories,
                        "model_ids": np.full(batch_size, model_id, dtype=np.int64),
                    }
                else:
                    bf_data = None
                    mf = self.model_class.model_families[model_name]
                    link_fun = self.model_class.link_funs[model_name]
                    test_samples = mf.batch_sample(
                        design_config=design_config,
                        batch_size=config["batch_size"],
                        num_obs=config["num_obs"],
                        max_num_regressors=max_num_regressors,
                        max_num_categories=max_num_categories,
                        keep_intercept=model_family_config["keep_intercept"],
                        flatten_param_outputs=True,
                        link_fun=link_fun,
                    )
                    lifted_params, lifted_masks = self.model_class.lift_to_global_space(
                        model_name, test_samples["param_matrices"], test_samples["param_masks"],
                    )
                    batch_size = lifted_params.shape[0]
                    test_samples["param_matrices"] = lifted_params
                    test_samples["param_masks"] = lifted_masks
                    test_samples["model_ids"] = np.full(batch_size, model_id, dtype=np.int64)

                adapted = self.adapter.adapt(test_samples, intrinsic_params=[], num_params=max_num_params)
                for k, v in adapted.items():
                    if torch.is_tensor(v):
                        adapted[k] = v.to(self.device)

                pred_set = self.cf.sample(
                    adapted["input_data"], adapted["param_indices"],
                    adapted["regressor_indices"], adapted["param_masks"],
                    steps=config["fm_sample_steps"], num_samples=config["fm_num_samples"],
                    model_ids=adapted["model_ids"],
                )

                true_set = adapted["param_matrices"].detach().cpu().numpy()
                n_rows = true_set.shape[1] // max_num_params
                true_set = true_set.reshape(batch_size, n_rows, max_num_params)[:, :, global_indices]
                pred_set = pred_set.reshape(batch_size, config["fm_num_samples"], n_rows, max_num_params)[:, :, :, global_indices]
                params_mask = adapted["param_masks"].detach().cpu().numpy().reshape(batch_size, n_rows, max_num_params)[0][:, global_indices]

                metrics_df = compute_adaptive_metrics(
                    true=true_set, pred=pred_set, design_config=design_config,
                    intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
                    parameter_mask=params_mask, variable_names=model_cfg["variable_names"],
                    skip_log_gamma=True,
                )
                for metric in ["NRMSE", "Calibration Error", "Posterior Contraction"]:
                    if metric in metrics_df.columns:
                        key = f"val/{model_name}/{case_name}/{metric.lower().replace(' ', '_')}"
                        log_dict[key] = float(metrics_df[metric].mean())

                if bf_data is not None:
                    fm_pred_path = paths.data_dir("predictions", f"{model_lower}_family_{case_name}_cf_pred.npz")
                    if fm_pred_path.exists():
                        fm_data = np.load(fm_pred_path, allow_pickle=True)
                        joint_score = compute_joint_c2st(
                            pred_a=pred_set, pred_b=fm_data["pred_set"],
                            design_config=design_config, intrinsic_params=intrinsic_params,
                            max_num_categories=max_num_categories, parameter_mask=params_mask,
                        )
                        log_dict[f"val/{model_name}/{case_name}/joint_c2st"] = joint_score
                        self.amortization_history[model_name][case_name]["steps"].append(global_step)
                        self.amortization_history[model_name][case_name]["joint_c2st"].append(joint_score)

        if log_dict and self.use_wandb:
            wandb.log(log_dict, step=global_step)

        self.cf.train()

    def plot_amortization_gap(self):
        ddm_history = self.amortization_history["DDM"]
        all_steps = [s for data in ddm_history.values() for s in data["steps"]]
        max_step = max(all_steps) if all_steps else 1
        palette = interpolate_palette(cogformer_mc_colors(), len(ddm_history))
        fig, ax = plt.subplots(figsize=(7, 4))
        for (case_name, data), color in zip(ddm_history.items(), palette):
            if data["steps"]:
                norm_steps = np.asarray(data["steps"]) / max_step
                ax.plot(norm_steps, data["joint_c2st"], marker="o", markersize=3, label=case_name, color=color)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance (0.5)")
        ax.set_xlabel("Normalized training step")
        ax.set_ylabel("Joint C2ST")
        ax.set_xlim(0.1, 1.1)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(ddm_history) + 1)
        fig_dir = paths.figures_dir("model_class", "cf")
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / "model_class_amortization_gap.pdf"
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
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable automatic mixed precision (float16) training")
    parser.add_argument("--no_model_embedding", action="store_true",
                        help="Ablation: train without model identity embedding")

    # Architecture
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=128)
    parser.add_argument("--model_embed_dim", type=int, default=8)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)

    # Training schedule
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--val_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)

    # Flow matching
    parser.add_argument("--fm_sample_steps", type=int, default=200)
    parser.add_argument("--fm_num_samples", type=int, default=100)
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile for faster training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True
    num_models = None if args.no_model_embedding else 3

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "add_interaction": True,
    }

    train_sample_config = {
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "fixed_config": False,
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
        "num_obs": args.val_num_obs,
        "model_family_config": model_family_config,
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
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
        "num_models": num_models,
        "model_embed_dim": args.model_embed_dim,
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
    }

    logging.info(
        f"Training ModelClass CF: {train_config['epochs']} epochs, "
        f"{train_config['steps_per_epoch']} steps/epoch, "
        f"batch_size={train_config['batch_size']}, device={device}"
    )

    # Build per-model families with their own mask randomizer kwargs
    model_families = {
        "DDM": NestedModelFamily(
            model=DDM(), name="DDM", prior_fun=ddm_priors(),
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "a", "z", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
            },
        ),
        "RDM": NestedModelFamily(
            model=RDM(), name="RDM", prior_fun=rdm_priors(),
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_diff", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
            },
        ),
        "CDM": NestedModelFamily(
            model=CDM(), name="CDM", prior_fun=cdm_priors(),
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_theta", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
            },
        ),
    }

    link_funs = {
        "DDM": ddm_link_fun(),
        "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(),
    }

    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()
    if args.compile:
        cogformer = torch.compile(cogformer)

    if args.use_wandb:
        run_name = "cogformer-model-class-no-embed" if args.no_model_embedding else "cogformer-model-class"
        tags = ["CogFormer", "ModelClass", "DDM", "RDM", "CDM"]
        if args.no_model_embedding:
            tags.append("NoModelEmbedding")
        wandb.init(
            project="cogformer-model-class",
            name=run_name,
            tags=tags,
            config={**train_config, "cogformer": cogformer_config},
        )

    embed_tag = "_no_embed" if args.no_model_embedding else ""

    checkpoint_path = (
        f"cogformer_model_class_fm"
        f"{embed_tag}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )

    fig_path = (
        f"model_class_fm"
        f"{embed_tag}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
    )

    trainer = CogFormerTrainer(
        cf=cogformer,
        model_class=model_class,
        adapter=adapter,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
        num_gpus=args.num_gpus,
    )

    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path,
        fig_path=fig_path,
    )

    if args.use_wandb:
        trainer.finish()
