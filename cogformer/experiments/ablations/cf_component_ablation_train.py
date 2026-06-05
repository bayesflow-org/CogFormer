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
from cogformer.simulators.benchmarks import DDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors
from cogformer.utils.training_utils import Prefetcher


ABLATION_CONFIGS = {
    "no_sab": {
        "decoder_layer_design": "cross_attention",
        "decoder_layer_kwargs": {},
        "use_film": True,
        "time_embedding_type": "fourier",
    },
    "no_mab": {
        "decoder_layer_design": "self_attention",
        "decoder_layer_kwargs": {},
        "use_film": True,
        "time_embedding_type": "fourier",
    },
    "no_film": {
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
        "use_film": False,
        "time_embedding_type": "fourier",
    },
    "no_fourier": {
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
        "use_film": True,
        "time_embedding_type": "sinusoidal",
    },
}

INTERACTION_DESIGN_CONFIG = {
    "1":       ["v", "a", "z", "tau", "s_v", "s_tau"],
    "u_1":     ["v", "a", "z", "tau", "s_v"],
    "u_2":     ["v", "a", "z", "tau"],
    "u_1:u_2": ["v", "a", "z"],
}


class CogFormerAblationTrainer:
    def __init__(self, cf, model_family, adapter, ablation, use_wandb=False, use_amp=False):
        self.cf = cf
        self.model_family = model_family
        self.adapter = adapter
        self.ablation = ablation
        self.use_wandb = use_wandb
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def _make_sample_fn(self, config):
        def sample_fn():
            return self.model_family.batch_sample(
                **config["model_family_config"],
                prior_fun=self.model_family.prior_fun,
                batch_size=config["batch_size"],
                flatten_param_outputs=True,
                link_fun=ddm_link_fun(),
            )
        return sample_fn

    def train(self, train_config, val_config, checkpoint_path="cogformer_ablation.pt", fig_path="fig.pdf"):
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
                loss, current_lr = self.train_step(samples, train_config, optimizer, scheduler)
                if self.use_wandb:
                    wandb.log(
                        {"train/loss": loss, "opt/lr": current_lr, "epoch": epoch + 1},
                        step=global_step,
                    )
                global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            if (epoch + 1) % 2500 == 0:
                self.val_step(val_config, global_step, fig_path)

            scheduler.step()
            pbar.close()

        prefetcher.shutdown()
        checkpoint_dir = Path("./cogformer/experiments/ablations/component_ablation_data/")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cf.state_dict(), checkpoint_dir / checkpoint_path)

    def train_step(self, samples, config, optimizer, scheduler):
        adapted = self.adapter.adapt(
            samples,
            intrinsic_params=self.model_family.intrinsic_params,
        )

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
        params = ["v", "a", "z", "tau", "s_v", "s_tau"]
        param_names = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"]
        colors = cogformer_fm_colors()
        max_num_categories = config["model_family_config"]["max_num_categories"]

        fig_base = Path("./cogformer/experiments/ablations/component_ablation_figures")
        stem = Path(fig_path).stem

        recovery_dir = fig_base / "recovery";        recovery_dir.mkdir(parents=True, exist_ok=True)
        posterior_dir = fig_base / "test_posterior"; posterior_dir.mkdir(parents=True, exist_ok=True)
        coverage_dir  = fig_base / "coverage";       coverage_dir.mkdir(parents=True, exist_ok=True)
        ecdf_dir      = fig_base / "ecdf";           ecdf_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir   = fig_base / "metrics";        metrics_dir.mkdir(parents=True, exist_ok=True)
        pred_dir      = Path("./cogformer/experiments/ablations/component_ablation_data")
        pred_dir.mkdir(parents=True, exist_ok=True)

        self.cf.eval()

        test_samples = self.model_family.batch_sample(
            **config["model_family_config"],
            **config["val_sample_config"],
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
            design_config=INTERACTION_DESIGN_CONFIG,
            link_fun=ddm_link_fun(),
        )
        adapted = self.adapter.adapt(
            test_samples,
            intrinsic_params=self.model_family.intrinsic_params,
        )

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        params_mask = adapted["param_masks"].detach().cpu().numpy()
        n_cols = len(params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(config["batch_size"], n_rows, n_cols)

        pred_set = self.cf.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=config["fm_sample_steps"],
            num_samples=config["fm_num_samples"],
        )
        pred_set = pred_set.reshape(config["batch_size"], config["fm_num_samples"], n_rows, n_cols)
        params_mask = params_mask.reshape(config["batch_size"], n_rows, n_cols)[0]

        np.savez(
            pred_dir / f"{stem}_interaction_pred.npz",
            pred_set=pred_set, true_set=true_set, params_mask=params_mask,
        )

        recovery = adaptive_recovery(
            true_set, pred_set,
            design_config=INTERACTION_DESIGN_CONFIG,
            intrinsic_params=params,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=param_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        recovery.savefig(recovery_dir / f"{stem}_interaction.pdf", bbox_inches="tight")
        plt.close(recovery)

        coverage = adaptive_coverage(
            true=true_set,
            pred=pred_set,
            design_config=INTERACTION_DESIGN_CONFIG,
            intrinsic_params=params,
            variable_names=param_names,
            max_num_categories=max_num_categories,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        coverage.savefig(coverage_dir / f"{stem}_interaction.pdf", bbox_inches="tight")
        plt.close(coverage)

        ecdf = adaptive_ecdf(
            true=true_set,
            pred=pred_set,
            design_config=INTERACTION_DESIGN_CONFIG,
            intrinsic_params=params,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=param_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            difference=True,
        )
        ecdf.savefig(ecdf_dir / f"{stem}_interaction.pdf", bbox_inches="tight")
        plt.close(ecdf)

        metrics_fig = plot_adaptive_metrics(
            true=true_set,
            pred=pred_set,
            design_config=INTERACTION_DESIGN_CONFIG,
            intrinsic_params=params,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=param_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        metrics_fig.savefig(metrics_dir / f"{stem}_interaction.pdf", bbox_inches="tight")
        plt.close(metrics_fig)

        metrics_df = compute_adaptive_metrics(
            true=true_set,
            pred=pred_set,
            design_config=INTERACTION_DESIGN_CONFIG,
            intrinsic_params=params,
            max_num_categories=max_num_categories,
            parameter_mask=params_mask,
            variable_names=param_names,
        )
        metrics_df.to_csv(metrics_dir / f"{stem}_interaction_metrics.csv")

        # Clean copy for downstream comparison (overwritten each val step)
        metrics_df.to_csv(pred_dir / f"{self.ablation}_metrics.csv")

        if self.use_wandb:
            wandb.log(
                {
                    "val/recovery_interaction": wandb.Image(recovery),
                    "val/coverage_interaction": wandb.Image(coverage),
                    "val/ecdf_interaction": wandb.Image(ecdf),
                    "val/metrics_interaction": wandb.Image(metrics_fig),
                },
                step=global_step,
            )

        self.cf.train()

    @staticmethod
    def finish():
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        type=str,
        choices=list(ABLATION_CONFIGS.keys()),
        required=True,
        help="Which component to ablate: no_sab, no_mab, no_film, no_fourier",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable automatic mixed precision (float16) training")

    # Architecture (defaults match the 'l' size from the size comparison)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=64)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Inference
    parser.add_argument("--fm_sample_steps", type=int, default=200)
    parser.add_argument("--fm_num_samples", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ac = ABLATION_CONFIGS[args.ablation]

    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "add_interaction": True,
    }

    train_params_kwargs = {
        "free_intrinsics": ["v", "a", "z", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"],
        "fixed_values": {"s_v": 0.0, "s_tau": 0.0},
    }
    val_params_kwargs = {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
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
        "fixed_config": False,
    }

    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    encoder_input_dim = max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)

    cogformer_config = {
        "encoder_input_dim": encoder_input_dim,
        "encoder_num_layers": args.num_layers,
        "decoder_num_layers": args.num_layers,
        "encoder_num_heads": args.num_heads,
        "decoder_num_heads": args.num_heads,
        "num_seeds": args.num_seeds,
        "seed_dim": args.seed_dim,
        "proj_dim": args.proj_dim,
        "dropout": args.dropout,
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
        "decoder_layer_design": ac["decoder_layer_design"],
        "decoder_layer_kwargs": ac["decoder_layer_kwargs"],
        "use_film": ac["use_film"],
        "time_embedding_type": ac["time_embedding_type"],
    }

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

    fig_path = (
        f"cogformer_{args.ablation}"
        f"_l{args.num_layers}"
        f"_h{args.num_heads}"
        f"_p{args.proj_dim}"
        f"_s{args.num_seeds}"
        f"_d{args.seed_dim}"
        f"_o{args.num_obs}"
        f"_b{args.train_batch_size}"
        f"_e{args.epochs}"
        f"_t{args.steps_per_epoch}.pdf"
    )
    checkpoint_path = Path(fig_path).with_suffix(".pt").name

    if args.use_wandb:
        wandb.init(
            project="cogformer-fm-ddm-ablation",
            name=f"cogformer-{args.ablation}",
            tags=["CogFormer", "ComponentAblation", args.ablation],
            config={**train_config, "cogformer": cogformer_config},
        )

    model_family = NestedModelFamily(
        model=DDM(),
        name="DDM",
        prior_fun=ddm_priors(),
        mask_randomizer_kwargs=train_params_kwargs,
    )
    adapter = Adapter()
    cf = CogFormer(**cogformer_config).to(device).train()

    trainer = CogFormerAblationTrainer(
        model_family=model_family,
        adapter=adapter,
        cf=cf,
        ablation=args.ablation,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
    )

    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path,
        fig_path=fig_path,
    )

    if args.use_wandb:
        trainer.finish()
