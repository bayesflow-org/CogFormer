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
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors


class CogFormerTrainer:
    def __init__(
        self,
        cf,
        model=None,
        prior_fun=None,
        link_fun=None,
        model_family=None,
        adapter=None,
        use_wandb=False,
    ):
        super().__init__()
        self.cf=cf
        self.model=model
        self.prior_fun=prior_fun
        self.link_fun=link_fun
        self.adapter=adapter
        self.use_wandb = use_wandb
        if model_family is not None:
            self.model_family = model_family
        else:
            self.model_family = NestedModelFamily(
                name='DDM',
                model=self.model,
                prior_fun=self.prior_fun
            )
        self.debug = False

    def train(
        self,
        train_config,
        val_config,
        checkpoint_path="cogformer_fm.pt",
        fig_path="fig.pdf"
    ):
        # Define global step
        global_step = 0

        # Define optimizer, scheduler, and loss function from config
        optimizer = AdamW(self.cf.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])

        # Training loop
        for epoch in range(train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {(epoch + 1)}/{train_config['epochs']}",
                miniters=100,
            )
            for step in range(train_config["steps_per_epoch"]):
                # Compute metrics
                loss, current_lr = self.train_step(
                    config=train_config,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                # Log metrics
                if self.use_wandb:
                     wandb.log(
                        {
                            "train/loss": loss,
                            "opt/lr": current_lr,
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                # Update internal steps
                global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            if (epoch + 1) % 250 == 0:
                self.val_step(val_config, global_step, fig_path)

            scheduler.step()
            pbar.close()

        checkpoint_dir = Path("./cogformer/experiments/checkpoints/fm/ddm/")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cf.state_dict(), checkpoint_dir / checkpoint_path)

    def train_step(self, config, optimizer, scheduler):
        """Training step"""

        # Generate training samples
        train_samples = self.model_family.batch_sample(
            **config["model_family_config"],
            prior_fun=self.prior_fun,
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
            link_fun=ddm_link_fun()
        )

        # Adapt for network
        adapted = self.adapter.adapt(
            train_samples,
            intrinsic_params=self.model_family.intrinsic_params
        )

        # Optimizer
        optimizer.zero_grad()

        # Train
        pred_velocity, target_velocity = self.cf(
            adapted["param_matrices"][..., None],
            adapted['input_data'],
            adapted['param_indices'],
            adapted['regressor_indices'],
            adapted['param_masks'],
        )

        # Compute loss
        L = self.cf.compute_loss(pred_velocity, target_velocity, adapted['param_masks'])
        L.backward()

        if config["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                self.cf.parameters(),
                config["gradient_clip_norm"],
            )

        # Update loss and learning rate
        optimizer.step()
        loss = L.detach().item()
        current_lr = scheduler.get_last_lr()[0]

        return loss, current_lr

    def val_step(self, config, global_step, fig_path):
        # Two validation scenarios
        config_1 = {
            "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "u_1": ["v", "a", "z", "tau", "s_v"],
            "u_2": ["v", "a", "z", "tau"],
            "u_1:u_2": ["v", "a", "z"],
        }
        config_2 = {
            "1": ["v", "a", "z", "tau"],
            "u_1": [],
            "u_2": [],
            "u_1:u_2": []
        }

        scenarios = [
            ("interaction", config_1),
            ("fixed", config_2),
        ]

        params = ["v", "a", "z", "tau", "s_v", "s_tau"]
        param_names = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

        colors = cogformer_fm_colors()
        max_num_categories = config["model_family_config"]["max_num_categories"]

        recovery_dir = Path("./cogformer/experiments/figures/fm/recovery")
        recovery_dir.mkdir(parents=True, exist_ok=True)
        posterior_dir = Path("./cogformer/experiments/figures/fm/test_posterior")
        posterior_dir.mkdir(parents=True, exist_ok=True)
        coverage_dir = Path("./cogformer/experiments/figures/fm/coverage")
        coverage_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = Path("./cogformer/experiments/figures/fm/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        self.cf.eval()

        for tag, design_config in scenarios:
            # Sample
            test_samples = self.model_family.batch_sample(
                **config["model_family_config"],
                **config["val_sample_config"],
                batch_size=config["batch_size"],
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=ddm_link_fun(),
            )

            # Adapt
            adapted = self.adapter.adapt(
                test_samples,
                intrinsic_params=self.model_family.intrinsic_params
            )

            # Forward
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
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                steps=fm_sample_steps,
                num_samples=fm_num_samples,
            )
            pred_set = pred_set.reshape(config["batch_size"], fm_num_samples, n_rows, n_cols)

            # Mask for plotting (same as before: take first batch element)
            params_mask = params_mask.reshape((config["batch_size"], n_rows, n_cols))[0]

            # Recovery plot
            recovery = adaptive_recovery(
                true_set, pred_set,
                design_config=design_config,
                intrinsic_params=params,
                max_num_categories=max_num_categories,
                parameter_mask=params_mask,
                variable_names=param_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )

            out_recovery = recovery_dir / Path(fig_path).with_stem(f"{Path(fig_path).stem}_{tag}")
            recovery.savefig(out_recovery, bbox_inches="tight")

            # Posterior plot
            posterior = adaptive_posterior(
                samples=pred_set[0],
                design_config=design_config,
                intrinsic_params=params,
                max_num_categories=max_num_categories,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
                unfold=False,
            )

            out_posterior = posterior_dir / Path(f"ddm_benchmark_test_posterior_{tag}.pdf")
            posterior.savefig(out_posterior, bbox_inches="tight")

            coverage = adaptive_coverage(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=params,
                variable_names=param_names,
                max_num_categories=max_num_categories,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"]
            )

            out_coverage = coverage_dir / Path(f"ddm_benchmark_test_coverage_{tag}.pdf")
            coverage.savefig(out_coverage, bbox_inches="tight")

            metrics_fig = plot_adaptive_metrics(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=params,
                max_num_categories=max_num_categories,
                parameter_mask=params_mask,
                variable_names=param_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out_metrics = metrics_dir / Path(f"ddm_benchmark_test_metrics_{tag}.pdf")
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


    @staticmethod
    def finish():
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb")

    # Network dimensions
    parser.add_argument("--encoder_num_layers", type=int, default=8, help="number of encoder layers")
    parser.add_argument("--decoder_num_layers", type=int, default=8, help="number of decoder layers")
    parser.add_argument("--encoder_num_heads", type=int, default=8, help="number of encoder heads")
    parser.add_argument("--decoder_num_heads", type=int, default=8, help="number of decoder heads")
    parser.add_argument("--projection_dim", type=int, default=256, help="dimension of projection dims")
    parser.add_argument("--num_seeds", type=int, default=32, help="number of seeds")
    parser.add_argument("--seed_dim", type=int, default=64, help="dimension of seeds")
    parser.add_argument("--time_embedding_dim", type=int, default=32, help="dimension of sinusoidal time embedding")
    parser.add_argument("--pos_embedding_dim", type=int, default=32, help="dimension of sinusoidal positional embedding")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout rate")
    parser.add_argument("--layer_dropout", type=float, default=0.05, help="layer dropout rate")

    # Training
    parser.add_argument("--num_obs", type=int, default=500, help="number of observations")
    parser.add_argument("--min_num_obs", type=int, default=200, help="minimum number of observations")
    parser.add_argument("--max_num_obs", type=int, default=500, help="maximum number of observations")
    parser.add_argument("--train_batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--val_batch_size", type=int, default=200, help="validation batch size")
    parser.add_argument("--epochs", type=int, default=5000, help="number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="number of steps per epoch")

    # Inference (for FM only)
    parser.add_argument("--fm_sample_steps", type=int, default=200, help="number of fm sample steps")
    parser.add_argument("--fm_num_samples", type=int, default=200, help="number of fm samples")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_wandb = args.use_wandb

    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True
    num_obs = args.num_obs

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "add_interaction": True
    }

    train_params_kwargs = {
        "free_intrinsics": ["v", "a", "z", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"],
        "fixed_values": {"s_v": 0.0, "s_tau": 0.0}
    }

    val_params_kwargs = {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
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

    # Automate input dim
    # input_dim = regressors * (categories - 1) + intercept + sim_data_dim (RTs, choices --> 2)
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
        # "free_params": val_params_kwargs["free_intrinsics"],
        # "fixed_params": val_params_kwargs["fixed_intrinsics"],
        # "fixed_values": val_params_kwargs["fixed_values"],
    }

    wandb_config = {
        "project_name": "cogformer-fm-ddm",
        "run_name": None,
        "tags": ["CogFormer", "ModelFamily", "Designer"],
        "watch_log": "gradients",
        "watch_freq": 200
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
        "decoder_layer_kwargs": {
            "mab_first": True
        },
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
    }

    logging.info(
        f"Training with "
        f"{cogformer_config['decoder_layer_design']} over "
        f"{train_config['epochs']} epochs, "
        f"{train_config['steps_per_epoch']} steps per epoch, and "
        f"{train_config['batch_size']} batches of dataset per step."
    )

    if use_wandb:
    # Initialize wandb
        wandb.init(
            project=wandb_config["project_name"],
            name=wandb_config["run_name"],
            tags=wandb_config["tags"],
            config={
                **train_config,
                **{"cogformer": cogformer_config},
            },
        )

    # Define model family, adapter, and network
    model_family = NestedModelFamily(
        model=DDM(),
        name="DDM",
        prior_fun=ddm_priors(),
        mask_randomizer_kwargs=train_params_kwargs
    )
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()

    # Pass to trainer
    trainer = CogFormerTrainer(
        model_family=model_family,
        adapter=adapter,
        cf=cogformer,
        use_wandb=use_wandb,
    )

    # Define checkpoint path
    checkpoint_path = (
        f"cogformer_mixed_attn"
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
        f"cogformer_mixed_attn"
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

    # Train
    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path,
        fig_path=fig_path
    )
    trainer.finish()
