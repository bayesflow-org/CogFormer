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

from bayesgpt.simulators import NestedModelFamily
from bayesgpt.simulators.benchmarks import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors, ddm_priors2
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun, ddm_link_fun2
from bayesgpt.adapters import Adapter
from bayesgpt.networks.transformers.gpt import BayesGPT
from bayesgpt.diagnostics.plot.adaptive_recovery import adaptive_recovery
from bayesgpt.utils.plot_utils import bayesgpt_fm_colors


class BayesGPTTrainer:
    def __init__(
        self,
        gpt,
        model=None,
        prior_fun=None,
        link_fun=None,
        model_family=None,
        adapter=None,
        use_wandb=False,
    ):
        super().__init__()
        self.gpt=gpt
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

    def train(self, train_config, val_config, checkpoint_path="bayesgpt_fm.pt"):
        # Define global step
        global_step = 0

        # Define optimizer, scheduler, and loss function from config
        optimizer = AdamW(self.gpt.parameters(), lr=train_config["learning_rate"])
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

            if (epoch + 1) % 100 == 0:
                self.val_step(val_config, global_step)

            scheduler.step()
            pbar.close()

        torch.save(self.gpt.state_dict(), checkpoint_path)

    def train_step(self, config, optimizer, scheduler):
        """Training step"""
        # Generate training samples
        train_samples = self.model_family.batch_sample(
            **config["model_family_config"],
            **config["train_sample_config"],
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
            link_fun=ddm_link_fun2()
        )

        # Adapt for network
        adapted = self.adapter.adapt(
            train_samples,
            intrinsic_params=self.model_family.intrinsic_params
        )

        # Optimizer
        optimizer.zero_grad()

        # Train
        pred_velocity, target_velocity = self.gpt(
            adapted["param_matrices"][..., None],
            adapted['input_data'],
            adapted['param_indices'],
            adapted['regressor_indices'],
            adapted['param_masks'],
        )

        # Compute loss
        L = self.gpt.compute_loss(pred_velocity, target_velocity, adapted['param_masks'])
        L.backward()

        if config["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                self.gpt.parameters(),
                config["gradient_clip_norm"],
            )

        # Update loss and learning rate
        optimizer.step()
        loss = L.detach().item()
        current_lr = scheduler.get_last_lr()[0]

        return loss, current_lr


    def val_step(self, config, global_step):
        # Generate training samples
        design_config = {
            '1': ["v", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "a", "tau", "s_v"],
            "u_2": ["v", "a", "tau"],
            "u_1:u_2": ["v", "a"],
        }

        test_samples = self.model_family.batch_sample(
            **config["model_family_config"],
            **config["val_sample_config"],
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
            design_config=design_config,
            link_fun=ddm_link_fun2()
        )

        # Adapt
        adapted = self.adapter.adapt(
            test_samples,
            intrinsic_params=self.model_family.intrinsic_params
        )

        self.gpt.eval()
        pred_velocity, target_velocity = self.gpt(
            adapted["param_matrices"][..., None],
            adapted['input_data'],
            adapted['param_indices'],
            adapted['regressor_indices'],
            adapted['param_masks']
        )
        if self.debug:
            print(pred_velocity.shape, target_velocity.shape)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        print(true_set.shape)

        params = ["v", "a", "tau", "s_v", "s_tau"]
        param_names = [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]
        params_mask = adapted["param_masks"].detach().cpu().numpy()
        n_cols = len(params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(config["batch_size"], n_rows, n_cols)

        fm_sample_steps = config["fm_sample_steps"]
        fm_num_samples = config["fm_num_samples"]
        pred_set = self.gpt.sample(
            adapted['input_data'],
            adapted['param_indices'],
            adapted['regressor_indices'],
            adapted['param_masks'],
            steps=fm_sample_steps,
            num_samples=fm_num_samples
        )
        print(pred_set.shape)
        pred_set = pred_set.reshape(config["batch_size"], fm_num_samples, n_rows, n_cols)
        if self.debug:
            print(pred_set.shape)

        params_mask = params_mask.reshape((config["batch_size"], n_rows, n_cols))[0]


        colors = bayesgpt_fm_colors()
        recovery_fig = adaptive_recovery(
            true_set, pred_set,
            design_config=design_config,
            intrinsic_params=params,
            max_num_categories=config["model_family_config"]["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=param_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )

        figures_dir = Path("./bayesgpt/experiments/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        recovery_fig.savefig(figures_dir / "ddm_family_gpt_fm_test_recovery_2.pdf", bbox_inches="tight")

        if self.use_wandb:
            wandb.log(
                {
                    "val/recovery": wandb.Image(recovery_fig),
                    # "val/correlation": wandb.Image(correlation_fig)
                },
                step=global_step,
            )
            plt.close(recovery_fig)
            # plt.close(correlation_fig)

        self.gpt.train()


    @staticmethod
    def finish():
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--num_obs", type=int, default=500, help="number of observations")
    parser.add_argument("--min_num_obs", type=int, default=200, help="minimum number of observations")
    parser.add_argument("--max_num_obs", type=int, default=500, help="maximum number of observations")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="number of steps per epoch")
    parser.add_argument("--fm_sample_steps", type=int, default=200, help="number of fm sample steps")
    parser.add_argument("--fm_num_samples", type=int, default=200, help="number of seeds")
    parser.add_argument("--train_batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--val_batch_size", type=int, default=200, help="validation batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb")
    parser.add_argument("--encoder_num_layers", type=int, default=3, help="number of encoder layers")
    parser.add_argument("--decoder_num_layers", type=int, default=3, help="number of decoder layers")
    parser.add_argument("--num_seeds", type=int, default=40, help="number of seeds")
    parser.add_argument("--seed_dim", type=int, default=64, help="dimension of seeds")
    parser.add_argument("--projection_dim", type=int, default=256, help="dimension of projection layers")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout rate")
    parser.add_argument("--layer_dropout", type=float, default=0.05, help="layer dropout rate")
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
        "free_intrinsics": ["v", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {}
    }

    val_params_kwargs = {
        "free_intrinsics": ["v", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"],
        "fixed_values": {"s_v": 0.0, "s_tau": 0.0},
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
        "free_params": val_params_kwargs["free_intrinsics"],
        "fixed_params": val_params_kwargs["fixed_intrinsics"],
        "fixed_values": val_params_kwargs["fixed_values"],
    }

    wandb_config = {
        "project_name": "bayesgpt-fm",
        "run_name": None,
        "tags": ["BayesGPT", "ModelFamily"],
        "watch_log": "gradients",
        "watch_freq": 200
    }

    bayesgpt_config = {
        "encoder_input_dim": encoder_input_dim,
        "encoder_num_layers": args.encoder_num_layers,
        "decoder_num_layers": args.decoder_num_layers,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "num_seeds": args.num_seeds,
        "seed_dim": args.seed_dim,
        "proj_dim": args.projection_dim,
        "dropout": args.dropout,
        "layer_dropout": args.layer_dropout,
    }

    logging.info(
        f"Training with "
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
                **{"bayesgpt": bayesgpt_config},
            },
        )

    # Define model family, adapter, and network
    model_family = NestedModelFamily(
        model=DDM(),
        name="DDM",
        prior_fun=ddm_priors2(),
        mask_randomizer_kwargs=train_params_kwargs
    )
    adapter = Adapter()
    bayesgpt = BayesGPT(**bayesgpt_config).to(device).train()

    # Pass to trainer
    trainer = BayesGPTTrainer(
        model_family=model_family,
        adapter=adapter,
        gpt=bayesgpt,
        use_wandb=use_wandb,
    )

    # Define checkpoint path
    checkpoint_path = (
        f"./bayesgpt/experiments/checkpoints/"
        f"bayesgpt_fm"
        f"_obs{val_sample_config['num_obs']}"
        f"_eps{train_config['epochs']}"
        f"_stp{train_config['steps_per_epoch']}"
        f"_bse{train_config['batch_size']}"
        f"_nls{bayesgpt_config['decoder_num_layers']}"
        f"_nhs{bayesgpt_config['decoder_num_heads']}"
        f"_nss{bayesgpt_config['num_seeds']}_2.pt"
    )

    # Train
    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path
    )
    trainer.finish()
