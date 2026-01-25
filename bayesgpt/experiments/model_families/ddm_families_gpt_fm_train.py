import torch
import wandb
from pathlib import Path

from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from bayesgpt.simulators import NestedModelFamily
from bayesgpt.simulators.benchmarks import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_full_priors, ddm_baseline_priors
from bayesgpt.adapters import Adapter
from bayesgpt.networks.transformers.gpt import BayesGPT
from bayesgpt.networks.loss import mse_loss, nll_loss
from bayesgpt.diagnostics.plot.adaptive_recovery import adaptive_recovery


class BayesGPTTrainer:
    def __init__(
        self,
        gpt,
        model=None,
        prior_fun=None,
        model_family=None,
        adapter=None,
        use_wandb=False,
    ):
        super().__init__()
        self.gpt=gpt
        self.model=model
        self.prior_fun=prior_fun
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

    def train(self, train_config, val_config, checkpoint_path="bayesgpt_vi.pt"):
        # Define global step
        global_step = 0

        # Define optimizer, scheduler, and loss function from config
        optimizer = AdamW(self.gpt.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])
        loss_fn = nll_loss

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

            if (epoch + 1) % 5 == 0:
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
            flatten_param_outputs=True
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


    def val_step(self, val_config, global_step):
        pass

    @staticmethod
    def finish():
        wandb.finish()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_wandb = True

    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True
    num_obs = 500

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "num_obs": num_obs,
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
        "fixed_config": False,
    }

    val_sample_config = {
        "mask_randomizer_kwargs": val_params_kwargs,
        "min_num_regressors": 2,
        "fixed_config": False
    }

    # Automate input dim
    # input_dim = regressors * (categories - 1) + intercept + sim_data_dim (RTs, choices --> 2)
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    encoder_input_dim = max_total_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)

    train_config = {
        "epochs": 10,
        "batch_size": 32,
        "steps_per_epoch": 10,
        "learning_rate": 2e-4,
        "gradient_clip_norm": 5.0,
        "device": device,
        "model_family_config": model_family_config,
        "train_sample_config": train_sample_config,
    }

    val_config = {
        "batch_size": 30,
        "model_family_config": model_family_config,
        "val_sample_config": val_sample_config,
        "free_params": val_params_kwargs["free_intrinsics"],
        "fixed_params": val_params_kwargs["fixed_intrinsics"],
        "fixed_values": val_params_kwargs["fixed_values"],
    }

    wandb_config = {
        "project_name": "bayesgpt-v1",
        "run_name": None,
        "tags": ["BayesGPTv1", "ModelFamily"],
        "watch_log": "gradients",
        "watch_freq": 200
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
        prior_fun=ddm_baseline_priors(),
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
        f"bayesgpt"
        f"_e{train_config['epochs']}"
        f"_b{train_config['batch_size']}"
        f"_l{bayesgpt_config['decoder_num_layers']}"
        f"_h{bayesgpt_config['encoder_num_heads']}"
        f"_s{bayesgpt_config['num_seeds']}.pt"
    )

    # Train
    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path
    )
    trainer.finish()
