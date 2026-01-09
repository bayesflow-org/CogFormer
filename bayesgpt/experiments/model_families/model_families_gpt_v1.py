import torch
import wandb
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from simulators import NestedModelFamily
from simulators.benchmarks import DDM
from simulators.benchmarks.ddms.ddm_priors import ddm_full_priors
from adapters import Adapter
from networks.transformers.gpt import BayesGPTv1
from networks.loss import mse_loss, nll_loss
from diagnostics.plot.recovery import recovery
from diagnostics.plot.matrix_recovery import matrix_recovery
from diagnostics.plot.correlation import correlation


class BayesGPTTrainer:
    def __init__(
        self,
        model_family,
        adapter,
        gpt
    ):
        super().__init__()
        self.model_family = model_family
        self.adapter = adapter
        self.gpt = gpt

    def train(self, train_config, val_config, checkpoint_path="bayesgpt_model.pt"):
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
                    scheduler=scheduler,
                    loss_fn=loss_fn
                )
                # Log metrics
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
                self.validate(val_config, global_step)

            scheduler.step()
            pbar.close()

        torch.save(self.gpt.state_dict(), checkpoint_path)

    def train_step(self, config, optimizer, scheduler, loss_fn):
        """Training step"""
        # Generate training samples
        train_samples = self.model_family.batch_sample(
            **config["sim_config"],
            batch_size=config["batch_size"],
            mask_randomizer_kwargs=dict(
                free_intrinsics={"v", "a", "tau", "s_v"},
                fixed_intrinsics={"s_tau"}
            ),
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
        mu, logvar = self.gpt(
            adapted['input_data'],
            adapted['param_indices'],
            adapted['regressor_indices'],
            adapted['param_masks'],
        )

        # Compute loss
        L = loss_fn(adapted["param_matrices"], mu, logvar, adapted["param_masks"])
        L.backward()

        if train_config["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                self.gpt.parameters(),
                config["gradient_clip_norm"],
            )

        # Update loss and learning rate
        optimizer.step()
        loss = L.detach().item()
        current_lr = scheduler.get_last_lr()[0]

        return loss, current_lr


    def validate(self, config, global_step):
        # Generate training samples
        test_samples = self.model_family.batch_sample(
            **config["sim_config"],
            batch_size=config["batch_size"],
            mask_randomizer_kwargs=dict(
                free_intrinsics={"v", "a", "tau", "s_v"},
                fixed_intrinsics={"s_tau"}
            ),
            flatten_param_outputs=True
        )

        max_num_categories = test_samples["max_num_categories"]

        # Adapt
        adapted = self.adapter.adapt(
            test_samples,
            intrinsic_params=self.model_family.intrinsic_params
        )

        # Evaluate with test set
        self.gpt.eval()
        mu, logvar = self.gpt(
            adapted['input_data'],
            adapted['param_indices'],
            adapted['regressor_indices'],
            adapted['param_masks']
        )

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        pred_set = mu.detach().cpu().numpy()[:,:,0]

        params = ["v", "a", "tau", "s_v", "s_tau"]
        n_cols = len(params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(config["batch_size"], n_rows, n_cols)
        pred_set = pred_set.reshape(config["batch_size"], n_rows, n_cols)

        # Log recovery plot
        # fig = recovery(true_set, pred_set, params=["v", "a", "tau", "s_v", "s_tau"])
        free_params = ["v", "a", "tau", "s_v"]
        fixed_params = ["s_tau"]
        recovery_fig = matrix_recovery(true_set, pred_set, free_params=free_params, fixed_params=fixed_params)
        correlation_fig = correlation(true_set, pred_set, free_params=free_params, fixed_params=fixed_params)
        wandb.log(
            {
                "val/recovery": wandb.Image(recovery_fig),
                "val/correlation": wandb.Image(correlation_fig)
            },
            step=global_step,
        )
        plt.close(recovery_fig)
        plt.close(correlation_fig)
        self.gpt.train()

    @staticmethod
    def finish():
        wandb.finish()


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    max_num_regressors = 3
    max_num_categories = 3
    keep_intercept = True
    num_obs = 500

    # Automate input dim
    # input_dim = regressors * (categories - 1) + intercept + sim_data_dim (RTs, choices --> 2)
    encoder_input_dim = max_num_regressors * (max_num_categories - 1) + (3 if keep_intercept else 2)

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "num_obs": num_obs,
    }

    train_config = {
        "epochs": 50,
        "batch_size": 32,
        "steps_per_epoch": 50,
        "learning_rate": 2e-4,
        "gradient_clip_norm": 5.0,
        "device": device,
        "sim_config": model_family_config,
    }

    val_config = {
        "batch_size": 300,
        "sim_config": model_family_config,
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
        prior_fun=ddm_full_priors()
    )
    adapter = Adapter()
    bayesgpt = BayesGPTv1(**bayesgpt_config).to(device).train()

    # Pass to trainer
    trainer = BayesGPTTrainer(
        model_family=model_family,
        adapter=adapter,
        gpt=bayesgpt
    )

    # Define checkpoint path
    checkpoint_path = (f"bayesgpt"
                       f"_e{train_config['epochs']}"
                       f"_bs{train_config['batch_size']}"
                       f"_l{bayesgpt_config['decoder_num_layers']}"
                       f"_h{bayesgpt_config['encoder_num_heads']}"
                       f"_s{bayesgpt_config['num_seeds']}.pt")
    # Train
    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path
    )
    trainer.finish()
