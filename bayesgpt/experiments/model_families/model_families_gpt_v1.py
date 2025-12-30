import torch
import wandb
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

from playgrounds.testing_workflow import grad_clip_norm

np.set_printoptions(suppress=True)


from simulators import NestedModelFamily
from simulators.benchmarks import DDM
from simulators.benchmarks.ddms.ddm_priors import ddm_full_priors
from adapters import Adapter
from networks.transformers.gpt import BayesGPTv1


class BayesGPTTrainer:
    def __init__(
        self,
        model_family,
        adapter,
        net,
        train_config,
        wandb_config,
    ):
        super().__init__()
        self.model_family = model_family
        self.adapter = adapter
        self.net = net

    def train(self, train_config: dict):
        optimizer = Adam(self.net.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])
        loss_fn = torch.nn.MSELoss()

        for epoch in range(train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {(epoch + 1)}/{train_config['epochs']}",
                miniters=100,
            )
            for step in range(train_config["steps_per_epoch"]):
                loss, current_lr = self.step(
                    train_config=train_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn
                )

                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            scheduler.step()
            pbar.close()



    def step(self, train_config, optimizer, scheduler, loss_fn):

        # Generate samples
        samples = self.model_family.batch_sample(
            batch_size=train_config.batch_size,
            mask_randomizer_kwargs=dict(
                free_intrinsics={"v", "a", "tau", "s_v"},
                fixed_intrinsics={"s_tau"}
            ),
            num_obs = 500,
            flatten_param_outputs=True
        )

        # Adapt for network
        adapted = self.adapter.adapt(
            samples,
            intrinsic_params=self.model_family.intrinsic_params
        )

        # Optimizer
        optimizer.zero_grad()

        # Train
        mu, logvar = self.net(
            adapted['input_data'],
            adapted['param_indices'],
            adapted['regressor_indices'],
            adapted['param_masks'],
        )

        # Compute loss
        L = loss_fn(adapted["param_matrices"], mu, logvar, adapted["param_masks"])
        L.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(),
                train_config["gradient_clip_norm"],
            )

        optimizer.step()

        loss = L.detach().item()
        current_lr = scheduler.get_last_lr()[0]

        return loss, current_lr


    def validate(self):
        pass


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_config = {
        "epochs": 100,
        "batch_size": 32,
        "steps_per_epoch": 100,
        "learning_rate": 2e-4,
        "gradient_clip_norm": 5.0,
        "device": device
    }

    wandb_config = {
        "project_name": "bayesgpt-v1",
        "run_name": None,
        "tags": ["BayesGPTv1", "ModelFamily"],
        "watch_log": "gradients",
        "watch_freq": 200
    }

    bayesgpt_config = {
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

    bayesgpt = BayesGPTv1(**bayesgpt_config)

    trainer = BayesGPTTrainer(
        train_config=train_config,
        wandb_config=wandb_config,
        net = bayesgpt
    )
