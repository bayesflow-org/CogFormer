import torch
import wandb
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
np.set_printoptions(suppress=True)


from simulators import NestedModelFamily
from simulators.benchmarks import DDM
from simulators.benchmarks.ddms.ddm_priors import ddm_full_priors
from adapters import Adapter
from networks.transformers.gpt import BayesGPTv1


class BayesGPTTrainer:
    def __init__(
        self,
        wandb_config,
        train_config,
        net
    ):
        super().__init__()
        self.train_config = train_config
        self.wandb_config = wandb_config
        self.net = net

    def train(self):
        pass

    def step(self, model_family):
        samples = model_family.batch_sample(
            batch_size=train_config.batch_size,
            mask_randomizer_kwargs=dict(
                free_intrinsics={"v", "a", "tau", "s_v", "decay"},
                fixed_intrinsics={"s_tau"}
            ),
            num_obs = 500,
            flatten_param_outputs=True
        )

    def validate(self):
        pass


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
