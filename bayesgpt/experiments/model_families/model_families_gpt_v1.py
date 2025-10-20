import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm.auto import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import wandb

from simulators import NestedModelFamily
from simulators.benchmarks import DDM
from adapters import Adapter
from networks.transformers.gpt import BayesGPTv1


if __name__ == "__main__":

    ddm_priors = {
        "v": {"intercept": lambda: np.random.gamma(3.0, 0.8),
              "slope": lambda: np.random.normal(0.0, 3.0)},
        "a": {"intercept": lambda: np.random.gamma(10.0, 0.3),
              "slope": lambda: np.random.normal(0.0, 1.0)},
        "tau": {"intercept": lambda: np.random.gamma(3.0, 0.2),
                "slope": lambda: 0.0},
        "s_v": {"intercept": lambda: np.random.gamma(1.0, 0.2),
                "slope": lambda: 0.0},
        "s_tau": {"intercept": lambda: np.random.uniform(0.0, 0.4),
                  "slope": lambda: 0.0},
        "decay": {"intercept": lambda: np.random.gamma(1.0, 0.4),
                  "slope": lambda: 0.0},
    }

    model_family = NestedModelFamily(name="DDM", model=DDM(), prior_fun=ddm_priors)
    ddm_intrinsics = model_family.intrinsic_params
    adapter = Adapter()

    grad_clip_norm = 5.
    batch_size = 32
    epochs = 500
    steps_per_epoch = 500
    learning_rate = 2e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bayesgpt = BayesGPTv1(encoder_num_layers=4, decoder_num_layers=4, seed_dim=64, num_seeds=10)
    bayesgpt.to(device)
    bayesgpt.train()

    optimizer = Adam(bayesgpt.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(epochs):
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {ep + 1}/{epochs}", miniters=100)

        for step in range(steps_per_epoch):

            samples = model_family.batch_sample(
                batch_size=batch_size,
                mask_randomizer_kwargs=dict(
                    free_intrinsics={"v", "a", "tau", "s_v", "decay"},
                    fixed_intrinsics={"s_tau"}
                ),
                num_obs=200,
                flatten_param_outputs=True
            )

            adapted = adapter.adapt(samples, intrinsic_params=ddm_intrinsics)

            optimizer.zero_grad()

            mu, log_var = bayesgpt(
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"]
            )

            L = bayesgpt.compute_loss(adapted["param_matrices"], mu, log_var, adapted["param_masks"])
            L.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(bayesgpt.parameters(), grad_clip_norm)

            optimizer.step()

            loss_val = L.detach().item()
            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")
            pbar.update(1)

        scheduler.step()
        pbar.close()
