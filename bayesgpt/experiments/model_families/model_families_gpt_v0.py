import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import wandb

class BayesGPTTrainer:
    """Minimal end-to-end trainer with optional Weights & Biases logging."""

    def __init__(
        self,
        model_family,                 # e.g., NestedModelFamily(name="DDM", model=DDM(), prior_fun=ddm_priors)
        adapter,                      # e.g., Adapter()
        net_cls,                      # e.g., BayesGPTv1
        net_kwargs: dict | None = None,
        device: torch.device | None = None,
        # training
        batch_size: int = 32,
        epochs: int = 200,
        steps_per_epoch: int = 500,
        learning_rate: float = 2e-4,
        grad_clip_norm: float | None = 5.0,
        # sampling
        sample_kwargs: dict | None = None,    # passed to model_family.batch_sample(...)
        # wandb
        use_wandb: bool = True,
        wandb_project: str = "sbi-bayesgpt",
        wandb_run_name: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_watch_log: str = "gradients",   # "all" | "gradients" | "parameters" | None
        wandb_watch_freq: int = 200,
    ):
        self.model_family = model_family
        self.adapter = adapter
        self.intrinsic_params = getattr(model_family, "intrinsic_params", [])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # network
        net_kwargs = net_kwargs or dict(encoder_num_layers=4, decoder_num_layers=4, seed_dim=64, num_seeds=10)
        self.model = net_cls(**net_kwargs).to(self.device).train()

        # optim/sched
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.grad_clip_norm = grad_clip_norm

        # training params
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # sampling cfg
        self.sample_kwargs = sample_kwargs or dict(
            batch_size=batch_size,
            mask_randomizer_kwargs=dict(
                free_intrinsics={"v", "a", "tau", "s_v", "decay"},
                fixed_intrinsics={"s_tau"},
            ),
            num_obs=200,
            flatten_param_outputs=True,
        )

        # wandb
        self.use_wandb = use_wandb
        self.wandb_run = None
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"bayesgpt-{model_family.name}-{int(time.time())}",
                tags=(wandb_tags or []) + [model_family.name, type(self.model).__name__],
                config=dict(
                    device=str(self.device),
                    batch_size=batch_size,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    learning_rate=learning_rate,
                    grad_clip_norm=grad_clip_norm,
                    net_kwargs=net_kwargs,
                    sample_kwargs=self.sample_kwargs,
                ),
            )
            if wandb_watch_log:
                wandb.watch(self.model, log=wandb_watch_log, log_freq=wandb_watch_freq)

    def _step(self, global_step: int, epoch_idx: int):
        samples = self.model_family.batch_sample(**self.sample_kwargs)
        adapted = self.adapter.adapt(samples, intrinsic_params=self.intrinsic_params, device=self.device)

        self.optimizer.zero_grad()

        mu, log_var = self.model(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
        )
        loss = self.model.compute_loss(adapted["param_matrices"], mu, log_var, adapted["param_masks"])
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        # logging
        loss_val = float(loss.detach().item())
        lr = self.scheduler.get_last_lr()[0]

        if self.use_wandb:
            wandb.log(
                {"train/loss": loss_val, "train/lr": lr, "train/epoch": epoch_idx + 1, "train/step": global_step},
                step=global_step,
            )

        return loss_val, lr

    def train(self, checkpoint_path: str = "bayesgpt_model.pt"):
        global_step = 0
        for ep in range(self.epochs):
            pbar = tqdm(total=self.steps_per_epoch, desc=f"Epoch {ep + 1}/{self.epochs}", miniters=100)
            for _ in range(self.steps_per_epoch):
                global_step += 1
                loss_val, lr = self._step(global_step, ep)
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")
                pbar.update(1)
            pbar.close()
            self.scheduler.step()

            if self.use_wandb:
                wandb.log({"epoch_end/epoch": ep + 1}, step=global_step)

        # save + upload
        torch.save(self.model.state_dict(), checkpoint_path)
        if self.use_wandb:
            wandb.save(checkpoint_path)

    def finish(self):
        if self.use_wandb and self.wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    import numpy as np
    import torch
    from simulators import NestedModelFamily
    from simulators.benchmarks import DDM
    from adapters import Adapter
    from networks.transformers.gpt import BayesGPTv1

    ddm_priors = {
        "v": {"intercept": lambda: np.random.gamma(3.0, 0.8), "slope": lambda: np.random.normal(0.0, 3.0)},
        "a": {"intercept": lambda: np.random.gamma(10.0, 0.3), "slope": lambda: np.random.normal(0.0, 1.0)},
        "tau": {"intercept": lambda: np.random.gamma(3.0, 0.2), "slope": lambda: 0.0},
        "s_v": {"intercept": lambda: np.random.gamma(1.0, 0.2), "slope": lambda: 0.0},
        "s_tau": {"intercept": lambda: np.random.uniform(0.0, 0.4), "slope": lambda: 0.0},
        "decay": {"intercept": lambda: np.random.gamma(1.0, 0.4), "slope": lambda: 0.0},
    }

    model_family = NestedModelFamily(name="DDM", model=DDM(), prior_fun=ddm_priors)
    adapter = Adapter()

    trainer = BayesGPTTrainer(
        model_family=model_family,
        adapter=adapter,
        net_cls=BayesGPTv1,
        net_kwargs=dict(encoder_num_layers=4, decoder_num_layers=4, seed_dim=64, num_seeds=10),
        batch_size=32,
        epochs=200,
        steps_per_epoch=500,
        learning_rate=2e-4,
        grad_clip_norm=5.0,
        # sample_kwargs can override defaults if you want
        use_wandb=True,
        wandb_project="bayesgpt-testing",
        wandb_run_name=None,
        wandb_tags=["DDM", "BayesGPTv1"],
        wandb_watch_log="gradients",
        wandb_watch_freq=200,
    )

    trainer.train(checkpoint_path="bayesgpt_ddm.pt")
    trainer.finish()
