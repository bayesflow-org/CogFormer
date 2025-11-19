import time
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from networks.loss import mse_loss



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
        epochs: int = 100,
        steps_per_epoch: int = 100,
        learning_rate: float = 2e-4,
        grad_clip_norm: float | None = 5.0,
        # sampling
        sample_kwargs: dict | None = None,    # passed to model_family.batch_sample(...)
        # validation
        do_validation: bool = True,
        val_every: int = 1,  # run validation every N epochs
        val_use_fixed: bool = True,  # fixed held-out set vs re-sample
        val_sample_kwargs: dict | None = None,  # overrides for val sampling
        # wandb
        use_wandb: bool = True,
        wandb_project: str = "bayesgpt-testing",
        wandb_run_name: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_watch_log: str = "gradients",   # "all" | "gradients" | "parameters" | None
        wandb_watch_freq: int = 200,
        # recovery plot
        log_scatter_every: int = 10,
        scatter_max_points: int = 5000,
        scatter_figsize_scale: float = 3.0,
    ):
        self.model_family = model_family
        self.adapter = adapter
        self.intrinsic_params = getattr(model_family, "intrinsic_params", [])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # network
        self.net_kwargs = net_kwargs
        self.model = net_cls(**net_kwargs).to(self.device).train()

        # optim/sched
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.grad_clip_norm = grad_clip_norm

        # training params
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # Validation
        self.do_validation = do_validation
        self.val_every = val_every
        self.val_use_fixed = val_use_fixed

        # sampling cfg
        self.sample_kwargs = sample_kwargs or dict(
            batch_size=batch_size,
            mask_randomizer_kwargs=dict(
                free_intrinsics={"v", "a", "tau", "s_v", "s_tau"},
                fixed_intrinsics={},
            ),
            num_obs=500,
            flatten_param_outputs=True,
        )

        # default val sampling cfg (smaller batch OK)
        self.val_sample_kwargs = val_sample_kwargs or dict(
            batch_size=300,
            mask_randomizer_kwargs=dict(
                free_intrinsics={"v", "a", "tau", "s_v", "s_tau"},
                fixed_intrinsics={},
            ),
            num_obs=500,
            flatten_param_outputs=True,
        )

        # optional fixed validation set (synthetic but held-out)
        self._fixed_val_adapted = None
        if self.do_validation and self.val_use_fixed:
            with torch.no_grad():
                val_samples = self.model_family.batch_sample(**self.val_sample_kwargs)
                self._fixed_val_adapted = self.adapter.adapt(
                    val_samples, intrinsic_params=self.intrinsic_params, device=self.device
                )

        self.log_scatter_every = log_scatter_every
        self.scatter_max_points = scatter_max_points
        self.scatter_figsize_scale = scatter_figsize_scale

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

    def step(self, global_step: int, epoch_idx: int):
        samples = self.model_family.batch_sample(**self.sample_kwargs)
        adapted = self.adapter.adapt(samples, intrinsic_params=self.intrinsic_params, device=self.device)

        self.optimizer.zero_grad()

        mu, log_var = self.model(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
        )
        loss = mse_loss(adapted["param_matrices"], mu, log_var, adapted["param_masks"])
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
                loss_val, lr = self.step(global_step, ep)
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")
                pbar.update(1)
            pbar.close()
            self.scheduler.step()

            if self.do_validation and ((ep + 1) % self.val_every == 0):
                self._validate(epoch_idx=ep, global_step=global_step)

            if self.use_wandb:
                wandb.log({"epoch_end/epoch": ep + 1}, step=global_step)

        torch.save(self.model.state_dict(), checkpoint_path)
        if self.use_wandb:
            wandb.save(checkpoint_path)

    def _validate(self, epoch_idx: int, global_step: int):
        if not self.do_validation:
            return

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            if self.val_use_fixed and (self._fixed_val_adapted is not None):
                adapted = self._fixed_val_adapted
            else:
                val_samples = self.model_family.batch_sample(**self.val_sample_kwargs)
                adapted = self.adapter.adapt(val_samples, intrinsic_params=self.intrinsic_params, device=self.device)

            mu, log_var = self.model(
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
            )

            # --- normalize shapes to (B, D) ---
            def squeeze2d(t: torch.Tensor) -> torch.Tensor:
                # remove trailing singleton if network returns (B, D, 1)
                return t.squeeze(-1) if (t.ndim == 3 and t.shape[-1] == 1) else t

            mu = squeeze2d(mu).to(torch.float32)
            true_params = squeeze2d(adapted["param_matrices"]).to(torch.float32)
            mask = squeeze2d(adapted["param_masks"]).to(torch.float32)

            B = mu.shape[0]
            D = mu.shape[1] if mu.ndim >= 2 else mu.numel() // B

            def align_bd(x: torch.Tensor) -> torch.Tensor:
                # Expect (B, D). Fix common issues: (D, B), (B, B), flat, etc.
                if x.ndim == 1:  # flattened
                    if x.numel() == B * D:
                        return x.view(B, D)
                if x.ndim == 2:
                    if x.shape == (D, B):
                        return x.t()
                    if x.shape == (B, B) and B * B == B * D:
                        # extremely unlikely unless D == B; keep as is otherwise
                        pass
                    if x.shape[0] == B and x.shape[1] != D and (x.numel() == B * D):
                        return x.reshape(B, D)
                    if x.shape[1] == B and x.shape[0] != D and (x.numel() == B * D):
                        return x.t().reshape(B, D)
                if x.ndim > 2 and x.numel() == B * D:
                    return x.view(B, D)
                return x  # assume already (B, D)

            true_params = align_bd(true_params)
            mask = align_bd(mask)

            # final sanity check (raise a clear error if still off)
            assert mu.shape == true_params.shape == mask.shape, (
                f"Shape mismatch after alignment: mu{mu.shape}, true{true_params.shape}, mask{mask.shape}"
            )

            diff = (mu - true_params) * mask
            denom = mask.sum().clamp_min(1.0)
            val_mse = (diff.pow(2).sum() / denom).item()

            y = (true_params * mask).view(-1)
            yhat = (mu * mask).view(-1)
            m = (mask.view(-1) > 0.5)
            if m.any():
                y = y[m]
                yhat = yhat[m]
                y_mean = y.mean()
                yhat_mean = yhat.mean()
                num = ((y - y_mean) * (yhat - yhat_mean)).sum()
                den = (y - y_mean).pow(2).sum().sqrt() * (yhat - yhat_mean).pow(2).sum().sqrt()
                val_pearson = (num / den.clamp_min(1e-12)).item()
            else:
                val_pearson = float("nan")

            if self.use_wandb:
                # log metrics + debug shapes for traceability
                wandb.log(
                    {
                        "val/masked_mse": val_mse,
                        "val/pearson": val_pearson,
                        "val/epoch": epoch_idx + 1,
                        "val/debug/mu_shape": str(tuple(mu.shape)),
                        "val/debug/true_shape": str(tuple(true_params.shape)),
                        "val/debug/mask_shape": str(tuple(mask.shape)),
                    },
                    step=global_step,
                )

            # recovery scatter (periodic)
            if self.use_wandb and self.log_scatter_every and ((epoch_idx + 1) % self.log_scatter_every == 0):
                self._log_recovery_plot(mu, true_params, mask, epoch_idx, global_step)

        if was_training:
            self.model.train()

    def _log_recovery_plot(
        self,
        mu: torch.Tensor,
        true_params: torch.Tensor,
        mask: torch.Tensor,
        epoch_idx: int,
        global_step: int,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        # (B, D) — squeeze optional trailing dim
        def squeeze2d(t):
            return t.squeeze(-1) if t.ndim == 3 and t.shape[-1] == 1 else t
        mu = squeeze2d(mu).detach().cpu()
        true_params = squeeze2d(true_params).detach().cpu()
        mask = squeeze2d(mask).detach().cpu()

        B, D = mu.shape
        num_intrinsic_params = len(self.intrinsic_params)  # number of intrinsic parameters

        # Build per-parameter index slices: [i, i+P, i+2P, ...]
        per_param_indices = [np.arange(i, D, num_intrinsic_params) for i in range(num_intrinsic_params)]

        fig, axes = plt.subplots(
            1, num_intrinsic_params, figsize=(self.scatter_figsize_scale * num_intrinsic_params, self.scatter_figsize_scale),
            squeeze=False
        )
        axes = axes[0]

        for i, ax in enumerate(axes):
            cols = per_param_indices[i]
            y_true = true_params[:, cols]
            y_pred = mu[:, cols]
            m = mask[:, cols] > 0.5  # only real (non-padded) entries

            if m.sum().item() == 0:
                ax.set_title(f"{self.intrinsic_params[i]} (no data)")
                ax.axis("off")
                continue

            # collect masked values
            yt = y_true[m]
            yp = y_pred[m]

            # subsample for readability
            if yt.numel() > self.scatter_max_points:
                idx = torch.randperm(yt.numel())[: self.scatter_max_points]
                yt = yt[idx]
                yp = yp[idx]

            yt_np = yt.numpy()
            yp_np = yp.numpy()

            # scatter
            ax.scatter(yt_np, yp_np, s=6, alpha=0.6)

            # quadratic axes + y=x line
            lo = float(min(yt_np.min(), yp_np.min()))
            hi = float(max(yt_np.max(), yp_np.max()))
            eps = (hi - lo) * 0.1 if hi > lo else 1.0
            ax.set_xlim(lo - eps, hi + eps)
            ax.set_ylim(lo - eps, hi + eps)
            ax.plot([lo - eps, hi + eps], [lo - eps, hi + eps], linestyle="--", linewidth=1.0)

            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
            ax.set_xlabel("Ground Truth")
            if i == 0:
                ax.set_ylabel("Estimation")
            ax.set_title(self.intrinsic_params[i])

        fig.tight_layout()

        if self.use_wandb:
            wandb.log(
            {
                "fig/recovery": wandb.Image(fig),
                "fig/epoch": epoch_idx + 1
            },
                step=global_step,
            )
        plt.close(fig)

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

    ddm_log_priors = {
        "v": {"intercept": lambda: np.random.normal(1., 1.),
              "slope": lambda: 0.0},
        "a": {"intercept": lambda: np.random.normal(-0.1, 0.3),
              "slope": lambda: 0.0},
        "tau": {"intercept": lambda: np.random.normal(-1.5, 0.3),
                "slope": lambda: 0.0},
        "s_v": {"intercept": lambda: np.random.normal(-1.2, 0.5),
                "slope": lambda: 0.0},
        "s_tau": {"intercept": lambda: np.random.beta(1.0, 3.0),
                  "slope": lambda: 0.0}
    }

    net_kwargs = {
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

    model_family = NestedModelFamily(name="DDM", model=DDM(), prior_fun=ddm_log_priors)
    adapter = Adapter()

    trainer = BayesGPTTrainer(
        model_family=model_family,
        adapter=adapter,
        net_cls=BayesGPTv1,
        net_kwargs=net_kwargs,
        batch_size=32,
        epochs=500,
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

    trainer.train(checkpoint_path="bayesgpt_ddm_8l_8h_40s_100.pt")
    trainer.finish()
