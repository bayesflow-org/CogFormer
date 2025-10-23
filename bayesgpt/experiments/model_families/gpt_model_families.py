import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import wandb
import time
from typing import Dict, Any, Optional, Type


class BayesGPTTrainer:
    """Clean trainer with intuitive kwargs-based configuration."""

    def __init__(
            self,
            model_family,  # e.g., NestedModelFamily
            adapter,  # e.g., Adapter
            **kwargs  # Everything else as kwargs!
    ):
        # Core components
        self.model_family = model_family
        self.adapter = adapter
        self.intrinsic_params = getattr(model_family, "intrinsic_params", [])

        # Parse kwargs into logical groups
        self.model_cfg = self._parse_model_config(kwargs)
        self.training_cfg = self._parse_training_config(kwargs)
        self.sampling_cfg = self._parse_sampling_config(kwargs)
        self.wandb_cfg = self._parse_wandb_config(kwargs)
        self.checkpoint_cfg = self._parse_checkpoint_config(kwargs)

        # Validate required params were provided
        self._validate_config()

        # Initialize everything
        self._build_network()
        self._build_optimizer()
        self._init_wandb()

        # Training state
        self.global_step = 0
        self.epoch_losses = []
        self.grad_clip_norm = self.training_cfg.get('grad_clip_norm', 5.0)

    def _parse_model_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model config from kwargs."""
        model_keys = {'net_cls', 'net_kwargs', 'device'}
        model_cfg = {k: v for k, v in kwargs.items() if k in model_keys}

        # Set defaults
        model_cfg.setdefault('net_kwargs', {})
        model_cfg['net_kwargs'].setdefault('encoder_num_layers', 4)
        model_cfg['net_kwargs'].setdefault('decoder_num_layers', 4)
        model_cfg['net_kwargs'].setdefault('seed_dim', 64)
        model_cfg['net_kwargs'].setdefault('num_seeds', 10)
        model_cfg.setdefault('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        return model_cfg

    def _parse_training_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training config from kwargs."""
        training_keys = {'batch_size', 'epochs', 'steps_per_epoch', 'learning_rate', 'grad_clip_norm'}
        training_cfg = {k: v for k, v in kwargs.items() if k in training_keys}

        # Set defaults
        training_cfg.setdefault('batch_size', 32)
        training_cfg.setdefault('epochs', 200)
        training_cfg.setdefault('steps_per_epoch', 500)
        training_cfg.setdefault('learning_rate', 2e-4)

        if training_cfg['steps_per_epoch'] * training_cfg['batch_size'] < 1000:
            print(
                f"⚠️  Small training set: {training_cfg['steps_per_epoch'] * training_cfg['batch_size']} samples/epoch")

        return training_cfg

    def _parse_sampling_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sampling config from kwargs."""
        sampling_keys = {'sample_batch_size', 'mask_randomizer_kwargs', 'num_obs', 'flatten_param_outputs'}
        sampling_cfg = {k: v for k, v in kwargs.items() if k in sampling_keys}

        # Map sample_batch_size to batch_size for backward compatibility
        if 'sample_batch_size' in kwargs:
            sampling_cfg['batch_size'] = kwargs['sample_batch_size']
        else:
            sampling_cfg['batch_size'] = self.training_cfg['batch_size']

        sampling_cfg.setdefault('mask_randomizer_kwargs', {})
        sampling_cfg['mask_randomizer_kwargs'].setdefault('free_intrinsics', {"v", "a", "tau", "s_v", "decay"})
        sampling_cfg['mask_randomizer_kwargs'].setdefault('fixed_intrinsics', {"s_tau"})
        sampling_cfg.setdefault('num_obs', 200)
        sampling_cfg.setdefault('flatten_param_outputs', True)

        return sampling_cfg

    def _parse_wandb_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract W&B config from kwargs."""
        wandb_keys = {'use_wandb', 'wandb_project', 'wandb_run_name', 'wandb_tags', 'wandb_watch_log',
                      'wandb_watch_freq'}
        wandb_cfg = {k: v for k, v in kwargs.items() if k in wandb_keys}

        # Map old names to new
        if 'use_wandb' in kwargs:
            wandb_cfg['enabled'] = kwargs['use_wandb']
        else:
            wandb_cfg['enabled'] = True

        wandb_cfg.setdefault('project', "sbi-bayesgpt")
        wandb_cfg.setdefault('watch_log', "gradients")
        wandb_cfg.setdefault('watch_freq', 200)

        return wandb_cfg

    def _parse_checkpoint_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract checkpoint config from kwargs."""
        checkpoint_keys = {'checkpoint_path', 'save_checkpoints', 'checkpoint_freq'}
        checkpoint_cfg = {k: v for k, v in kwargs.items() if k in checkpoint_keys}

        checkpoint_cfg.setdefault('path', "bayesgpt_model.pt")
        checkpoint_cfg.setdefault('save_checkpoints', True)
        checkpoint_cfg.setdefault('checkpoint_freq', 1000)

        return checkpoint_cfg

    def _validate_config(self):
        """Ensure required config is present."""
        if 'net_cls' not in self.model_cfg:
            raise ValueError("net_cls is required in model config")

    def _build_network(self):
        """Build model and move to device."""
        self.model = self.model_cfg['net_cls'](**self.model_cfg['net_kwargs'])
        self.model = self.model.to(self.model_cfg['device']).train()

    def _build_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = Adam(self.model.parameters(), lr=self.training_cfg['learning_rate'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.training_cfg['epochs'])

    def _init_wandb(self):
        """Initialize W&B with clean config."""
        if not self.wandb_cfg.get('enabled', True):
            self.wandb_run = None
            return

        run_name = self.wandb_cfg.get('wandb_run_name') or f"bayesgpt-{self.model_family.name}-{int(time.time())}"
        tags = self.wandb_cfg.get('wandb_tags', []) or []

        self.wandb_run = wandb.init(
            project=self.wandb_cfg['project'],
            name=run_name,
            tags=tags + [self.model_family.name, self.model_cfg['net_cls'].__name__],
            config={
                **self.model_cfg,
                **self.training_cfg,
                **self.sampling_cfg,
                "model_family": self.model_family.name,
                "adapter": type(self.adapter).__name__,
            },
        )

        watch_log = self.wandb_cfg.get('watch_log')
        if watch_log:
            wandb.watch(self.model, log=watch_log, log_freq=self.wandb_cfg.get('watch_freq', 200))

    def train(self):
        """Main training loop."""
        for epoch_idx in range(self.training_cfg['epochs']):
            # Training phase
            epoch_loss = self._train_epoch(epoch_idx)
            self.epoch_losses.append(epoch_loss)

            # Update scheduler
            self.scheduler.step()

            # Log epoch summary
            self._log_epoch_summary(epoch_idx, epoch_loss)

        # Final save
        self._save_final_model()

    def _train_epoch(self, epoch_idx: int) -> float:
        """Execute one training epoch."""
        pbar = tqdm(
            total=self.training_cfg['steps_per_epoch'],
            desc=f"Epoch {epoch_idx + 1}/{self.training_cfg['epochs']}",
            miniters=10
        )

        epoch_losses = []
        for step in range(self.training_cfg['steps_per_epoch']):
            self.global_step += 1
            loss = self._training_step()
            epoch_losses.append(loss)

            pbar.set_postfix(loss=f"{loss:.4f}")
            pbar.update(1)

            # Periodic checkpointing
            if (self.checkpoint_cfg['save_checkpoints'] and
                    self.global_step % self.checkpoint_cfg['checkpoint_freq'] == 0):
                self._save_checkpoint(f"step_{self.global_step}")

        pbar.close()
        return sum(epoch_losses) / len(epoch_losses)

    def _training_step(self) -> float:
        """Single training step."""
        # Sample data
        sample_kwargs = {k: v for k, v in self.sampling_cfg.items() if k != 'batch_size'}
        samples = self.model_family.batch_sample(
            batch_size=self.sampling_cfg['batch_size'],
            **sample_kwargs
        )
        adapted = self.adapter.adapt(
            samples,
            intrinsic_params=self.intrinsic_params,
            device=self.model_cfg['device']
        )

        # Forward + backward
        self.optimizer.zero_grad()
        mu, log_var = self.model(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
        )
        loss = self.model.compute_loss(
            adapted["param_matrices"], mu, log_var, adapted["param_masks"]
        )

        loss.backward()
        if self.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Log
        self._log_step(loss)

        return float(loss.detach().item())

    def _log_step(self, loss: torch.Tensor):
        """Log metrics to W&B."""
        if self.wandb_run:
            wandb.log({
                "train/loss": float(loss.detach().item()),
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/step": self.global_step,
                "train/epoch": (self.global_step - 1) // self.training_cfg['steps_per_epoch'] + 1,
            }, step=self.global_step)

    def _log_epoch_summary(self, epoch_idx: int, epoch_loss: float):
        """Log epoch summary."""
        if self.wandb_run:
            wandb.log({
                "epoch/loss": epoch_loss,
                "epoch": epoch_idx + 1,
            }, step=self.global_step)

    def _save_checkpoint(self, suffix: str):
        """Save checkpoint."""
        checkpoint_path = f"{self.checkpoint_cfg['path']}.{suffix}"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': len(self.epoch_losses),
        }, checkpoint_path)
        if self.wandb_run:
            wandb.save(checkpoint_path)

    def _save_final_model(self):
        """Save final model."""
        torch.save(self.model.state_dict(), self.checkpoint_cfg['path'])
        if self.wandb_run:
            wandb.save(self.checkpoint_cfg['path'])

    def finish(self):
        """Clean up."""
        if self.wandb_run:
            wandb.finish()
