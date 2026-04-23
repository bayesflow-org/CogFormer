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

from cogformer.simulators import NestedModelFamily, ModelClass
from cogformer.simulators.benchmarks import DDM, RDM, CDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from cogformer.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
from cogformer.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from cogformer.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.utils.plot_utils import cogformer_fm_colors


# Per-model metadata used in val_step (fixed design config + display info)
MODEL_CONFIGS = {
    "DDM": {
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "val_design_config": {
            "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "u_1": ["v", "a", "z", "tau", "s_v"],
            "u_2": ["v", "a", "z", "tau"],
            "u_1:u_2": ["v", "a", "z"],
        },
    },
    "RDM": {
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "val_design_config": {
            "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_diff", "a", "tau", "s_v"],
            "u_2": ["v", "v_diff", "a", "tau"],
            "u_1:u_2": ["v", "v_diff", "a"],
        },
    },
    "CDM": {
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "val_design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v"],
            "u_2": ["v", "v_theta", "a", "tau"],
            "u_1:u_2": ["v", "v_theta", "a"],
        },
    },
}


class CogFormerTrainer:
    def __init__(self, cf, model_class, adapter, use_wandb=False):
        self.cf = cf
        self.model_class = model_class
        self.adapter = adapter
        self.use_wandb = use_wandb

    def train(self, train_config, val_config, checkpoint_path="cogformer_model_class_fm.pt", fig_path="fig"):
        global_step = 0
        optimizer = AdamW(self.cf.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])

        for epoch in range(train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {epoch + 1}/{train_config['epochs']}",
                miniters=100,
            )
            for _ in range(train_config["steps_per_epoch"]):
                loss, current_lr = self.train_step(
                    config=train_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                if self.use_wandb:
                    wandb.log(
                        {"train/loss": loss, "opt/lr": current_lr, "epoch": epoch + 1},
                        step=global_step,
                    )
                global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            if (epoch + 1) % 1000 == 0:
                self.val_step(val_config, global_step, fig_path)

            scheduler.step()
            pbar.close()

        checkpoint_dir = Path("./cogformer/experiments/checkpoints/fm/model_class/")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cf.state_dict(), checkpoint_dir / checkpoint_path)
        logging.info(f"[saved] {checkpoint_dir / checkpoint_path}")

    def train_step(self, config, optimizer, scheduler):
        train_samples = self.model_class.batch_sample(
            **config["model_family_config"],
            **config["train_sample_config"],
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
        )

        adapted = self.adapter.adapt(
            train_samples,
            intrinsic_params=[],
            num_params=train_samples["max_num_params"],
        )

        optimizer.zero_grad()

        pred_velocity, target_velocity = self.cf(
            adapted["param_matrices"][..., None],
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            model_ids=adapted["model_ids"],
        )

        L = self.cf.compute_loss(pred_velocity, target_velocity, adapted["param_masks"])
        L.backward()

        if config["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.cf.parameters(), config["gradient_clip_norm"])

        optimizer.step()
        loss = L.detach().item()
        current_lr = scheduler.get_last_lr()[0]
        return loss, current_lr

    @torch.no_grad()
    def val_step(self, config, global_step, fig_path):
        self.cf.eval()

        model_family_config = config["model_family_config"]
        max_num_params = self.model_class.max_num_params
        colors = cogformer_fm_colors()

        figures_dir = Path("./cogformer/experiments/figures/fm/model_class/recovery")
        figures_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_cfg in MODEL_CONFIGS.items():
            mf = self.model_class.model_families[model_name]
            model_id = self.model_class.model_registry[model_name]
            link_fun = self.model_class.link_funs[model_name]

            intrinsic_params = model_cfg["intrinsic_params"]
            variable_names = model_cfg["variable_names"]
            design_config = model_cfg["val_design_config"]

            test_samples = mf.batch_sample(
                design_config=design_config,
                batch_size=config["batch_size"],
                num_obs=config["num_obs"],
                max_num_regressors=model_family_config["max_num_regressors"],
                max_num_categories=model_family_config["max_num_categories"],
                keep_intercept=model_family_config["keep_intercept"],
                flatten_param_outputs=True,
                link_fun=link_fun,
            )
            test_samples["model_ids"] = np.full(config["batch_size"], model_id, dtype=np.int64)

            # Lift local param positions to global space before adapting
            test_samples["param_matrices"], test_samples["param_masks"] = \
                self.model_class.lift_to_global_space(
                    model_name,
                    test_samples["param_matrices"],
                    test_samples["param_masks"],
                )

            adapted = self.adapter.adapt(
                test_samples,
                intrinsic_params=[],
                num_params=max_num_params,
            )

            pred_set = self.cf.sample(
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                steps=config["fm_sample_steps"],
                num_samples=config["fm_num_samples"],
                model_ids=adapted["model_ids"],
            )

            batch_size = config["batch_size"]
            global_indices = self.model_class.local_to_global[model_name]

            true_set = adapted["param_matrices"].detach().cpu().numpy()
            n_rows = true_set.shape[1] // max_num_params
            true_set = true_set.reshape(batch_size, n_rows, max_num_params)[:, :, global_indices]
            pred_set = pred_set.reshape(batch_size, config["fm_num_samples"], n_rows, max_num_params)[:, :, :, global_indices]

            params_mask = adapted["param_masks"].detach().cpu().numpy()
            params_mask = params_mask.reshape(batch_size, n_rows, max_num_params)[0][:, global_indices]

            recovery_fig = adaptive_recovery(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=model_family_config["max_num_categories"],
                parameter_mask=params_mask,
                variable_names=variable_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )

            fig_name = f"{model_name.lower()}_{fig_path}_recovery.pdf"
            recovery_fig.savefig(figures_dir / fig_name, bbox_inches="tight")
            logging.info(f"[saved] {figures_dir / fig_name}")

            if self.use_wandb:
                wandb.log({f"val/{model_name}/recovery": wandb.Image(recovery_fig)}, step=global_step)

            plt.close(recovery_fig)

        self.cf.train()

    @staticmethod
    def finish():
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true")

    # Architecture
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=128)
    parser.add_argument("--model_embed_dim", type=int, default=8)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)

    # Training schedule
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--val_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)

    # Flow matching
    parser.add_argument("--fm_sample_steps", type=int, default=200)
    parser.add_argument("--fm_num_samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_num_regressors = 2
    max_num_categories = 2
    keep_intercept = True
    num_models = 3

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "add_interaction": True,
    }

    train_sample_config = {
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "fixed_config": False,
    }

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
        "num_obs": args.val_num_obs,
        "model_family_config": model_family_config,
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
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
        "num_models": num_models,
        "model_embed_dim": args.model_embed_dim,
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
    }

    logging.info(
        f"Training ModelClass CF: {train_config['epochs']} epochs, "
        f"{train_config['steps_per_epoch']} steps/epoch, "
        f"batch_size={train_config['batch_size']}, device={device}"
    )

    # Build per-model families with their own mask randomizer kwargs
    model_families = {
        "DDM": NestedModelFamily(
            model=DDM(), name="DDM", prior_fun=ddm_priors(),
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "a", "z", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
            },
        ),
        "RDM": NestedModelFamily(
            model=RDM(), name="RDM", prior_fun=rdm_priors(),
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_diff", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
            },
        ),
        "CDM": NestedModelFamily(
            model=CDM(), name="CDM", prior_fun=cdm_priors(),
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_theta", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
            },
        ),
    }

    link_funs = {
        "DDM": ddm_link_fun(),
        "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(),
    }

    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()

    if args.use_wandb:
        wandb.init(
            project="cogformer-model-class",
            tags=["CogFormer", "ModelClass", "DDM", "RDM", "CDM"],
            config={**train_config, "cogformer": cogformer_config},
        )

    checkpoint_path = (
        f"cogformer_model_class_fm"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )

    fig_path = (
        f"model_class_fm"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
    )

    trainer = CogFormerTrainer(
        cf=cogformer,
        model_class=model_class,
        adapter=adapter,
        use_wandb=args.use_wandb,
    )

    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path,
        fig_path=fig_path,
    )

    if args.use_wandb:
        trainer.finish()
