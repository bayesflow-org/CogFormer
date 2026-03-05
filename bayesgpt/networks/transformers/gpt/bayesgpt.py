import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from ..encoder_design import Encoder
from ..decoder_fm_design import Decoder

from bayesgpt.utils.tensor_utils import broadcast_right


class BayesGPT(nn.Module):

    def __init__(
            self,
            encoder_input_dim: int = 18,
            decoder_input_dim: int = 2,
            proj_dim: int = 64,
            encoder_num_layers: int = 3,
            decoder_num_layers: int = 3,
            encoder_num_heads: int = 4,
            decoder_num_heads: int = 4,
            num_seeds: int = 10,
            seed_dim: int = 128,
            layer_norm: bool = True,
            dropout: float = 0.0,
            layer_dropout: float = 0.0,
            encoder_layer_design: str = None,
            decoder_layer_design: str = None,
            encoder_layer_kwargs: dict = None,
            decoder_layer_kwargs: dict = None,
            time_embedding_dim: int = 16,
            pos_embedding_dim: int = 16,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=encoder_input_dim,
            num_layers=encoder_num_layers,
            num_seeds=num_seeds,
            seed_dim=seed_dim,
            num_heads=encoder_num_heads,
            layer_norm=layer_norm,
            dropout=dropout,
            layer_dropout=layer_dropout,
            layer_design=encoder_layer_design,
            layer_kwargs=encoder_layer_kwargs,
        )

        self.decoder = Decoder(
            input_dim=decoder_input_dim,
            seed_dim=seed_dim,
            proj_dim=proj_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            layer_norm=layer_norm,
            dropout=dropout,
            layer_design=decoder_layer_design,
            layer_kwargs=decoder_layer_kwargs,
            time_embedding_dim=time_embedding_dim,
            pos_embedding_dim=pos_embedding_dim,
        )

    def forward(self, params, input_data, param_indices, regressor_indices, params_mask=None):

        encoder_tokens = self.encoder(input_data)

        pos_embeddings = torch.cat([param_indices, regressor_indices], dim=-1)

        pred_velocity, target_velocity = self.decoder(
            theta=params,
            query=pos_embeddings,
            key=encoder_tokens,
            query_mask=params_mask
        )

        return pred_velocity, target_velocity

    @torch.no_grad()
    def sample(self, input_data, param_indices, regressor_indices, params_mask, steps=1000, num_samples=100):

        batch_size = input_data.shape[0]
        num_tokens = param_indices.shape[1]

        encoder_tokens = self.encoder(input_data)                                    # (batch_size, num_seeds, seed_dim)
        pos_embeddings = torch.cat([param_indices, regressor_indices], dim=-1)       # (batch_size, num_tokens, 2)

        # Expand encoder outputs and position embeddings across samples
        encoder_tokens = encoder_tokens.repeat_interleave(num_samples, dim=0)        # (batch_size*num_samples, num_seeds, seed_dim)
        pos_embeddings = pos_embeddings.repeat_interleave(num_samples, dim=0)        # (batch_size*num_samples, num_tokens, 2)
        if params_mask is not None:
            params_mask_expanded = params_mask.repeat_interleave(num_samples, dim=0) # (batch_size*num_samples, num_tokens)
        else:
            params_mask_expanded = None

        # Initialize all trajectories from noise
        theta_t = torch.randn((batch_size * num_samples, num_tokens, 1), device=input_data.device)
        t = torch.zeros((batch_size * num_samples, num_tokens, 1), device=input_data.device)

        for _ in tqdm(range(steps), desc="Sampling", unit="step"):
            dt = 1 / steps
            v = self.decoder.velocity(
                theta_t=theta_t,
                query=pos_embeddings,
                key=encoder_tokens,
                t=t,
                query_mask=params_mask_expanded,
            )
            theta_t = theta_t + v * dt
            t = t + dt

        # Reshape: (batch_size*num_samples, num_tokens, 1) → (batch_size, num_samples, num_tokens, 1)
        samples = theta_t.reshape(batch_size, num_samples, num_tokens, 1).cpu().numpy()
        return samples

    def compute_loss(self, pred_velocity, target_velocity, param_masks) -> torch.Tensor:
        """"""

        sq_err = F.mse_loss(pred_velocity, target_velocity, reduction="none").squeeze(-1)
        sq_err_masked = sq_err * param_masks
        num_active_params = param_masks.sum(dim=-1)
        return torch.mean(sq_err_masked.sum(dim=-1) / num_active_params)
