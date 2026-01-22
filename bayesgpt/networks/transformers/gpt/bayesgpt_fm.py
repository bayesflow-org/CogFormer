import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from ..encoder import Encoder
from ..decoder_fm import Decoder

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
        )

        self.decoder = Decoder(
            input_dim=decoder_input_dim,
            seed_dim=seed_dim,
            proj_dim=proj_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            layer_norm=layer_norm,
            dropout=dropout,
            layer_dropout=layer_dropout,
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

        theta_t = torch.randn((num_samples, *param_indices.shape), device=input_data.device)

        samples = torch.zeros_like(theta_t, device=theta_t.device)

        encoder_tokens = self.encoder(input_data)
        pos_embeddings = torch.cat([param_indices, regressor_indices], dim=-1)

        for sample_id in tqdm(range(num_samples), desc="Sampling", unit="sample"):
            
            t = torch.zeros_like(theta_t[sample_id], device=theta_t.device)

            for _ in range(steps):
                dt = 1 / steps

                v = self.decoder.velocity(
                    theta_t=theta_t[sample_id], 
                    query=pos_embeddings, 
                    key=encoder_tokens, 
                    t=t,
                    query_mask=params_mask
                )

                theta_t[sample_id] = theta_t[sample_id] + v * dt
                t = t + dt

            samples[sample_id] = theta_t[sample_id]

        samples = torch.swapaxes(samples, 0, 1).cpu().numpy()
        return samples

    def compute_loss(self, pred_velocity, target_velocity, param_masks) -> torch.Tensor:
        """"""
    
        sq_err = F.mse_loss(pred_velocity, target_velocity, reduction="none").squeeze(-1)
        sq_err_masked = sq_err * param_masks
        num_active_params = param_masks.sum(dim=-1)
        return torch.mean(sq_err_masked.sum(dim=-1) / num_active_params)
