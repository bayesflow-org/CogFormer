import torch
import torch.nn as nn
from networks.transformers.encoder import Encoder
from networks.transformers.decoder import Decoder


class BayesGPTv1(nn.Module):

    def __init__(
        self,
        encoder_input_dim: int = 32,
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

    def forward(self, input_data, param_indices, regressor_indices, params_mask=None):

        encoder_tokens = self.encoder(input_data)

        pos_embeddings = torch.cat([param_indices, regressor_indices], dim=-1)

        decoded_mu, decoded_logvar = self.decoder(
            query=pos_embeddings,
            key=encoder_tokens,
            query_mask=params_mask
        )

        return decoded_mu, decoded_logvar


    def compute_loss(self, true_params, mu, log_var, param_masks) -> torch.Tensor:
        """
        Heteroskedastic Gaussian NLL with masking.
        """
        mu = mu.squeeze(dim=-1)
        log_var = log_var.squeeze(dim=-1)

        inv_var = torch.exp(-log_var)
        nll = 0.5 * ((true_params - mu) ** 2 * inv_var + log_var)

        nll_masked = nll * param_masks
        num_active_params = param_masks.sum(dim=-1)
        return torch.mean(nll_masked.sum(dim=-1) / num_active_params)
