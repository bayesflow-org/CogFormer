import torch
import torch.nn as nn

from networks.transformers.encoder import Encoder
from bayesgpt.networks.transformers.decoder_vi import Decoder


class BayesGPTv0(nn.Module):

    def __init__(
        self,
        encoder_input_dim: int,
        proj_dim: int = 64,
        encoder_num_layers: int = 3,
        decoder_num_layers: int = 3,
        encoder_num_heads: int = 4,
        decoder_num_heads: int = 4,
        seed_dim: int = 128,
        ln: bool = True,
        dropout: float = 0.0,
        layer_dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=encoder_input_dim,
            num_layers=encoder_num_layers,
            seed_dim=seed_dim,
            num_heads=encoder_num_heads,
            ln=ln,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )

        self.decoder = Decoder(
            input_dim=seed_dim+2,
            proj_dim=proj_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            ln=ln,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )

    def forward(self, input_data, param_indices, regressor_indices, attn_mask=None, key_padding_mask=None):

        data_rep = self.encoder(input_data)
        data_rep = data_rep.repeat(1, regressor_indices.size(1), 1)

        data_rep_pos = torch.cat([data_rep, param_indices, regressor_indices], dim=-1)

        decoded_mu, decoded_logvar = self.decoder(data_rep_pos, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return decoded_mu, decoded_logvar