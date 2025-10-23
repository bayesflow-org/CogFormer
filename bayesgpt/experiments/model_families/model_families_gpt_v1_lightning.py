import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import wandb
import time
from typing import Dict, Any, Optional, Type
import lightning as L
from lightning import LightningModule

from bayesgpt.networks.transformers.encoder import Encoder
from bayesgpt.networks.transformers.decoder import Decoder


class BayesGPTv1Lightning(LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.save_hyperparameters()
