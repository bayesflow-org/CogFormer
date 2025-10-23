import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
np.set_printoptions(suppress=True)

import wandb
from tqdm.auto import tqdm

from simulators import NestedModelFamily
from simulators.benchmarks import DDM
from adapters import Adapter
from networks.transformers.gpt import BayesGPTv1


class BayesGPTTrainer:
    pass

if __name__ == "__main__":
    pass
