import torch
import numpy as np


class TorchAdapter:

    @staticmethod
    def concatenate(x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x)
