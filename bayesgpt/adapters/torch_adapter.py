import torch
import numpy as np


class TorchAdapter:

    @staticmethod
    def to_torch_tensor(x: np.ndarray, copy: bool = False) -> torch.Tensor:
        return torch.tensor(x) if copy else torch.from_numpy(x)

    @staticmethod
    def concatenate(x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x)