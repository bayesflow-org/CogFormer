import numpy as np
from simulators import NestedModelFamily


class ModelClass:

    def __init__(self, model_families: list[NestedModelFamily]):
        super().__init__()
        self.model_families = model_families

    def sample(self):
        raise NotImplementedError

    def batch_sample(self):
        raise NotImplementedError
