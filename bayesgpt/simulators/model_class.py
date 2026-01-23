import numpy as np

from .model_family import NestedModelFamily


class ModelClass:

    def __init__(self, model_families: list[NestedModelFamily]):
        super().__init__()
        self.model_families = model_families

    def sample(
        self,
        model_families: list[NestedModelFamily],
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:

        samples = {}
        for model_family in model_families:
            # Call batch_sample for each model family
            batch = model_family.batch_sample(
                batch_size=batch_size
            )
            # Store results keyed by model family name
            samples[model_family.name] = batch

        return samples
