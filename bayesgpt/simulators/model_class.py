import numpy as np
from collections.abc import Callable
from simulators import NestedModelFamily, ContextManager, Model


class ModelClass:

    def __init__(self, models: list[Model]):
        super().__init__()
        self.models = models

    def sample(
        self,
        model_families: list[NestedModelFamily],
        priors: dict[str, dict[str, float | Callable]],
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:

        samples = {}
        model_id = np.random.choice(len(self.models))
        context_manager = ContextManager()
        # model_family = NestedModelFamily(
        #     context_manager=context_manager,
        #     model=self.models[model_id],
        #     priors=priors,
        #     intrinsic_params=
        # )

        return samples
