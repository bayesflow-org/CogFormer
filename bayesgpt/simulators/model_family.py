import numpy as np
from collections.abc import Callable

from .model import Model
from .context_manager import ContextManager
from utils.simulator_utils import shifted_softplus


class NestedModelFamily:
    def __init__(
        self,
        name: str,
        model: Model,
        context_manager: ContextManager,
        prior_fun: dict[str, Callable],
        intrinsic_params: list[str],
    ):
        self.name = name
        self.model = model
        self.context_manager = context_manager
        self.prior_fun = prior_fun
        self.intrinsic_params = intrinsic_params

    @property
    def parameter_names(self) -> list[str]:
        return self.context_manager.parameter_names

    def sample(
        self,
        design_config: dict[str, list[str]],
        *,
        num_obs: int = 10,
        num_regressors: int = 0,
        link_fun: Callable = shifted_softplus,
        context: dict[str, np.ndarray] | None = None,
        mask_randomizer_kwargs: dict | None = None,
    ):

        # Create design config and parameter mask, either dynamically or based on user input
        if design_config is None:

            kwargs = mask_randomizer_kwargs or {}
            parameter_mask = self.context_manager.build_random_parameter_mask(
                intrinsic_params=self.intrinsic_params,
                num_regressors=num_regressors,
                **kwargs,
            )
            design_config = self.context_manager.mask_to_design_config(parameter_mask, self.intrinsic_params)

        else:
            parameter_mask = self.context_manager.build_parameter_mask(
                design_config=design_config,
                intrinsic_params=self.intrinsic_params,
            )

        # Design matrix
        design_matrix = self.context_manager.build_design_matrix(
            design_config=design_config, num_obs=num_obs, context=context
        )

        # Parameter matrix
        parameter_matrix = self.context_manager.sample_parameter_matrix(
            parameter_mask=parameter_mask, priors=self.prior_fun, intrinsic_params=self.intrinsic_params
        )

        # Compose per-trial intrinsic values
        intrinsic_values_matrix = link_fun(design_matrix @ parameter_matrix)

        # Package for model
        params = {
            name: intrinsic_values_matrix[:, j].astype(np.float32, copy=False)
            for j, name in enumerate(self.intrinsic_params)
        }

        sim_trials = self.model.simulate(params, context=None)

        return {
            "model_name": f"{self.name}",
            "design_config": design_config,
            "design_matrix": design_matrix,
            "parameter_mask": parameter_mask,
            "parameter_matrix": parameter_matrix,
            "sim_trials": sim_trials,
        }

    def batch_sample(
            self,
            *,
            batch_size: int,
            num_obs: int | None = None,
            design_config: dict[str, list[str]] | None = None,
            num_regressors: int | None = None,
            mask_randomizer_kwargs: dict | None = None,
            context: dict[str, np.ndarray] | None = None,
            min_num_obs: int = 10,
            max_num_obs: int = 600,
            min_num_regressors: int = 0,
            max_num_regressors: int = 10,
    ):
        num_obs = num_obs or np.random.randint(min_num_obs, max_num_obs + 1)
        num_regressors = num_regressors or np.random.randint(min_num_regressors, max_num_regressors + 1)

        batch = []
        for i in range(batch_size):

            sim_instance = self.sample(
                design_config=design_config,
                num_obs=num_obs,
                context=context,
                num_regressors=num_regressors,
                mask_randomizer_kwargs={} if mask_randomizer_kwargs is None else mask_randomizer_kwargs
            )

            sim_instance["batch_id"] = i
            sim_instance["num_obs"] = num_obs
            sim_instance["num_factors"] = num_regressors
            batch.append(sim_instance)

        return batch
