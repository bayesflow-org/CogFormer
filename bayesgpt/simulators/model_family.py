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
        design_config: dict[str, list[str]] = None,
        num_obs: int = 10,
        num_regressors: int = 0,
        link_fun: Callable = shifted_softplus,
        context: dict[str, np.ndarray] | None = None,
        mask_randomizer_kwargs: dict | None = None,
        discrete_mask: np.ndarray | None = None,
        discrete_prob: float = 0.5,
        keep_intercept: bool = False,
    ):
        # Create design config and parameter mask, either dynamically or based on user input
        if design_config is None:
            kwargs = mask_randomizer_kwargs or {}
            parameter_mask = self.context_manager.build_random_parameter_mask(
                intrinsic_params=self.intrinsic_params,
                num_regressors=num_regressors,
                keep_intercept=keep_intercept,
                **kwargs,
            )
            design_config = self.context_manager.mask_to_design_config(
                parameter_mask=parameter_mask,
                intrinsic_params=self.intrinsic_params,
                keep_intercept=keep_intercept,
            )

        else:
            parameter_mask = self.context_manager.build_parameter_mask(
                design_config=design_config,
                intrinsic_params=self.intrinsic_params,
            )

        # Discrete mask
        regressor_keys = [k for k in design_config.keys() if k != 1]
        num_regressors_from_config = len(regressor_keys)

        if discrete_mask is None:
            discrete_mask = self.context_manager.build_random_discrete_mask(
                num_regressors=num_regressors_from_config, discrete_prob=discrete_prob
            )
            #
            # No discrete mask, sample internally
            # Flip a coin with p = 0.5, if 0
            # sample a continuous regressor
            # if 1, sample k categories k ~ U(2, 4), then create a vector of
            # pvals = [1/k, 1/k,...], pass to np.random.multinomial(n=1, pvals=pvals, size=num_obs)
            # dummy encode the one-hot-encoded outputs
            # voila

        # Design matrix
        design_matrix = self.context_manager.build_design_matrix(
            design_config=design_config,
            num_obs=num_obs,
            context=context,
            discrete_mask=discrete_mask,
            discrete_prob=discrete_prob,
            keep_intercept=keep_intercept
        )

        # Parameter matrix
        parameter_matrix = self.context_manager.sample_parameter_matrix(
            parameter_mask=parameter_mask, prior_fun=self.prior_fun, intrinsic_params=self.intrinsic_params
        )

        # Compose per-trial intrinsic values
        regressed_parameters = link_fun(design_matrix @ parameter_matrix)

        # Package for model
        params = {
            name: regressed_parameters[:, j].astype(np.float32, copy=False)
            for j, name in enumerate(self.intrinsic_params)
        }

        #explicate what prepare params does / think about whether this is not more suitable
        # as a task for the context manager? O_o
        params = self.model.prepare_params(params=params, num_obs=num_obs, context=context)
        sim_trials = self.model.simulate(params, context=context)

        return {
            "model_name": f"{self.name}",
            "design_config": design_config,
            "design_matrix": design_matrix,
            "param_mask": parameter_mask,
            "param_matrix": parameter_matrix,
            "sim_trials": sim_trials,
            "discrete_mask": discrete_mask
        }

    def batch_sample(
            self,
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
            discrete_prob: float = 0.5,
            keep_intercept: bool = False,
    ):
        num_obs = num_obs or np.random.randint(min_num_obs, max_num_obs + 1)
        num_regressors = num_regressors or np.random.randint(min_num_regressors, max_num_regressors + 1)

        # Initialize batch and keep track of the maximum num_obs
        list_batch = []
        list_num_obs = np.zeros(batch_size)
        list_num_regressors = np.zeros(batch_size)

        for i in range(batch_size):
            list_num_obs[i] = num_obs
            list_num_regressors[i] = num_regressors

            sim_instance = self.sample(
                design_config=design_config,
                num_obs=num_obs,
                context=context,
                num_regressors=num_regressors,
                mask_randomizer_kwargs={} if mask_randomizer_kwargs is None else mask_randomizer_kwargs,
                discrete_mask=None,
                discrete_prob=discrete_prob,
                keep_intercept=keep_intercept
            )

            sim_instance["num_obs"] = num_obs
            sim_instance["num_regressors"] = num_regressors
            list_batch.append(sim_instance)

        # batch = self.collate(list_batch)
        batch = list_batch
        return batch

    def collate(self, list_batch: list[dict]) -> dict[str, np.ndarray]:
        # Infer batch size
        batch_size = len(list_batch)
        num_params = len(self.intrinsic_params)
        num_obs = list_batch[0]["design_matrix"].shape[0]

        # Use max num_regressors across batch, plus 1 for intercept if present
        num_regressors = max(
            list_batch[i]['design_matrix'].shape[1] - (1 if "1" in list_batch[i]['design_config'] else 0)
            for i in range(batch_size)
        )
        print(type(num_regressors))
        # Assume keep_intercept is consistent across batch; use first instance to check
        keep_intercept = "1" in list_batch[0]['design_config']
        num_columns = num_regressors + (1 if keep_intercept else 0)

        # Preallocate arrays
        design_matrices = np.empty((batch_size, num_obs, num_columns))
        param_mask = np.empty((batch_size, num_columns, num_params))
        param_matrices = np.empty((batch_size, num_columns, num_params))
        discrete_masks = np.empty((batch_size, num_regressors))
        num_obs_array = np.empty(batch_size)
        num_regressors_array = np.empty(batch_size)

        # Collect lists
        model_names, design_configs = [], []

        # Initialize sim_data dict with zero arrays
        sim_keys = list_batch[0]["sim_trials"].keys()
        sim_data = {k: np.empty((batch_size, num_obs)) for k in sim_keys}

        # Collate batch entries
        for i, batch in enumerate(list_batch):
            model_names.append(batch["model_name"])
            design_configs.append(batch["design_config"])
            num_obs_array[i] = batch["num_obs"]
            num_regressors_array[i] = batch["num_regressors"]
            design_matrices[i] = batch["design_matrix"]
            param_mask[i] = batch["param_mask"]
            param_matrices[i] = batch["param_matrix"]
            discrete_masks[i] = batch["discrete_mask"]

            for k, v in batch["sim_trials"].items():
                sim_data[k][i] = v

        return {
            "model_names": model_names,
            "design_configs": design_configs,
            "design_matrices": design_matrices,
            "param_mask": param_mask,
            "param_matrices": param_matrices,
            "sim_data": sim_data,
            "discrete_masks": discrete_masks,
            "num_obs": num_obs_array,
            "num_regressors": num_regressors_array,
        }
