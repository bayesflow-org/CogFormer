import numpy as np
from collections.abc import Callable

from .model import Model
from .context_manager import ContextManager
from utils.simulator_utils import shifted_softplus
from adapters import Adapter


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
        max_num_regressors: int = 10,
        max_num_categories: int = 4,
        link_fun: Callable = shifted_softplus,
        context: dict[str, np.ndarray] | None = None,
        mask_randomizer_kwargs: dict | None = None,
        discrete_prob: float = 0.5,
        free_prob: float = 0.5,
        keep_intercept: bool = False,
        flatten_param_outputs: bool = True,
    ):
        # Create design config and parameter mask, either dynamically or based on user input
        if design_config is None:
            kwargs = mask_randomizer_kwargs or {}
            parameter_mask, design_config = self.context_manager.build_random_parameter_mask(
                intrinsic_params=self.intrinsic_params,
                num_regressors=num_regressors,
                max_num_regressors=max_num_regressors,
                max_num_categories=max_num_categories,
                keep_intercept=keep_intercept,
                free_prob=free_prob,
                free_intrinsics=kwargs.get("free_intrinsics"),
                fixed_intrinsics=kwargs.get("fixed_intrinsics")
            )
        else:
            parameter_mask = self.context_manager.build_parameter_mask(
                design_config=design_config,
                intrinsic_params=self.intrinsic_params,
                max_num_regressors=max_num_regressors,
                max_num_categories=max_num_categories,
                keep_intercept=keep_intercept,
            )

        # Discrete mask
        regressor_keys = [k for k in design_config.keys() if k != "1"]
        num_regressors_from_config = len(regressor_keys)


        discrete_mask = self.context_manager.build_random_discrete_mask(
            num_regressors=num_regressors_from_config, discrete_prob=discrete_prob
        )

        # Design matrix
        design_matrix = self.context_manager.build_design_matrix(
            design_config=design_config,
            num_obs=num_obs,
            context=context,
            discrete_prob=discrete_prob,
            keep_intercept=keep_intercept,
            max_num_regressors=max_num_regressors,
            max_num_categories=max_num_categories,
        )

        # Parameter matrix
        parameter_matrix = self.context_manager.sample_parameter_matrix(
            parameter_mask=parameter_mask,
            prior_fun=self.prior_fun,
            intrinsic_params=self.intrinsic_params
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

        # Regressor mask
        regressor_mask = self.context_manager.build_regressor_mask(
            num_regressors=num_regressors_from_config,
            max_num_categories=max_num_categories,
            keep_intercept=keep_intercept
        )

        # Flatten param outputs
        if flatten_param_outputs:
            parameter_mask = parameter_mask.flatten()
            parameter_matrix = parameter_matrix.flatten()

        return {
            "model_name": f"{self.name}",
            "design_config": design_config,
            "design_matrix": design_matrix,
            "param_mask": parameter_mask,
            "param_matrix": parameter_matrix,
            "sim_trials": sim_trials,
            "discrete_mask": discrete_mask,
            "regressor_mask": regressor_mask,
            "max_num_regressors": max_num_regressors,
            "keep_intercept": keep_intercept,
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
        max_num_categories: int = 4,
        discrete_prob: float = 0.5,
        keep_intercept: bool = False,
        flatten_param_outputs: bool = True,
    ) -> list[dict] | dict:
        # Sample num_obs and num_regressors per batch element if not provided
        if num_obs is None:
            num_obs_array = np.random.randint(min_num_obs, max_num_obs + 1, size=batch_size)
        else:
            num_obs_array = np.full(batch_size, num_obs)

        if num_regressors is None:
            num_regressors_array = np.random.randint(min_num_regressors, max_num_regressors + 1, size=batch_size)
        else:
            num_regressors_array = np.full(batch_size, num_regressors)

        # Initialize batch and keep track of the maximum num_obs
        list_batch = []

        for i in range(batch_size):

            sim_instance = self.sample(
                design_config=design_config,
                num_obs=num_obs_array[i],
                context=context,
                num_regressors=num_regressors_array[i],
                mask_randomizer_kwargs={} if mask_randomizer_kwargs is None else mask_randomizer_kwargs,
                discrete_prob=discrete_prob,
                keep_intercept=keep_intercept,
                max_num_regressors=max_num_regressors,
                max_num_categories=max_num_categories,
                flatten_param_outputs=flatten_param_outputs
            )

            sim_instance["num_obs"] = num_obs_array[i]
            sim_instance["num_regressors"] = num_regressors_array[i]
            sim_instance["max_num_regressors"] = max_num_regressors
            sim_instance["max_num_categories"] = max_num_categories
            list_batch.append(sim_instance)

        batch = self.collate(list_batch)
        return batch

    def collate(self, list_batch: list[dict]) -> dict[str, np.ndarray]:

        # Infer batch size
        batch_size = len(list_batch)
        num_params = len(self.intrinsic_params)
        max_num_obs = max(b["num_obs"] for b in list_batch)
        max_num_regressors = max(b["num_regressors"] for b in list_batch)
        max_num_categories = list_batch[0]["max_num_categories"]

        # Calculate max column width
        block_width = max_num_categories - 1
        max_num_cols = max_num_regressors * block_width + (1 if list_batch[0]["keep_intercept"] else 0)

        # Preallocate arrays
        design_matrices = np.zeros((batch_size, max_num_obs, max_num_cols))
        param_masks = np.zeros((batch_size, max_num_cols * num_params))
        param_matrices = np.zeros((batch_size, max_num_cols * num_params))
        regressor_masks = np.zeros((batch_size, max_num_cols))
        discrete_masks = -np.ones((batch_size, max_num_regressors))
        num_obs_array = np.zeros((batch_size, 1))
        num_regressors_array = np.zeros((batch_size, 1))
        position_encodings = np.zeros((batch_size, max_num_cols * num_params))

        # Collect lists
        model_names, design_configs = [], []

        # Initialize sim_data dict with zero arrays
        sim_keys = list_batch[0]["sim_trials"].keys()
        sim_data = {k: np.empty((batch_size, max_num_obs)) for k in sim_keys}

        # Collate batch entries
        for i, b in enumerate(list_batch):
            model_names.append(b["model_name"])
            design_configs.append(b["design_config"])

            # Pad design_matrix, param_mask, param_matrix, and regressor_mask to max_num_regressors
            num_obs = b["num_obs"]
            num_regressors = b["num_regressors"]
            num_cols = num_regressors * block_width + (1 if b["keep_intercept"] else 0)

            design_matrices[i, :num_obs, :num_cols] = b["design_matrix"]
            param_masks[i, :num_cols * num_params] = b["param_mask"]
            param_matrices[i, :num_cols * num_params] = b["param_matrix"]
            regressor_masks[i, :num_cols] = b["regressor_mask"]
            discrete_masks[i, :num_regressors] = b["discrete_mask"]
            num_obs_array[i] = b["num_obs"]
            num_regressors_array[i] = b["num_regressors"]
            position_encodings[i, :num_cols * num_params] = Adapter.encode_position(
                np.arange(num_cols * num_params), sinusoidal=True
            )

            for k in sim_keys:
                v = b["sim_trials"][k]
                sim_data[k][i, :v.shape[0]] = v

        return {
            "model_names": model_names,
            "design_configs": design_configs,
            "design_matrices": design_matrices,
            "param_masks": param_masks,
            "param_matrices": param_matrices,
            "sim_data": sim_data,
            "regressor_masks": regressor_masks,
            "discrete_masks": discrete_masks,
            "num_obs": num_obs_array,
            "num_regressors": num_regressors_array,
            "position_encodings": position_encodings,
        }
