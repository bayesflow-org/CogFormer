import numpy as np
from collections.abc import Callable

from .model_family import NestedModelFamily


class ModelClass:
    """
    A collection of NestedModelFamily instances representing multiple cognitive
    model families. During simulation, a model is sampled per batch element,
    enabling a single CF to perform inference across all families and design
    configurations.

    Parameters
    ----------
    model_families : dict[str, NestedModelFamily]
        Mapping from model name to its NestedModelFamily instance.
    link_funs : dict[str, dict | Callable]
        Per-model link functions, keyed by model name.
    weights : list[float] | None
        Sampling weights over model families. Uniform if None.
    """

    def __init__(
        self,
        model_families: dict[str, NestedModelFamily],
        link_funs: dict[str, dict | Callable],
        weights: list[float] | None = None,
    ):
        if set(model_families.keys()) != set(link_funs.keys()):
            raise ValueError("model_families and link_funs must have the same keys.")

        self.model_families = model_families
        self.link_funs = link_funs
        self.model_names = list(model_families.keys())
        self.model_registry = {name: i for i, name in enumerate(self.model_names)}
        self.num_models = len(self.model_names)

        # Build global parameter space as ordered union across all model families
        seen_order: dict[str, int] = {}
        for mf in model_families.values():
            for p in mf.intrinsic_params:
                if p not in seen_order:
                    seen_order[p] = len(seen_order)
        self.all_params: list[str] = list(seen_order.keys())
        self.max_num_params: int = len(self.all_params)

        # Per-model local→global index mappings and binary model masks
        self.local_to_global: dict[str, list[int]] = {}
        self.model_masks: dict[str, np.ndarray] = {}
        for name, mf in model_families.items():
            g_indices = [self.all_params.index(p) for p in mf.intrinsic_params]
            self.local_to_global[name] = g_indices
            mask = np.zeros(self.max_num_params, dtype=np.float32)
            mask[g_indices] = 1.0
            self.model_masks[name] = mask

        if weights is not None:
            weights = np.array(weights, dtype=np.float64)
            weights /= weights.sum()
        self.weights = weights

    def batch_sample(
        self,
        batch_size: int,
        num_obs: int | None = None,
        design_config: dict[str, list[str]] | None = None,
        num_regressors: int | None = None,
        min_num_obs: int = 10,
        max_num_obs: int = 600,
        min_num_regressors: int = 0,
        max_num_regressors: int = 3,
        max_num_categories: int = 3,
        discrete_prob: float = 0.5,
        keep_intercept: bool = True,
        remove_intercept_on_collate: bool = False,
        mask_randomizer_kwargs: dict | None = None,
        flatten_param_outputs: bool = True,
        fixed_config: bool = False,
        add_interaction: bool = False,
    ) -> dict:
        if num_obs is None:
            num_obs = np.random.randint(min_num_obs, max_num_obs + 1)

        if num_regressors is None:
            num_regressors_array = np.random.randint(
                min_num_regressors, max_num_regressors + 1, size=batch_size
            )
        else:
            num_regressors_array = np.full(batch_size, num_regressors)

        sampled_names = np.random.choice(
            self.model_names, size=batch_size, p=self.weights
        )

        list_batch = []
        for i, model_name in enumerate(sampled_names):
            mf = self.model_families[model_name]

            ctx_i = None
            if hasattr(mf.model, "build_default_context"):
                ctx_i = mf.model.build_default_context(num_obs=num_obs)

            sim_instance = mf.sample(
                design_config=design_config,
                num_obs=num_obs,
                num_regressors=int(num_regressors_array[i]),
                context=ctx_i,
                mask_randomizer_kwargs=mask_randomizer_kwargs,
                max_num_categories=max_num_categories,
                discrete_prob=discrete_prob,
                keep_intercept=keep_intercept,
                flatten_param_outputs=False,  # collate handles the final form
                fixed_config=fixed_config,
                add_interaction=add_interaction,
                link_fun=self.link_funs[model_name],
            )

            sim_instance["model_id"] = self.model_registry[model_name]
            sim_instance["num_obs"] = num_obs
            sim_instance["num_regressors"] = int(num_regressors_array[i])
            sim_instance["max_num_regressors"] = max_num_regressors
            sim_instance["max_num_categories"] = max_num_categories
            list_batch.append(sim_instance)

        return self.collate(
            list_batch,
            flatten_param_outputs=flatten_param_outputs,
            remove_intercept=remove_intercept_on_collate,
        )

    def collate(
        self,
        list_batch: list[dict],
        flatten_param_outputs: bool = True,
        remove_intercept: bool = False,
    ) -> dict:
        batch_size = len(list_batch)
        max_num_obs = max(b["num_obs"] for b in list_batch)
        max_num_regressors = list_batch[0]["max_num_regressors"]
        max_num_categories = list_batch[0]["max_num_categories"]

        block_width = max_num_categories - 1
        max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
        max_num_cols = max_total_regressors * block_width + (
            1 if list_batch[0]["keep_intercept"] else 0
        )
        max_num_params = self.max_num_params

        if flatten_param_outputs:
            param_outputs_shape = (batch_size, max_num_cols * max_num_params)
        else:
            param_outputs_shape = (batch_size, max_num_cols, max_num_params)

        design_matrices = np.zeros((batch_size, max_num_obs, max_num_cols))
        param_masks = np.zeros(param_outputs_shape)
        param_matrices = np.zeros(param_outputs_shape)
        regressor_masks = np.zeros((batch_size, max_num_cols))
        discrete_masks = -np.ones((batch_size, max_total_regressors))
        num_obs_array = np.zeros((batch_size, 1))
        num_regressors_array = np.zeros((batch_size, 1))
        model_ids = np.zeros(batch_size, dtype=np.int64)

        model_names_out, design_configs = [], []

        # Union of sim_data keys across all models in this batch
        all_sim_keys = set().union(*(b["sim_trials"].keys() for b in list_batch))
        sim_data = {k: np.zeros((batch_size, max_num_obs, 1)) for k in all_sim_keys}

        for i, b in enumerate(list_batch):
            model_name = b["model_name"]

            model_names_out.append(model_name)
            design_configs.append(b["design_config"])
            model_ids[i] = b["model_id"]

            num_obs = b["num_obs"]
            num_cols = b["design_matrix"].shape[1]

            design_matrices[i, :num_obs, :num_cols] = b["design_matrix"]

            # param_mask and param_matrix are (num_cols, num_params) — place at global positions
            global_indices = self.local_to_global[model_name]
            if flatten_param_outputs:
                padded_mask = np.zeros((max_num_cols, max_num_params))
                padded_matrix = np.zeros((max_num_cols, max_num_params))
                for local_idx, global_idx in enumerate(global_indices):
                    padded_mask[:num_cols, global_idx] = b["param_mask"][:, local_idx]
                    padded_matrix[:num_cols, global_idx] = b["param_matrix"][:, local_idx]
                param_masks[i] = padded_mask.flatten()
                param_matrices[i] = padded_matrix.flatten()
            else:
                for local_idx, global_idx in enumerate(global_indices):
                    param_masks[i, :num_cols, global_idx] = b["param_mask"][:, local_idx]
                    param_matrices[i, :num_cols, global_idx] = b["param_matrix"][:, local_idx]

            regressor_masks[i, :num_cols] = b["regressor_mask"]
            discrete_masks[i, : b["discrete_mask"].shape[0]] = b["discrete_mask"]
            num_obs_array[i] = num_obs
            num_regressors_array[i] = b["num_regressors"]

            for k, v in b["sim_trials"].items():
                sim_data[k][i, : v.shape[0]] = v

        if remove_intercept:
            design_matrices = design_matrices[:, :, 1:]

        return {
            "model_names": model_names_out,
            "model_ids": model_ids,
            "design_configs": design_configs,
            "design_matrices": design_matrices,
            "param_masks": param_masks,
            "param_matrices": param_matrices,
            "sim_data": sim_data,
            "regressor_masks": regressor_masks,
            "discrete_masks": discrete_masks,
            "num_obs": num_obs_array,
            "num_regressors": num_regressors_array,
            "max_num_regressors": max_num_regressors,
            "max_num_categories": max_num_categories,
            "max_num_params": self.max_num_params,
            "num_models": self.num_models,
        }

    def lift_to_global_space(
        self,
        model_name: str,
        param_matrices: np.ndarray,
        param_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reindex flattened param_matrices/param_masks from model-local to global param positions.

        Parameters
        ----------
        model_name : str
        param_matrices : ndarray, shape (batch_size, num_cols * n_local)
        param_masks    : ndarray, shape (batch_size, num_cols * n_local)

        Returns
        -------
        (param_matrices_global, param_masks_global), both shape (batch_size, num_cols * max_num_params)
        """
        n_local = len(self.model_families[model_name].intrinsic_params)
        n_global = self.max_num_params
        batch_size = param_matrices.shape[0]
        n_rows = param_matrices.shape[1] // n_local
        global_indices = self.local_to_global[model_name]

        pm_local = param_matrices.reshape(batch_size, n_rows, n_local)
        pmask_local = param_masks.reshape(batch_size, n_rows, n_local)

        pm_global = np.zeros((batch_size, n_rows, n_global), dtype=param_matrices.dtype)
        pmask_global = np.zeros((batch_size, n_rows, n_global), dtype=param_masks.dtype)

        for local_idx, global_idx in enumerate(global_indices):
            pm_global[:, :, global_idx] = pm_local[:, :, local_idx]
            pmask_global[:, :, global_idx] = pmask_local[:, :, local_idx]

        return (
            pm_global.reshape(batch_size, n_rows * n_global),
            pmask_global.reshape(batch_size, n_rows * n_global),
        )
