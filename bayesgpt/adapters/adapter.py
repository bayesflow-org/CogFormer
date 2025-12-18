import torch
import numpy as np
from collections.abc import Mapping, Sequence

class Adapter:


    @staticmethod
    def build_parameter_indices(
        intrinsic_params: list[str],
        num_regressors: int,
        num_categories: int,
    ) -> np.ndarray:
        # Get number of intrinsic params
        num_intrinsic_params = len(intrinsic_params)
        num_tiles = num_regressors * (num_categories - 1) + 1
        indices = np.tile(np.linspace(0.0, 1.0, num_intrinsic_params), num_tiles)[..., None]
        return indices

    @staticmethod
    def build_regressor_indices(
        intrinsic_params: list[str],
        num_regressors: int,
        num_categories: int,
    ):
        num_intrinsic_params = len(intrinsic_params)
        num_indices = num_regressors * (num_categories - 1) + 1
        indices = np.repeat(np.linspace(0.0, 1.0, num_indices), num_intrinsic_params)[..., None]
        return indices

    @staticmethod
    def build_token_embeddings(
        regressor_indices: np.ndarray | torch.Tensor,
        parameter_indices: np.ndarray | torch.Tensor,
        additional_tokens: np.ndarray
    ):
        regressor_indices = np.atleast_2d(regressor_indices)
        parameter_indices = np.atleast_2d(parameter_indices)
        additional_tokens = np.atleast_2d(additional_tokens)

        # Check if dimensions of indices align
        num_tokens = regressor_indices.shape[0]
        assert (
            parameter_indices.shape[0] == num_tokens and
            additional_tokens.shape[0] == num_tokens
        ), "All inputs must have the same number of tokens."

        embeddings = np.concatenate(
            (regressor_indices, parameter_indices, additional_tokens),
            axis=-1
        )

        return embeddings


    @staticmethod
    def to_torch_tensor(x: np.ndarray, copy: bool = False) -> torch.Tensor:
        return torch.tensor(x) if copy else torch.from_numpy(x)

    @staticmethod
    def to_device(batch, device):
        """Recursively move tensors in mappings/sequences to the given device."""
        if isinstance(batch, Mapping):
            return {k: Adapter.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            return [Adapter.to_device(v, device) for v in batch]
        elif hasattr(batch, "to"):
            return batch.to(device)
        else:
            return batch

    @staticmethod
    def adapt(
        samples: dict,
        intrinsic_params: list[str],
        device: str | torch.device = torch.device("cuda"),
        additional_tokens: np.ndarray | None = None,
    ) -> dict:
        # Fetch inputs from samples and adapt
        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]

        batch_size, max_num_obs, max_num_cols = design_matrices.shape

        input_data = Adapter.stack([design_matrices, rts, choices], axis=-1)
        input_data = Adapter.to_torch_tensor(input_data).to(torch.float32)

        # Build indices
        parameter_indices = [Adapter.build_parameter_indices(
            intrinsic_params,
            num_regressors=samples["max_num_regressors"],
            num_categories=samples["max_num_categories"],
        ) for _ in range(batch_size)]
        parameter_indices = Adapter.to_torch_tensor(np.array(parameter_indices)).to(torch.float32)

        regressor_indices = [Adapter.build_regressor_indices(
            intrinsic_params,
            num_regressors=samples["max_num_regressors"],
            num_categories=samples["max_num_categories"],
        ) for _ in range(batch_size)]
        regressor_indices = Adapter.to_torch_tensor(np.array(regressor_indices)).to(torch.float32)

        # Build tokens
        num_tokens = parameter_indices[0].shape[0]
        if additional_tokens is None:
            additional_tokens = np.zeros((batch_size, num_tokens, 1))
        else:
            additional_tokens = np.asarray(additional_tokens)
            if additional_tokens.ndim == 2:
                additional_tokens = additional_tokens[..., None]

        token_embeddings_list = []
        for b in range(batch_size):
            embeddings = Adapter.build_token_embeddings(
                regressor_indices=regressor_indices[b],
                parameter_indices=parameter_indices[b],
                additional_tokens=additional_tokens[b]
            )
            token_embeddings_list.append(embeddings)

        token_embeddings = np.stack(token_embeddings_list, axis=0)
        token_embeddings = Adapter.to_torch_tensor(token_embeddings).to(torch.float32)


        param_masks = Adapter.to_torch_tensor(samples["param_masks"]).to(torch.float32)
        param_matrices = Adapter.to_torch_tensor(samples["param_matrices"]).to(torch.float32)

        out = {
            "input_data": input_data,
            "param_indices": parameter_indices,
            "regressor_indices": regressor_indices,
            "param_masks": param_masks,
            "param_matrices": param_matrices,
            "token_embeddings": token_embeddings
        }

        return Adapter.to_device(out, device)

    @staticmethod
    def adapt_v2(
        samples: dict,
        intrinsic_params: list[str],
        device: str | torch.device = torch.device("cuda"),
    ) -> dict:
        # Fetch inputs from samples and adapt
        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]

        batch_size, max_num_obs, max_num_cols = design_matrices.shape

        input_data = Adapter.stack([design_matrices, rts, choices], axis=-1)
        input_data = Adapter.to_torch_tensor(input_data).to(torch.float32)

        # Build indices
        parameter_indices = [Adapter.build_parameter_indices(
            intrinsic_params,
            num_regressors=samples["max_num_regressors"],
            num_categories=samples["max_num_categories"],
        ) for _ in range(batch_size)]
        parameter_indices = Adapter.to_torch_tensor(np.array(parameter_indices)).to(torch.float32)

        regressor_indices = [Adapter.build_regressor_indices(
            intrinsic_params,
            num_regressors=samples["max_num_regressors"],
            num_categories=samples["max_num_categories"],
        ) for _ in range(batch_size)]
        regressor_indices = Adapter.to_torch_tensor(np.array(regressor_indices)).to(torch.float32)

        # TODO - This is very, very wrong
        num_params = len(intrinsic_params)
        block = samples["max_num_categories"] - 1
        keep_intercept = samples.get("keep_intercept", True)

        # regressor_id per column (B, C) with 0=intercept, 1..R
        regressor_id = np.zeros((batch_size, max_num_cols), dtype=np.float32)
        col = 0
        if keep_intercept:
            regressor_id[:, col] = 0.0
            col += 1
        for r in range(samples["max_num_regressors"]):
            regressor_id[:, col:col + block] = r + 1
            col += block

        # Broadcast to encoder time dimension: (B, T, C)
        regressor_id_3d = np.broadcast_to(regressor_id[:, None, :], (batch_size, max_num_obs, max_num_cols))

        encoder_input = np.stack([design_matrices, regressor_id_3d], axis=-1).astype(np.float32)

        # param_indices: last dim 0..P-1 for every column
        param_idx = np.tile(np.arange(num_params, dtype=np.float32)[None, None, :], (batch_size, max_num_cols, 1))

        # regressor_indices: duplicate each column’s regressor id across parameters
        reg_idx = np.repeat(regressor_id[:, :, None], num_params, axis=2).astype(np.float32)

        input_data = torch.from_numpy(encoder_input).to(device)
        param_indices = torch.from_numpy(param_idx).to(device)
        regressor_indices = torch.from_numpy(reg_idx).to(device)

        # rts/choices
        rts = torch.from_numpy(samples["sim_data"]["rts"]).to(device)
        choices = torch.from_numpy(samples["sim_data"]["choices"]).to(device)

        # Convert to torch tensor
        param_masks = torch.from_numpy(samples["param_masks"]).to(device)
        param_matrices = torch.from_numpy(samples["param_matrices"]).to(device)

        # Return (keep both key spellings for compatibility with trainer)
        out = {
            "input_data": input_data,
            "param_indices": param_indices,
            "regressor_indices": regressor_indices,
            "param_masks": param_masks,               # for later use in trainer
            "param_matrices": param_matrices,
            "rts": rts,
            "choices": choices,
        }
        return out
