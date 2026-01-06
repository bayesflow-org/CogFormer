import torch
import numpy as np
from collections.abc import Mapping, Sequence
from typing import Iterable


class Adapter:
    """Utilities for dtype conversion, 2D normalization, and safe concatenation."""

    @staticmethod
    def asarray(x) -> np.ndarray:
        return x if isinstance(x, np.ndarray) else np.asarray(x)

    @staticmethod
    def pad(x: np.ndarray, to_shape: tuple[int, ...], pad_value: float = 0.0) -> np.ndarray:

        if x.shape == to_shape:
            return x

        if len(to_shape) != x.ndim:
            raise ValueError(f"ndim mismatch: {x.ndim} vs target {len(to_shape)}")

        pads = []

        for curr, targ in zip(x.shape, to_shape):

            if targ < curr:
                raise ValueError(f"Cannot pad to smaller size: {x.shape} -> {to_shape}")

            pads.append((0, targ - curr))

        return np.pad(x, pads, mode="constant", constant_values=pad_value)

    @staticmethod
    def concatenate(
        arrays: Iterable,
        axis: int = 0,
        dtype: np.dtype | None = None,
        pad: bool = False,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """
        Concatenate arrays along axis. If pad=True, pad other dims to max size.
        """
        arrs = [Adapter.asarray(a) for a in arrays]

        if not arrs:
            return np.array([], dtype=dtype if dtype is not None else np.float32)

        if dtype is not None:
            arrs = [a.astype(dtype, copy=False, casting="same_kind") for a in arrs]

        if pad:
            ndim = arrs[0].ndim
            if any(a.ndim != ndim for a in arrs):
                raise ValueError("All arrays must have the same ndim when padding.")
            target = list(arrs[0].shape)
            for a in arrs[1:]:
                target = [max(s, t) if ax != axis else t for ax, (s, t) in enumerate(zip(a.shape, target))]
            padded = []
            for a in arrs:
                tgt = list(target)
                tgt[axis] = a.shape[axis]
                padded.append(Adapter.pad(a, tuple(tgt), pad_value=pad_value))
            arrs = padded

        out = np.concatenate(arrs, axis=axis)
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype, copy=False, casting="same_kind")
        return out

    @staticmethod
    def stack(x: dict[str: np.ndarray] | list[np.ndarray], axis=-1) -> np.ndarray:
        if isinstance(x, dict):
            shapes = [v.shape[0] for v in x.values()]
        else:
            shapes = [v.shape[0] for v in x]

        assert len(set(shapes)) == 1, "Arrays must have the same shape."
        return Adapter.concatenate(list(x.values()) if isinstance(x, dict) else x, axis=axis)

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
