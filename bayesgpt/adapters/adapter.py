import torch
import numpy as np
from collections.abc import Mapping, Sequence
from typing import Iterable


class Adapter:
    """Utilities for dtype conversion and safe array manipulation."""

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
        """Concatenate arrays along axis. If pad=True, pad other dims to max size."""
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
    def stack(x: dict[str, np.ndarray] | list[np.ndarray], axis=-1) -> np.ndarray:
        arrays = list(x.values()) if isinstance(x, dict) else x
        batch_sizes = [a.shape[0] for a in arrays]
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"All arrays must have the same batch size; got {batch_sizes}.")
        return Adapter.concatenate(arrays, axis=axis)

    @staticmethod
    def build_parameter_indices(
        num_params: int,
        num_regressors: int,
        num_categories: int,
    ) -> np.ndarray:
        """Build per-token parameter position indices. Shape: (num_tokens, 1)."""
        num_total_regressors = num_regressors * (num_regressors + 1) // 2
        num_tiles = num_total_regressors * (num_categories - 1) + 1
        indices = np.tile(np.linspace(0.0, 1.0, num_params), num_tiles)[..., None]
        return indices

    @staticmethod
    def build_regressor_indices(
        num_params: int,
        num_regressors: int,
        num_categories: int,
    ) -> np.ndarray:
        """Build per-token regressor position indices. Shape: (num_tokens, 1)."""
        num_total_regressors = num_regressors * (num_regressors + 1) // 2
        num_indices = num_total_regressors * (num_categories - 1) + 1
        indices = np.repeat(np.linspace(0.0, 1.0, num_indices), num_params)[..., None]
        return indices

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
        num_params: int | None = None,
    ) -> dict:
        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]

        batch_size = design_matrices.shape[0]

        input_data = Adapter.stack([design_matrices, rts, choices], axis=-1)
        input_data = Adapter.to_torch_tensor(input_data).to(torch.float32)

        # num_params overrides intrinsic_params length when called from a ModelClass pipeline
        n_params = num_params if num_params is not None else len(intrinsic_params)

        # Build indices once and tile across batch (identical per element)
        parameter_indices = Adapter.to_torch_tensor(
            np.tile(
                Adapter.build_parameter_indices(
                    num_params=n_params,
                    num_regressors=samples["max_num_regressors"],
                    num_categories=samples["max_num_categories"],
                ),
                (batch_size, 1, 1),
            )
        ).to(torch.float32)

        regressor_indices = Adapter.to_torch_tensor(
            np.tile(
                Adapter.build_regressor_indices(
                    num_params=n_params,
                    num_regressors=samples["max_num_regressors"],
                    num_categories=samples["max_num_categories"],
                ),
                (batch_size, 1, 1),
            )
        ).to(torch.float32)

        param_masks = Adapter.to_torch_tensor(samples["param_masks"]).to(torch.float32)
        param_matrices = Adapter.to_torch_tensor(samples["param_matrices"]).to(torch.float32)

        raw_ids = samples.get("model_ids", None)
        model_ids = (
            Adapter.to_torch_tensor(np.asarray(raw_ids, dtype=np.int64))
            if raw_ids is not None else None
        )

        out = {
            "input_data": input_data,
            "param_indices": parameter_indices,
            "regressor_indices": regressor_indices,
            "param_masks": param_masks,
            "param_matrices": param_matrices,
            "model_ids": model_ids,
        }

        return Adapter.to_device(out, device)
