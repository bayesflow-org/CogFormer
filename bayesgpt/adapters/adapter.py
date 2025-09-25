import numpy as np
from collections.abc import Mapping, Sequence
from typing import Iterable


class Adapter:
    """Utilities for dtype conversion, 2D normalization, and safe concatenation."""

    @staticmethod
    def convert_dtype(
        x,
        dtype: np.dtype = np.float32,
        copy: bool = False,
    ):
        """Cast to dtype (supports ndarray, Mapping, Sequence)."""
        if isinstance(x, Mapping):
            return Adapter.apply_mapping(x, lambda v: Adapter.convert_one(v, dtype, copy))
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return Adapter.apply_sequence(x, lambda v: Adapter.convert_one(v, dtype, copy))
        return Adapter._convert_one(x, dtype, copy)

    @staticmethod
    def atleast_2d(x, orientation: str = "row"):
        """Ensure ≥2D; 1D -> (1,N) if 'row', else (N,1)."""
        if orientation not in ("row", "col"):
            raise ValueError("orientation must be 'row' or 'col'")
        if isinstance(x, Mapping):
            return Adapter.apply_mapping(x, lambda v: Adapter.to_2d(v, orientation))
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return Adapter.apply_sequence(x, lambda v: Adapter.to_2d(v, orientation))
        return Adapter.to_2d(x, orientation)

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
                padded.append(Adapter.pad_to_shape(a, tuple(tgt), pad_value=pad_value))
            arrs = padded

        out = np.concatenate(arrs, axis=axis)
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype, copy=False, casting="same_kind")
        return out

    # ----------------- helpers -----------------
    @staticmethod
    def asarray(x) -> np.ndarray:
        return x if isinstance(x, np.ndarray) else np.asarray(x)

    @staticmethod
    def apply_mapping(d: Mapping, fn):
        return {k: fn(v) for k, v in d.items()}

    @staticmethod
    def apply_sequence(seq: Sequence, fn):
        return [fn(v) for v in seq]

    @staticmethod
    def pad_to_shape(x: np.ndarray, target_shape: tuple[int, ...], pad_value: float = 0.0) -> np.ndarray:
        if x.shape == target_shape:
            return x
        if len(target_shape) != x.ndim:
            raise ValueError(f"ndim mismatch: {x.ndim} vs target {len(target_shape)}")
        pads = []
        for curr, targ in zip(x.shape, target_shape):
            if targ < curr:
                raise ValueError(f"Cannot pad to smaller size: {x.shape} -> {target_shape}")
            pads.append((0, targ - curr))
        return np.pad(x, pads, mode="constant", constant_values=pad_value)

    @staticmethod
    def _convert_one(v, dtype, copy):
        a = Adapter.asarray(v)
        return a.astype(dtype, copy=copy, casting="same_kind")

    @staticmethod
    def to_2d(v, orientation: str):
        a = Adapter.asarray(v)
        if a.ndim >= 2:
            return a
        if a.ndim == 0:
            return a.reshape(1, 1)
        return a.reshape(1, -1) if orientation == "row" else a.reshape(-1, 1)
