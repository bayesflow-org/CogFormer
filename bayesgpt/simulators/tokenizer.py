import numpy as np
from typing import Optional, Union, Callable, Dict, Tuple
from collections.abc import Mapping
import matplotlib.pyplot as plt

class Tokenizer:
    """
    Tri-state tokenizer over a global superset of parameters with optional
    per-parameter dimensionality.

    This class builds a *flattened* representation of parameters used by a
    model variant and exposes three main artifacts:

    1. **mask**: encodes whether each slot is inactive (-1), fixed (0), or free (1).
    2. **base_values**: aligned vector with free slots = 1.0, fixed slots = user values,
       inactive slots = 0.0.
    3. **sampling/combine API**: draws values for free parameters and constructs
       per-parameter batch dictionaries for simulation.

    Parameters
    ----------
    parameter_names : list of str
        Global ordered superset of parameter names.
    variant_parameters : set of str
        Subset of parameters used by the variant.
    fixed_parameters : dict of {str: float, np.ndarray, or Callable[[int], np.ndarray]}
        Fixed values or generators for parameters.
    free_parameters : dict of {str: Callable[[int, Optional[np.ndarray]], np.ndarray]}
        Sampling functions for free parameters.
    parameter_dims : dict of {str: int}
        Per-parameter dimensionality (default 1).
    constraints : dict of {str: Callable[[np.ndarray], np.ndarray]}
        Constraint functions for free parameters (e.g., clip for positivity).
    context_shape : tuple of int, optional
        Expected shape of context array (excluding batch dimension).
    """

    def __init__(
        self,
        *,
        parameter_names: list[str],
        variant_parameters: set[str],
        fixed_parameters: Mapping[str, Union[float, np.ndarray, Callable[[int], np.ndarray]]] | None = None,
        free_parameters: Mapping[str, Callable[[int, Optional[np.ndarray]], np.ndarray]] | None = None,
        parameter_dims: dict[str, int] | None = None,
        constraints: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
        context_shape: tuple[int, ...] | None = None,
    ) -> None:
        # Initialize basic attributes
        self.parameter_names = list(parameter_names)
        self.variant_parameters = set(variant_parameters)
        self.fixed_parameters = dict(fixed_parameters or {})
        self.free_parameters = dict(free_parameters or {})
        self.parameter_dims = dict(parameter_dims or {})
        self.constraints = dict(constraints or {})
        self.context_shape = context_shape
        self._cache = {"mask": {}, "base_values": {}, "inference_conditions": {}}

        # Validate inputs and set up parameter processing
        self.validate_parameters()
        self.parameter_slices, self.total_slots = self.build_parameter_slices()
        self.fixed_parameter_vectors = self.build_fixed_parameter_vectors()
        self.mask = self.build_mask()
        self.base_values = self.build_base_values()

        # Convenience sets
        self.used_params = {n for n in self.parameter_names if n in self.variant_parameters}
        self.fixed_params = {n for n in self.used_params if n in self.fixed_parameters}
        self.free_params = {n for n in self.used_params if n not in self.fixed_parameters}

    def sample(
        self, batch_size: int, context: Optional[np.ndarray] = None
    ) -> dict[str, np.ndarray]:
        """
        Sample all free parameters, applying constraints if provided.

        Parameters
        ----------
        batch_size : int
            Number of samples per parameter.
        context : np.ndarray, optional
            Conditioning array for samplers.

        Returns
        -------
        dict of {str: np.ndarray}
            Mapping from free parameter name to array of shape (batch_size, dims[p]).
        """
        if context is not None and self.context_shape is not None:
            if context.shape != (batch_size, *self.context_shape):
                raise ValueError(f"Context must have shape ({batch_size}, {self.context_shape}); got {context.shape}")

        out: dict[str, np.ndarray] = {}
        for name in self.free_params:
            dim = self.get_parameter_dims(name)
            arr = self.free_parameters.get(name, lambda b, c: np.random.randn(b, dim))(batch_size, context)
            arr = np.asarray(arr, dtype=np.float32)
            if name in self.constraints:
                arr = self.constraints[name](arr)
            if arr.shape != (batch_size, dim):
                raise ValueError(f"Sampler for '{name}' must return shape ({batch_size}, {dim}); got {arr.shape}")
            out[name] = arr
        return out

    def combine(
        self, sampled: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        """
        Build a batch dictionary of parameter values for simulation.

        Parameters
        ----------
        sampled : dict of {str: np.ndarray}
            Samples for free parameters, shape (batch_size, dims[p]).
        batch_size : int
            Number of rows to produce.

        Returns
        -------
        dict of {str: np.ndarray}
            Mapping from used parameter name to array of shape (batch_size, dims[name]).
        """
        out: dict[str, np.ndarray] = {}
        for name in self.used_params:
            dim = self.get_parameter_dims(name)
            slot_mask = self.mask[self.parameter_slices[name]][0]
            if slot_mask == 1.0:
                if name in sampled:
                    arr = np.asarray(sampled[name], dtype=np.float32)
                    if arr.shape != (batch_size, dim):
                        raise ValueError(f"Sample for '{name}' must have shape ({batch_size}, {dim}); got {arr.shape}")
                else:
                    arr = np.random.randn(batch_size, dim).astype(np.float32)
                out[name] = arr
            elif slot_mask == 0.0:
                if callable(self.fixed_parameters[name]):
                    arr = np.asarray(self.fixed_parameters[name](batch_size), dtype=np.float32)
                    if arr.shape != (batch_size, dim):
                        raise ValueError(f"Fixed generator for '{name}' must return shape ({batch_size}, {dim})")
                else:
                    arr = np.repeat(self.fixed_parameter_vectors[name][None, :], batch_size, axis=0).astype(np.float32)
                out[name] = arr
        return out

    def summarize(self) -> Dict[str, Dict]:
        """
        Returns a summary of the parameter schema and states.

        Returns
        -------
        dict of {str: dict}
            Mapping of parameter names to their state, value, and dimension.
        """
        summary = {}
        for name in self.parameter_names:
            sl = self.parameter_slices[name]
            state = {-1.0: "inactive", 0.0: "fixed", 1.0: "free"}[float(self.mask[sl][0])]
            value = self.base_values[sl][0] if state != "free" else "sampled"
            summary[name] = {"state": state, "value": value, "dim": self.get_parameter_dims(name)}
        return summary

    def visualize_mask(self, save_path: Optional[str] = None) -> None:
        """
        Visualizes the tri-state mask as a heatmap.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        """
        mask = self.mask.reshape(1, -1)
        plt.figure(figsize=(len(self.parameter_names) * 0.5, 1))
        plt.imshow(mask, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
        plt.xticks(np.arange(len(self.parameter_names)), self.parameter_names, rotation=45)
        plt.yticks([])
        plt.colorbar(label="Mask (-1: inactive, 0: fixed, 1: free)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def validate_parameters(self) -> None:
        """
        Validate parameter names, overlaps, and constraints.
        """
        # Check unknowns
        unknown_fixed_parameters = set(self.fixed_parameters) - set(self.parameter_names)
        unknown_free_parameters = set(self.free_parameters) - set(self.parameter_names)
        unknown_constraints = set(self.constraints) - set(self.parameter_names)
        if unknown_fixed_parameters:
            raise ValueError(f"Fixed parameters not in superset: {sorted(unknown_fixed_parameters)}")
        if unknown_free_parameters:
            raise ValueError(f"Free parameters not in superset: {sorted(unknown_free_parameters)}")
        if unknown_constraints:
            raise ValueError(f"Constraints not in superset: {sorted(unknown_constraints)}")

        # Check overlaps
        overlap = set(self.fixed_parameters) & set(self.free_parameters)
        if overlap:
            raise ValueError(f"Cannot be both fixed and free: {sorted(overlap)}")

    def build_parameter_slices(self) -> Tuple[Dict[str, slice], int]:
        """
        Build flattened slices for each parameter and compute total slots.
        """
        # Initialize
        parameter_slices: Dict[str, slice] = {}

        # Keeping track of offsets
        offset = 0

        # Loop and build slices
        for name in self.parameter_names:
            dim = int(self.parameter_dims.get(name, 1))
            # Check if the user puts in negative dims for parameters
            if dim <= 0:
                raise ValueError(f"parameter_dims[{name}] must be >= 1 (got {dim}).")
            parameter_slices[name] = slice(offset, offset + dim)
            offset += dim
        return parameter_slices, offset

    def build_fixed_parameter_vectors(self) -> Dict[str, np.ndarray]:
        """
        Normalize fixed parameter values into arrays.
        """
        fixed_parameter_vectors: Dict[str, np.ndarray] = {}
        for name, val in self.fixed_parameters.items():
            dim = self.get_parameter_dims(name)
            if callable(val):
                arr = np.asarray(val(1), dtype=np.float32)
                if arr.shape != (1, dim):
                    raise ValueError(f"Fixed generator for '{name}' must return shape (1, {dim}); got {arr.shape}")
            else:
                arr = np.asarray(val, dtype=np.float32)
                if arr.shape == ():
                    arr = np.full((dim,), float(arr), dtype=np.float32)
                elif arr.shape != (dim,):
                    raise ValueError(f"Fixed value for '{name}' must have shape ({dim},) or be scalar; got {arr.shape}")
            fixed_parameter_vectors[name] = arr
        return fixed_parameter_vectors

    def build_mask(self) -> np.ndarray:
        """
        Build tri-state mask for parameter roles.
        """
        mask = np.full((self.total_slots,), -1.0, dtype=np.float32)
        for name in self.parameter_names:
            sl = self.parameter_slices[name]
            if name not in self.variant_parameters:
                mask[sl] = -1.0
            elif name in self.fixed_parameters:
                mask[sl] = 0.0
            else:
                mask[sl] = 1.0
        return mask

    def build_base_values(self) -> np.ndarray:
        """
        Build base values aligned with mask.
        """
        base = np.zeros((self.total_slots,), dtype=np.float32)
        for name in self.parameter_names:
            sl = self.parameter_slices[name]
            slot_mask = self.mask[sl][0]
            if slot_mask == -1.0:
                base[sl] = 0.0
            elif slot_mask == 0.0:
                base[sl] = self.fixed_parameter_vectors[name]
            else:
                base[sl] = 1.0
        return base

    def build_inference_conditions(
        self,
        batch_size: int,
        *,
        one_hot_variant: np.ndarray = None,
        context: np.ndarray = None,
        include_variant: bool = True,
        include_context: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Construct batched inference-condition tensors.

        Parameters
        ----------
        batch_size : int
            Number of rows to produce.
        one_hot_variant : np.ndarray, optional
            One-hot encoded variant.
        context : np.ndarray, optional
            Context variables.
        include_variant : bool
            Whether to include variant encoder.
        include_context : bool
            Whether to include context encoder.

        Returns
        -------
        dict with keys:
        - mask : np.ndarray of shape (batch_size, D)
        - base_values : np.ndarray of shape (batch_size, D)
        - variant : np.ndarray, optional
        - context : np.ndarray, optional
        - full_conditions : np.ndarray of shape (batch_size, 2*D + variant/context dims)
        """
        cache_key = (batch_size, include_variant, include_context)
        if cache_key in self._cache["inference_conditions"]:
            return self._cache["inference_conditions"][cache_key]

        batched_mask = self.get_mask(batch_size)
        batched_base = self.get_base_values(batch_size)
        full_conditions = [batched_mask, batched_base]

        inference_conditions = {
            "mask": batched_mask,
            "base_values": batched_base,
        }

        if include_variant and one_hot_variant is not None:
            inference_conditions["variant"] = one_hot_variant.astype(np.float32)
            full_conditions.append(one_hot_variant)

        if include_context and context is not None:
            if self.context_shape is not None and context.shape != (batch_size, *self.context_shape):
                raise ValueError(f"Context encoder must have shape ({batch_size}, {self.context_shape})")
            inference_conditions["context"] = context.astype(np.float32)
            full_conditions.append(context)

        inference_conditions["full_conditions"] = np.concatenate(full_conditions, axis=1)
        self._cache["inference_conditions"][cache_key] = inference_conditions
        return inference_conditions

    def get_mask(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated tri-state mask for parameter roles.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Tri-state mask (-1.0: inactive, 0.0: fixed, 1.0: free).
        """
        if batch_size not in self._cache["mask"]:
            self._cache["mask"][batch_size] = np.tile(self.mask, (batch_size, 1)).astype(np.float32)
        return self._cache["mask"][batch_size]

    def get_base_values(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated conditioning vector with fixed/default values.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Conditioning vector with fixed/default values.
        """
        if batch_size not in self._cache["base_values"]:
            self._cache["base_values"][batch_size] = np.tile(self.base_values, (batch_size, 1)).astype(np.float32)
        return self._cache["base_values"][batch_size]

    def get_parameter_dims(self, name: str) -> int:
        """
        Return the flattened slot length for a parameter.
        """
        sl = self.parameter_slices[name]
        return sl.stop - sl.start

    def _clear_cache(self):
        """
        Clear cached arrays to free memory or reset state.
        """
        self._cache = {"mask": {}, "base_values": {}, "inference_conditions": {}}
        