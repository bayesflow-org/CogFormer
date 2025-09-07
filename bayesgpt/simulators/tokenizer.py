import numpy as np
from typing import Optional, Union, Callable, Dict, Tuple
from collections.abc import Mapping


class Tokenizer:
    """
    Tri-state tokenizer over a global superset of parameters with optional
    per-parameter dimensionality.

    This class builds a flattened representation of parameters for a single simulation,
    exposing three main artifacts:
    1. **mask**: Encodes whether each slot is inactive (-1), fixed (0), or free (1).
    2. **base_values**: Aligned vector with free slots = 1.0, fixed slots = user values,
       inactive slots = 0.0.
    3. **sampling/combine API**: Draws values for free parameters and constructs
       per-parameter dictionaries for simulation.

    Parameters
    ----------
    parameter_names : list of str
        Global ordered superset of parameter names.
    variant_parameters : set of str
        Subset of parameters used by the variant.
    fixed_parameters : dict of {str: float, np.ndarray, or Callable[[], np.ndarray]}
        Fixed values or generators for parameters, returning shape (dims[p],).
    free_parameters : dict of {str: Callable[[Optional[np.ndarray]], np.ndarray]}
        Sampling functions for free parameters, returning shape (dims[p],).
    parameter_dims : dict of {str: int}
        Per-parameter dimensionality (default 1).
    constraints : dict of {str: Callable[[np.ndarray], np.ndarray]}
        Constraint functions for free parameters (e.g., clip for positivity).
    context_shape : tuple of int, optional
        Expected shape of context array for a single simulation.
    """

    def __init__(
        self,
        *,
        parameter_names: list[str],
        variant_parameters: set[str],
        fixed_parameters: Mapping[str, Union[float, np.ndarray, Callable[[], np.ndarray]]] | None = None,
        free_parameters: Mapping[str, Callable[[Optional[np.ndarray]], np.ndarray]] | None = None,
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
        self._cache = {"mask": None, "base_values": None, "inference_conditions": None}  # Cache for efficiency

        # Validate inputs and set up parameter processing
        self.validate_parameters()
        self.parameter_slices, self.total_slots = self.build_parameter_slices()
        self.fixed_parameter_vectors = self.build_fixed_parameter_vectors()
        self.mask = self.build_mask()
        self.base_values = self.build_base_values()

        # Convenience sets for parameter tracking
        self.used_params = {n for n in self.parameter_names if n in self.variant_parameters}  # Active parameters
        self.fixed_params = {n for n in self.used_params if n in self.fixed_parameters}  # Fixed parameters
        self.free_params = {n for n in self.used_params if n not in self.fixed_parameters}  # Free parameters

    def sample(
            self,
            num_samples: int,
            context: Optional[np.ndarray] = None
    ) -> dict[str, np.ndarray]:
        """
        Sample all free parameters for num_samples simulations, applying constraints if provided.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        context : np.ndarray, optional
            Conditioning array for samplers, shape (context_shape,).

        Returns
        -------
        dict of {str: np.ndarray}
            Mapping from free parameter name to array of shape (num_samples, dims[p]).

        Raises
        ------
        ValueError
            If context shape is invalid or sampler output shapes mismatch.
        """
        # Validate context shape if provided
        if context is not None and self.context_shape is not None:
            if context.shape != self.context_shape:
                raise ValueError(f"Context must have shape {self.context_shape}; got {context.shape}")

        out: dict[str, np.ndarray] = {}
        for name in self.free_params:
            dim = self.get_parameter_dims(name)
            # Sample num_samples times, ensuring shape (num_samples, dim)
            arr = np.array(
                [self.free_parameters.get(name, lambda c: np.random.randn(dim))(context) for _ in range(num_samples)])
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != (num_samples, dim):
                arr = arr.reshape(num_samples, dim)
            if name in self.constraints:
                arr = self.constraints[name](arr)
            if arr.shape != (num_samples, dim):
                raise ValueError(f"Sampler for '{name}' must return shape ({num_samples}, {dim}); got {arr.shape}")
            out[name] = arr
        return out

    def combine(
            self,
            sampled: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Build a dictionary of parameter values for multiple simulations.

        Parameters
        ----------
        sampled : dict of {str: np.ndarray}
            Samples for free parameters, shape (num_samples, dims[p]).

        Returns
        -------
        dict of {str: np.ndarray}
            Mapping from used parameter name to array of shape (num_samples, dims[name]).

        Raises
        ------
        ValueError
            If sampled or fixed parameter shapes are incorrect.
        """
        out: dict[str, np.ndarray] = {}
        num_samples = next(iter(sampled.values())).shape[0] if sampled else 1
        for name in self.used_params:
            dim = self.get_parameter_dims(name)
            slot_mask = self.mask[self.parameter_slices[name]][0]
            if slot_mask == 1.0:  # Free parameter
                if name in sampled:
                    arr = np.asarray(sampled[name], dtype=np.float32)
                    if arr.shape != (num_samples, dim):
                        raise ValueError(f"Sample for '{name}' must have shape ({num_samples}, {dim}); got {arr.shape}")
                else:
                    arr = np.random.randn(num_samples, dim).astype(np.float32)
                out[name] = arr
            elif slot_mask == 0.0:  # Fixed parameter
                if callable(self.fixed_parameters[name]):
                    arr = np.asarray(self.fixed_parameters[name](), dtype=np.float32)
                    if arr.shape != (dim,):
                        raise ValueError(f"Fixed generator for '{name}' must return shape ({dim},)")
                else:
                    arr = self.fixed_parameter_vectors[name].astype(np.float32)
                # Broadcast fixed parameters to match num_samples
                arr = np.repeat(arr[None, :], num_samples, axis=0)
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

    def validate_parameters(self) -> None:
        """
        Validate parameter names, overlaps, and constraints.
        """
        # Check for unknown parameters
        unknown_fixed_parameters = set(self.fixed_parameters) - set(self.parameter_names)
        unknown_free_parameters = set(self.free_parameters) - set(self.parameter_names)
        unknown_constraints = set(self.constraints) - set(self.parameter_names)
        if unknown_fixed_parameters:
            raise ValueError(f"Fixed parameters not in superset: {sorted(unknown_fixed_parameters)}")
        if unknown_free_parameters:
            raise ValueError(f"Free parameters not in superset: {sorted(unknown_free_parameters)}")
        if unknown_constraints:
            raise ValueError(f"Constraints not in superset: {sorted(unknown_constraints)}")

        # Check for parameter overlaps
        overlap = set(self.fixed_parameters) & set(self.free_parameters)
        if overlap:
            raise ValueError(f"Cannot be both fixed and free: {sorted(overlap)}")

    def build_parameter_slices(self) -> Tuple[Dict[str, slice], int]:
        """
        Build flattened slices for each parameter and compute total slots.
        """
        parameter_slices: Dict[str, slice] = {}
        offset = 0  # Track current slot position

        for name in self.parameter_names:
            # Get parameter dimension (default 1)
            dim = int(self.parameter_dims.get(name, 1))
            if dim <= 0:
                raise ValueError(f"parameter_dims[{name}] must be >= 1 (got {dim}).")
            # Assign slice for parameter
            parameter_slices[name] = slice(offset, offset + dim)
            offset += dim
        return parameter_slices, offset

    def build_fixed_parameter_vectors(self) -> Dict[str, np.ndarray]:
        """
        Normalize fixed parameter values into arrays.
        """
        fixed_parameter_vectors: Dict[str, np.ndarray] = {}
        for name, val in self.fixed_parameters.items():
            # Get parameter dimension
            dim = self.get_parameter_dims(name)
            if callable(val):
                arr = np.asarray(val(), dtype=np.float32)
                if arr.shape != (dim,):
                    raise ValueError(f"Fixed generator for '{name}' must return shape ({dim},); got {arr.shape}")
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
        mask = np.full((self.total_slots,), -1.0, dtype=np.float32)  # Initialize all slots as inactive
        for name in self.parameter_names:
            sl = self.parameter_slices[name]
            if name not in self.variant_parameters:
                mask[sl] = -1.0  # Inactive parameters
            elif name in self.fixed_parameters:
                mask[sl] = 0.0  # Fixed parameters
            else:
                mask[sl] = 1.0  # Free parameters
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
                base[sl] = 0.0  # Inactive slots
            elif slot_mask == 0.0:
                base[sl] = self.fixed_parameter_vectors[name]  # Fixed values
            else:
                base[sl] = 1.0  # Free slots
        return base

    def build_inference_conditions(
        self,
        *,
        one_hot_variant: np.ndarray = None,
        context: np.ndarray = None,
        include_variant: bool = True,
        include_context: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Construct inference-condition tensors for a single simulation.

        Parameters
        ----------
        one_hot_variant : np.ndarray, optional
            One-hot encoded variant, shape (num_variants,).
        context : np.ndarray, optional
            Context variables, shape (context_shape,).
        include_variant : bool
            Whether to include variant encoder.
        include_context : bool
            Whether to include context encoder.

        Returns
        -------
        dict with keys:
        - mask : np.ndarray of shape (D,)
        - base_values : np.ndarray of shape (D,)
        - variant : np.ndarray, optional, shape (num_variants,)
        - context : np.ndarray, optional, shape (context_shape,)
        - full_conditions : np.ndarray of shape (2*D + variant/context dims,)

        Raises
        ------
        ValueError
            If context or variant encoder shapes are invalid.
        """
        mask = self.mask  # Get tri-state mask
        base_values = self.base_values  # Get base values
        full_conditions = [mask, base_values]  # Initialize conditions list

        inference_conditions = {
            "mask": mask,
            "base_values": base_values,
        }

        if include_variant and one_hot_variant is not None:
            inference_conditions["variant"] = one_hot_variant.astype(np.float32)
            full_conditions.append(one_hot_variant)

        if include_context and context is not None:
            if self.context_shape is not None and context.shape != self.context_shape:
                raise ValueError(f"Context encoder must have shape {self.context_shape}")
            inference_conditions["context"] = context.astype(np.float32)
            full_conditions.append(context)

        inference_conditions["full_conditions"] = np.concatenate(full_conditions)
        return inference_conditions

    def get_mask(self) -> np.ndarray:
        """
        Returns the tri-state mask for parameter roles.

        Returns
        -------
        np.ndarray of shape (num_parameters,)
            Tri-state mask (-1.0: inactive, 0.0: fixed, 1.0: free).
        """
        return self.mask

    def get_base_values(self) -> np.ndarray:
        """
        Returns the conditioning vector with fixed/default values.

        Returns
        -------
        np.ndarray of shape (num_parameters,)
            Conditioning vector with fixed/default values.
        """
        return self.base_values

    def get_parameter_dims(self, name: str) -> int:
        """
        Return the flattened slot length for a parameter.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        int
            Number of slots for the parameter.
        """
        sl = self.parameter_slices[name]
        return sl.stop - sl.start

    def _clear_cache(self):
        """
        Clear cached arrays to free memory or reset state.
        """
        self._cache = {"mask": None, "base_values": None, "inference_conditions": None}
