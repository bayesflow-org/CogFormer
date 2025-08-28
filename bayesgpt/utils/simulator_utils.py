import numpy as np
from typing import Optional, Union
from collections.abc import Callable

import numpy as np
from collections.abc import Callable
from typing import Optional


class Tokenizer:
    """
    Tri-state tokenizer over a global superset of parameters with optional
    per-parameter dimensionality.

    This class builds a *flattened* representation of parameters used by a
    model variant and exposes three main artifacts:

    1. **mask** (tri-state): encodes whether each slot is inactive, fixed,
       or free (values in ``{-1, 0, 1}``).
    2. **base_values**: aligned vector with the same length as ``mask``:
       free slots carry ``1.0``, fixed slots carry the user's fixed values,
       and inactive slots carry ``0.0``.
    3. **sampling/combine API**: draws values for free parameters (either via
       user-provided samplers or from a standard normal) and constructs a
       per-parameter batch dictionary suitable for simulation.

    **Mask semantics (per flattened slot)**

    - ``-1`` : Inactive — parameter is not used by this variant.
    - ``0``  : Fixed    — parameter is used and its value is fixed by user.
    - ``1``  : Free     — parameter is used and will be sampled (either via
      a user sampler or, by default, from :math:`\\mathcal{N}(0, I)`).

    **Residual rule**

    Any *used* parameter that is neither explicitly fixed nor declared with a
    user sampler is treated as **free** and sampled from a standard normal.

    Parameters
    ----------
    parameter_names : list of str
        Global ordered superset of parameter names. This order defines the
        flattened slot layout.
    variant_used_params : set of str
        Subset of ``parameter_names`` actually used by the variant.
        Parameters not in this set are marked inactive.
    fixed_parameters : dict of {str: array-like or float}, optional
        User-specified fixed values. Each value must match the parameter's
        dimensionality (see ``parameter_dims``). Scalars are broadcast.
    free_parameters : dict of {str: Callable}, optional
        Mapping from parameter name to a sampler function with signature
        ``sampler(batch_size: int, context: Optional[np.ndarray]) -> np.ndarray``.
        Each sampler must return an array of shape ``(batch_size, dims[p])``.
    parameter_dims : dict of {str: int}, optional
        Per-parameter dimensionality. Defaults to 1 for any name not present.
        All dimensions must be positive integers.
    dtype : numpy.dtype, optional
        Floating dtype for internal arrays. Defaults to ``np.float32``.

    Attributes
    ----------
    mask : np.ndarray, shape (D,), dtype float
        Tri-state mask over the flattened slots in superset order
        (after applying variant usage and fixed/free declarations).
    base_values : np.ndarray, shape (D,), dtype float
        Aligned vector: free slots = 1.0, fixed slots = user fixed values,
        inactive slots = 0.0.
    total_slots : int
        Total number of flattened slots, ``D = sum(parameter_dims[name])``.
    used_params : set of str
        Parameters used by this variant (i.e., active).
    free_params : set of str
        Subset of ``used_params`` that are free (explicit or residual).
    fixed_params : set of str
        Subset of ``used_params`` that are fixed.

    Notes
    -----
    - All arrays produced by this class use ``dtype``.
    - The flattened layout concatenates per-parameter slots following
      ``parameter_names`` order. All slots of a given parameter share the
      same mask state.
    - This class does not manage variant or context embeddings. Those can be
      appended externally if needed.

    Examples
    --------
    >>> names = ["v", "a", "z", "tau", "angle"]
    >>> used = {"v", "a", "z", "tau"}  # 'angle' inactive
    >>> fixed = {"a": 1.2, "tau": 0.30}
    >>> tok = Tokenizer(parameter_names=names,
    ...                 variant_used_params=used,
    ...                 fixed_parameters=fixed)
    >>> tok.mask
    array([ 1.,  0.,  1.,  0., -1.], dtype=float32)
    >>> tok.base_values
    array([1. , 1.2, 1. , 0.3, 0. ], dtype=float32)
    """

    def __init__(
        self,
        *,
        parameter_names: list[str],
        variant_used_params: set[str],
        fixed_parameters: dict[str, np.ndarray] | dict[str, float] | None = None,
        free_parameters: dict[str, Callable[[int, Optional[np.ndarray]], np.ndarray]]
        | None = None,
        parameter_dims: dict[str, int] | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.parameter_names = list(parameter_names)
        self.variant_used_params = set(variant_used_params)
        self.fixed_parameters = dict(fixed_parameters or {})
        self.free_parameters = dict(free_parameters or {})
        self.parameter_dims = dict(parameter_dims or {})
        self.dtype = dtype

        # Validate names and overlaps
        unknown_fixed = set(self.fixed_parameters) - set(self.parameter_names)
        unknown_free = set(self.free_parameters) - set(self.parameter_names)
        if unknown_fixed:
            raise ValueError(
                f"Fixed parameters not in superset: {sorted(unknown_fixed)}"
            )
        if unknown_free:
            raise ValueError(
                f"Free parameters not in superset: {sorted(unknown_free)}"
            )
        overlap = set(self.fixed_parameters) & set(self.free_parameters)
        if overlap:
            raise ValueError(f"Cannot be both fixed and free: {sorted(overlap)}")

        # Build flattened slices
        self._name_slices: dict[str, slice] = {}
        offset = 0
        for name in self.parameter_names:
            dim = int(self.parameter_dims.get(name, 1))
            if dim <= 0:
                raise ValueError(
                    f"parameter_dims[{name}] must be >= 1 (got {dim})."
                )
            self._name_slices[name] = slice(offset, offset + dim)
            offset += dim
        self.total_slots = offset

        # Normalize fixed vectors
        self._fixed_vectors: dict[str, np.ndarray] = {}
        for name, val in self.fixed_parameters.items():
            dim = self._slice_len(name)
            arr = np.asarray(val, dtype=self.dtype)
            if arr.shape == ():  # scalar -> broadcast
                arr = np.full((dim,), float(arr), dtype=self.dtype)
            elif arr.shape == (dim,):
                arr = arr.astype(self.dtype, copy=False)
            else:
                raise ValueError(
                    f"Fixed value for '{name}' must have shape ({dim},) "
                    f"or be scalar; got {arr.shape}."
                )
            self._fixed_vectors[name] = arr

        # Build tri-state mask
        mask = np.full((self.total_slots,), -1.0, dtype=self.dtype)
        for name in self.parameter_names:
            sl = self._name_slices[name]
            if name not in self.variant_used_params:
                mask[sl] = -1.0
            elif name in self.fixed_parameters:
                mask[sl] = 0.0
            else:
                # Used but not fixed -> FREE (explicit or residual)
                mask[sl] = 1.0
        self.mask = mask

        # Build base_values aligned with mask
        base = np.zeros((self.total_slots,), dtype=self.dtype)
        for name in self.parameter_names:
            sl = self._name_slices[name]
            slot_mask = self.mask[sl][0]  # all equal within the slice
            if slot_mask == -1.0:
                base[sl] = 0.0
            elif slot_mask == 0.0:
                base[sl] = self._fixed_vectors[name]
            else:  # slot_mask == 1.0
                base[sl] = 1.0
        self.base_values = base

        # Convenience sets
        self.used_params = {
            n for n in self.parameter_names if n in self.variant_used_params
        }
        self.fixed_params = {n for n in self.used_params if n in self.fixed_parameters}
        self.free_params = {n for n in self.used_params if n not in self.fixed_params}


    def sample(
        self, batch_size: int, context: Optional[np.ndarray] = None
    ) -> dict[str, np.ndarray]:
        """
        Sample all FREE (used) parameters.

        For each free parameter ``p``, this method uses the user-provided
        sampler if available; otherwise, it draws from a standard normal
        :math:`\\mathcal{N}(0, I)` with appropriate dimensionality.

        Parameters
        ----------
        batch_size : int
            Number of samples per parameter.
        context : np.ndarray, optional
            Optional conditioning array forwarded to user samplers.

        Returns
        -------
        dict of {str: np.ndarray}
            Mapping from free parameter name to an array of shape
            ``(batch_size, dims[p])`` with dtype ``self.dtype``.

        Raises
        ------
        ValueError
            If a sampler returns an array with an unexpected shape.
        """
        out: dict[str, np.ndarray] = {}
        for name in self.free_params:
            dim = self._slice_len(name)
            if name in self.free_parameters:
                arr = self.free_parameters[name](batch_size, context)
                arr = np.asarray(arr, dtype=self.dtype)
                if arr.shape != (batch_size, dim):
                    raise ValueError(
                        f"Sampler for '{name}' must return shape "
                        f"({batch_size}, {dim}); got {arr.shape}."
                    )
            else:
                arr = np.random.randn(batch_size, dim).astype(
                    self.dtype, copy=False
                )
            out[name] = arr
        return out

    def combine(
        self, sampled: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        """
        Build a batch dictionary of concrete parameter values for simulation.

        The output includes **only used** parameters. For each parameter:
        - FREE  → values are taken from ``sampled[name]`` if present; otherwise
          default samples are drawn from :math:`\\mathcal{N}(0, I)`.
        - FIXED → the user fixed vector is broadcast to shape
          ``(batch_size, dims[name])``.
        - INACTIVE → omitted.

        Parameters
        ----------
        sampled : dict of {str: np.ndarray}
            Samples for (some or all) free parameters. Each array must have
            shape ``(batch_size, dims[p])`` if provided.
        batch_size : int
            Number of rows to produce (used for broadcasting and validation).

        Returns
        -------
        dict of {str: np.ndarray}
            Mapping from used parameter name to an array shaped
            ``(batch_size, dims[name])``.

        Raises
        ------
        ValueError
            If a provided sample has an unexpected shape.
        """
        out: dict[str, np.ndarray] = {}
        for name in self.used_params:
            dim = self._slice_len(name)
            slot_mask = self.mask[self._name_slices[name]][0]
            if slot_mask == 1.0:  # FREE
                if name in sampled:
                    arr = np.asarray(sampled[name], dtype=self.dtype)
                    if arr.shape != (batch_size, dim):
                        raise ValueError(
                            f"Sample for '{name}' must have shape "
                            f"({batch_size}, {dim}); got {arr.shape}."
                        )
                else:
                    arr = np.random.randn(batch_size, dim).astype(
                        self.dtype, copy=False
                    )
                out[name] = arr
            elif slot_mask == 0.0:  # FIXED
                vec = self._fixed_vectors[name][None, :].astype(
                    self.dtype, copy=False
                )
                out[name] = np.repeat(vec, batch_size, axis=0)
            # slot_mask == -1.0 → skip
        return out

    def build_inference_conditions(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Construct batched inference-condition tensors.

        Returns a dictionary containing the tiled tri-state mask, the tiled
        base values, and a concatenated ``full_conditions`` vector in the
        order ``[mask, base_values]``.

        Parameters
        ----------
        batch_size : int
            Number of rows to produce.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``"mask"`` : ``np.ndarray`` of shape ``(batch_size, D)``.
            - ``"base_values"`` : ``np.ndarray`` of shape ``(batch_size, D)``.
            - ``"full_conditions"`` : ``np.ndarray`` of shape
              ``(batch_size, 2*D)`` equal to ``concatenate([mask, base_values], axis=1)``.
        """
        batched_mask = np.tile(self.mask[None, :], (batch_size, 1)).astype(
            self.dtype, copy=False
        )
        batched_base = np.tile(self.base_values[None, :], (batch_size, 1)).astype(
            self.dtype, copy=False
        )
        full = np.concatenate([batched_mask, batched_base], axis=1)
        return {
            "mask": batched_mask,
            "base_values": batched_base,
            "full_conditions": full,
        }

    def _slice_len(self, name: str) -> int:
        """Return the flattened slot length (dimensionality) for ``name``."""
        sl = self._name_slices[name]
        return sl.stop - sl.start


# class Tokenizer:
#     """
#     Manages parameter sampling and simulation input construction
#     for a model variant using a shared global parameter schema.
#
#     Each parameter in the global schema can be masked as:
#     - Free (1): to be estimated
#     - Fixed (0): to be given a value
#     - Inactive (-1): to be omitted
#
#     Parameters
#     ----------
#     parameter_names : list of str
#         Ordered list of all parameter names in the global schema.
#     free_parameters : dict of {str: Callable[[int, Optional[np.ndarray]], np.ndarray]}
#         Mapping of parameter names to sampling functions for free parameters.
#         Samplers accept batch_size and optional context array.
#     fixed_parameters : dict of {str: float}
#         Mapping of parameter names to fixed values.
#     fallback_value : float, optional
#         Value used for parameters that are neither free nor fixed.
#         Defaults to 0.0.
#
#     Raises
#     ------
#     ValueError
#         If any parameter is defined as both free and fixed.
#     """
#
#     def __init__(
#         self,
#         *,
#         parameter_names: list[str],
#         fixed_parameters: dict[str, Union[float, np.ndarray]],
#         free_parameters: dict[str, Callable[[int, Optional[np.ndarray]], np.ndarray]],
#         fallback_value: float = 0.0,
#     ):
#         self.parameter_names = list(parameter_names)
#         self.free_parameters = free_parameters
#         self.fixed_parameters = fixed_parameters
#         self.fallback_value = fallback_value
#
#         # Sanity check: no parameters should be both fixed and free
#         overlap = set(free_parameters) & set(fixed_parameters)
#         if overlap:
#             raise ValueError(
#                 f"The following parameters cannot be both fixed and free: {overlap}"
#             )
#
#         # Initialize base_values for fixed and default parameters
#         self.base_values = []
#         self.array_parameters = set()
#         for name in self.parameter_names:
#             if name in fixed_parameters:
#                 value = fixed_parameters[name]
#                 if isinstance(value, np.ndarray):
#                     self.array_parameters.add(name)
#                     self.base_values.append(np.array([fallback_value]))
#                 else:
#                     self.base_values.append(np.array([value]))
#             else:
#                 self.base_values.append(np.array([fallback_value]))
#         self.base_values = np.concatenate(self.base_values, axis=0).astype(np.float32)
#
#         # Binary infer masks for the parameters (1.0 for free parameters, 0.0 otherwise)
#         self.infer_mask = np.array(
#             [1.0 if name in free_parameters else 0.0 for name in self.parameter_names],
#             dtype=np.float32,
#         )
#
#         self.active_mask = np.array(
#             [
#                 1.0
#                 if (n in self.free_parameters or n in self.fixed_parameters)
#                 else 0.0
#                 for n in self.parameter_names
#             ],
#             dtype=np.float32,
#         )
#
#         self.num_parameters = len(self.parameter_names)
#
#     def sample(
#         self, batch_size: int, context: Optional[np.ndarray] = None
#     ) -> dict[str, np.ndarray]:
#         """
#         Samples a batch of values for the free parameters, optionally conditioned on context.
#
#         Parameters
#         ----------
#         batch_size : int
#             The number of samples to generate for each parameter.
#         context : np.ndarray, optional
#             Context array to condition sampling (e.g., for conditional priors).
#
#         Returns
#         -------
#         dict of {str: np.ndarray}
#             A dictionary mapping free parameter names to sampled values.
#             Each array has shape (batch_size,).
#         """
#         return {
#             name: sampler(batch_size, context)
#             for name, sampler in self.free_parameters.items()
#         }
#
#     def combine(
#         self,
#         sample_parameters: dict[str, np.ndarray],
#         batch_size: int = None
#     ) -> dict[str, float]:
#         """
#         Combines sampled parameters along with the fixed and default parameters
#         into a full dictionary for a single simulation instance.
#
#         Parameters
#         ----------
#         sample_parameters : dict of {str: np.ndarray}
#             A batch of sampled free parameters. Each array should have shape (batch_size,).
#         batch_size : int, optional
#             If provided, returns a single parameter dict for the entire batch.
#
#         Returns
#         -------
#         dict of {str: float}
#             A dictionary containing the full parameter set for a single simulation.
#         """
#
#         if batch_size is None:
#             raise ValueError("batch_size must be provided for batch simulation")
#
#         full_parameters = {}
#
#         for i, name in enumerate(self.parameter_names):
#             if self.active_mask[i] == 0.0:
#                 continue
#             if self.infer_mask[i] == 1.0:
#                 full_parameters[name] = sample_parameters[name]
#             elif name in self.fixed_parameters:
#                 full_parameters[name] = self.fixed_parameters[name]
#             else:
#                 full_parameters[name] = self.fallback_value
#
#         return full_parameters
#
#     def build_inference_conditions(
#         self,
#         batch_size: int,
#         *,
#         include_variant: bool = True,
#         include_context: bool = True,
#         variant_encoder: np.ndarray = None,
#         context_encoder: np.ndarray = None,
#     ):
#         """
#         Builds an inference condition using a shared global parameter schema.
#
#         Parameters
#         ----------
#         batch_size      : int
#             The number of samples to generate for each parameter.
#         include_variant : bool, optional, default: True
#             Whether to include variant one-hot encoder inference condition.
#         include_context : bool, optional, default: True
#             Whether to include context encoder inference condition.
#         variant_encoder : np.ndarray, optional, default: None
#             The one-hot encoded variant encoder.
#         context_encoder : np.ndarray, optional, default: None
#             The context variables.
#
#         Returns
#         -------
#         A dictionary consisting of all components of the inference conditions:
#         - infer_mask: binary infer_mask for free and fixed parameters
#         - active_mask: binary mask for active parameters
#         - base_values: fixed default values for the fixed parameters
#         - variant: one-hot encoded variant encoder
#         - context: context variables
#         - full_embeddings: all of the above concatenated for training purposes.
#         """
#         # Batch infer_mask and conditions
#         batched_infer_mask = np.tile(self.infer_mask, (batch_size, 1)).astype(np.float32)
#         batched_active_mask = np.tile(self.active_mask, (batch_size, 1)).astype(np.float32)
#         batched_base_values = np.tile(self.base_values, (batch_size, 1)).astype(np.float32)
#
#         # Make a list of inference conditions to be concatenated as embeddings
#         full_conditions = [batched_infer_mask, batched_active_mask, batched_base_values]
#         inference_conditions = {
#             "infer_mask": batched_infer_mask,
#             "active_mask": batched_active_mask,
#             "base_values": batched_base_values,
#         }
#
#         if include_variant:
#             if variant_encoder is not None:
#                 inference_conditions["variant"] = variant_encoder.astype(np.float32)
#                 full_conditions.append(variant_encoder)
#
#         if include_context:
#             if context_encoder is not None:
#                 inference_conditions["context"] = context_encoder.astype(np.float32)
#                 full_conditions.append(context_encoder)
#
#         inference_conditions["full_conditions"] = np.concatenate(full_conditions, axis=1)
#         return inference_conditions
#
#     def get_infer_mask(self) -> np.ndarray:
#         """
#         Returns the infer_mask array for the free parameters.
#
#         Returns
#         -------
#         np.ndarray of shape (num_parameters,)
#             Binary infer_mask indicating which parameters are free (and therefore learnable).
#         """
#         return self.infer_mask
#
#     def get_active_mask(self) -> np.ndarray:
#         """
#         Returns the active_mask array for the free and fixed parameters.
#
#         Returns
#         -------
#         np.ndarray of shape (num_parameters,)
#             Binary mask indicating which parameters are active (free or fixed).
#         """
#         return self.active_mask
#
#     def get_base_values(self) -> np.ndarray:
#         """
#         Returns the conditioning vector for the fixed and default parameters.
#
#         Returns
#         -------
#         np.ndarray of shape (num_parameters,)
#             Conditioning vector for the fixed and default values for all parameters.
#             This vector aligns with the global parameter schema.
#         """
#         return self.base_values
