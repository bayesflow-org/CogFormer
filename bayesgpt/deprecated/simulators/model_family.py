import numpy as np
from joblib import Parallel, delayed
from typing import Tuple, Type, List, Optional, Callable, Dict, Union
from collections.abc import Mapping

from simulators.model import Model
from simulators.model_family import ModelVariant
from simulators.context_manager import Tokenizer

class NestedModelFamily:
    """Manages batch simulations across multiple model variants with Joblib."""

    def __init__(self, variants: List[ModelVariant], n_jobs: int = -1):
        """Initialize with list of model variants and number of parallel jobs."""
        self.variants = variants
        self.variant_names = [v.name for v in variants]
        self.n_jobs = n_jobs

    def _sample_one_variant(self, variant: ModelVariant, n_samples: int, context: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample parameters for a single variant."""
        sampled = variant.tokenizer.sample(num_samples=n_samples, context=context)
        return variant.tokenizer.combine(sampled)

    def batch_sample(
        self,
        num_samples_per_variant: Union[int, List[int]],
        context: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """Batch-sample parameters for all variants."""
        if isinstance(num_samples_per_variant, int):
            num_samples = [num_samples_per_variant] * len(self.variants)
        else:
            num_samples = num_samples_per_variant

        params = Parallel(n_jobs=self.n_jobs)(
            delayed(self._sample_one_variant)(v, n, context if not isinstance(context, list) else context[i])
            for i, (v, n) in enumerate(zip(self.variants, num_samples))
        )
        return {"parameters": params}

    def _simulate_one_variant(
        self,
        variant: ModelVariant,
        num_samples: int,
        context: Optional[np.ndarray],
        modulation: Optional[Union[Callable, Dict[str, Callable]]]
    ) -> Dict[str, Union[Dict, np.ndarray, str]]:
        """Simulate a single variant with optional modulation."""
        mod_fn = modulation if not isinstance(modulation, dict) else modulation.get(variant.name, lambda x, y: x)
        result = variant.sample(num_samples=num_samples, context=context)
        params = mod_fn(result["full_params"], context) if context is not None else result["full_params"]
        return {
            "sim_data": result["sim_data"],
            "full_params": params,
            "inference_conditions": result["inference_conditions"],
            "variant_name": result["variant_name"],
            "sampled_parameters": result["sampled_parameters"]
        }

    def batch_simulate(
        self,
        num_samples_per_variant: Union[int, List[int]],
        context: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        modulation: Optional[Union[Callable, Dict[str, Callable]]] = None
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """Run batched simulations across variants with Joblib."""
        if isinstance(num_samples_per_variant, int):
            num_samples = [num_samples_per_variant] * len(self.variants)
        else:
            num_samples = num_samples_per_variant

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._simulate_one_variant)(v, n, context if not isinstance(context, list) else context[i], modulation)
            for i, (v, n) in enumerate(zip(self.variants, num_samples))
        )

        # Vectorize outputs
        sim_data_list = [r["sim_data"] for r in results]
        # Assume sim_data is a dictionary with at least two primary arrays (e.g., rts, choices)
        sim_data_arrays = []
        for sim_data in sim_data_list:
            # Get the first two keys (assuming at least two arrays, e.g., rts and choices)
            keys = list(sim_data.keys())[:2]
            if len(keys) < 2:
                raise ValueError(f"sim_data must contain at least two arrays; got {keys}")
            # Stack the two arrays along the last axis
            sim_data_arrays.append(np.stack([sim_data[keys[0]], sim_data[keys[1]]], axis=-1))
        sim_data = np.stack(sim_data_arrays)  # Shape: (num_variants, num_samples, 2)
        full_params = np.stack([r["full_params"] for r in results])  # Shape: (num_variants, total_slots)
        inference_conditions = np.stack([r["inference_conditions"] for r in results])  # Shape: (num_variants, condition_dim)
        sampled_parameters = np.array([r["sampled_parameters"] for r in results], dtype=object)  # Shape: (num_variants,)
        variant_indices = np.arange(len(self.variants), dtype=np.int32)[:, None]  # Shape: (num_variants, 1)
        output = {
            "sim_data": sim_data,
            "full_params": full_params,
            "inference_conditions": inference_conditions,
            "sampled_parameters": sampled_parameters,
            "variant_names": self.variant_names,
            "variant_indices": variant_indices,
        }
        if context is not None:
            output["context"] = np.stack(context) if isinstance(context, list) else context
        return output

    def _default_summary(self, sim_data: Dict[str, np.ndarray], data_key: str = "rts") -> Dict[str, np.ndarray]:
        """Default summary statistics for a single variant output."""
        data = sim_data.get(data_key, next(iter(sim_data.values())))  # Use first array if key not found
        valid = ~np.isnan(data)
        quantiles = np.quantile(data[valid], [0.1, 0.3, 0.5, 0.7, 0.9], method="linear") if valid.any() else np.full(5, np.nan)
        return {
            "quantiles": quantiles,
            "mean": np.mean(data[valid]) if valid.any() else np.nan,
            "invalid_rate": 1.0 - valid.mean(),
            "variance": np.var(data[valid]) if valid.any() else np.nan
        }

    def summarize(
        self,
        outputs: Dict[str, np.ndarray],
        summary_fn: Optional[Callable] = None,
        data_key: str = "rts"
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """Compute summary statistics across variant outputs."""
        summary_fn = summary_fn or (lambda x: self._default_summary(x, data_key))
        stats = Parallel(n_jobs=self.n_jobs)(
            delayed(summary_fn)(variant_out) for variant_out in outputs["sim_data"]
        )
        return {"summary_stats": stats}


class LegacyModelFamily:
    """
    [This version of NestedModelFamily is considered now as legacy.]
    A collection of related model variants sharing a common interface.

    Useful for flexible simulation, inference, and benchmarking workflows
    where each variant represents a different configuration of the same model class.

    Parameters
    ----------
    parameter_names : list of str
        Global schema of all parameters across variants.
    """

    def __init__(self, parameter_names: list[str]):
        self.parameter_names = list(parameter_names)
        self.variants: dict[str, ModelVariant] = {}

    # noinspection PyTypeChecker
    def add_variant(
        self,
        name: str,
        model: type[Model],
        fixed_parameters: Mapping[str, Union[float, np.ndarray, Callable[[int], np.ndarray]]],
        free_parameters: Mapping[str, Callable[[int, Optional[np.ndarray]], np.ndarray]],
        num_samples: int = 1,
    ):
        """
        Adds a new variant to the model family.

        Parameters
        ----------
        name : str
            Name of the variant.
        model : type
            Model class implementing simulate(params: dict, batch_size: int, num_samples: int) -> np.ndarray or Mapping
        free_parameters : dict
            Sampling functions for free parameters, accepting batch_size and context.
        fixed_parameters : dict
            Fixed values for some parameters.
        num_samples : int
            Number of samples to generate.
        """
        new_params = set(free_parameters.keys()) | set(fixed_parameters.keys()) - set(self.parameter_names)
        self.parameter_names.extend(new_params)

        tokenizer = Tokenizer(
            parameter_names=self.parameter_names,
            variant_parameters=set(free_parameters.keys()) | set(fixed_parameters.keys()),
            fixed_parameters=fixed_parameters,
            free_parameters=free_parameters
        )

        self.variants[name] = ModelVariant(name=name, model=model, tokenizer=tokenizer, num_samples=num_samples)

    def remove_variant(self, name: str):
        """
        Removes a variant from the model family.

        Parameters
        ----------
        name : str
            Name of the variant to remove.

        Raises
        ------
        KeyError
            If the variant name does not exist.
        """
        if name not in self.variants:
            raise KeyError(f"Variant '{name}' not found in the model family.")
        del self.variants[name]

    def add_all_variants(
        self,
        variants: List[
            Tuple[
                str,
                Type[Model],
                dict[str, Callable[[int, Optional[np.ndarray]], np.ndarray]],
                dict[str, float],
                float,
            ]
        ],
    ):
        """
        Adds multiple variants to the model family at once.

        Parameters
        ----------
        variants : list of tuples
            Each tuple contains (name, model, free_parameters, fixed_parameters, fallback_value).
        """
        for name, model, free_parameters, fixed_parameters, fallback_value in variants:
            self.add_variant(
                name, model, free_parameters, fixed_parameters, num_samples=1
            )

    def remove_all_variants(self):
        """
        Removes all variants from the model family.
        """
        self.variants.clear()

    def sample(
        self,
        variant_name: str,
        batch_size: int,
        context: Optional[np.ndarray] = None,
        *,
        flatten: bool = True
    ) -> dict[str, Union[np.ndarray, Mapping[str, np.ndarray]]]:
        """
        Samples a batch of simulations from a specified variant.

        Parameters
        ----------
        variant_name : str
            Name of the variant to use.
        batch_size : int
            Number of simulations to run.
        context : np.ndarray, optional
            Context array to condition parameter sampling.
        flatten: bool, optional
            Whether to flatten samples.

        Returns
        -------
        dict with keys:
        - sim_data : np.ndarray or Mapping[str, np.ndarray]
            Simulated data from the model.
        - full_params : np.ndarray
            Values of sampled and fixed parameters.
        - inference_conditions : np.ndarray
            Concatenated inference conditions (mask, base values, etc.).
        """
        if variant_name not in self.variants:
            raise KeyError(f"Variant '{variant_name}' not found in the model family.")

        variant = self.variants[variant_name]
        samples = variant.sample(batch_size=batch_size, context=context, flatten=flatten)
        output = samples

        sim_data = output["sim_data"]
        if isinstance(sim_data, np.ndarray):
            nan_count = np.isnan(sim_data).sum()
        else:
            nan_count = sum(np.isnan(sim_data[key]).sum() for key in sim_data)
        if nan_count > 0:
            print(f"Warning: {variant_name} has {nan_count} non-terminating trials")

        one_hot_variant = self.get_variant_encoder(variant_name=variant_name, batch_size=batch_size)
        inference_conditions = variant.build_inference_conditions(
            batch_size=batch_size,
            one_hot_variant=one_hot_variant,
            context=context,
        )

        output["inference_conditions"] = inference_conditions["full_conditions"]
        return output

    def get_mask(self, variant_name: str, batch_size: int) -> np.ndarray:
        """
        Returns the tri-state parameter mask for a given variant.

        Parameters
        ----------
        variant_name : str
            Name of the variant.
        batch_size : int
            Number of rows in the returned batch.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Tri-state mask where -1.0 indicates inactive, 0.0 fixed, and 1.0 free parameters.
        """
        if variant_name not in self.variants:
            raise KeyError(f"Variant '{variant_name}' not found in the model family.")

        return self.variants[variant_name].get_mask(batch_size)

    def get_variant_encoder(self, variant_name: str, batch_size: int) -> np.ndarray:
        """
        Returns a one-hot encoded model identity vector.

        Parameters
        ----------
        variant_name : str
            Name of the variant.
        batch_size : int
            Number of rows in the returned batch.

        Returns
        -------
        np.ndarray of shape (batch_size, num_variants)
            One-hot vector encoding model identity.
        """
        if variant_name not in self.variants:
            raise KeyError(f"Variant '{variant_name}' not found in the model family.")

        variant_names = self.variant_names
        idx = variant_names.index(variant_name)
        encoder = np.zeros((batch_size, len(variant_names)), dtype=np.float32)
        encoder[:, idx] = 1.0
        return encoder

    @property
    def variant_names(self) -> list[str]:
        """
        Returns
        -------
        list[str]
            List of available variant names.
        """
        return list(self.variants.keys())
