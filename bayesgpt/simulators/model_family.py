import numpy as np
from collections.abc import Callable, Mapping
from typing import Union, Optional, List, Tuple, Type

from .model import Model
from .model_variant import ModelVariant
from ..utils.tokenizer import Tokenizer


class NestedModelFamily:
    """
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

    def add_variant(
        self,
        name: str,
        model: type[Model],
        fixed_parameters: dict[str, Union[float, np.ndarray, Callable[[int], np.ndarray]]],
        free_parameters: dict[str, Callable[[int, Optional[np.ndarray]], np.ndarray]],
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
            free_parameters=free_parameters,
            fixed_parameters=fixed_parameters
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
        self, variant_name: str, batch_size: int, context: Optional[np.ndarray] = None
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
        samples = variant.sample(batch_size=batch_size, context=context)
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