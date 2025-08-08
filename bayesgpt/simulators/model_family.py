import numpy as np
from collections.abc import Callable

from .model import Model
from .model_variant import ModelVariant
from ..utils.simulator_utils import ParameterManager


class ModelFamily:
    """
    A collection of related model variants sharing a common interface.

    Useful for flexible simulation, inference, and benchmarking workflows
    where each variant represents a different configuration of the same model class.

    Parameters
    ----------
    parameter_names : list of str
        Global schema of all parameters across variants.
    """

    def __init__(
        self,
        parameter_names: list[str]
    ):
        self.parameter_names = list(parameter_names)
        self.variants: dict[str, ModelVariant] = {}


    def add_variant(
        self,
        name: str,
        model: type[Model],
        free_parameters: dict[str, Callable[[int], np.ndarray]],
        fixed_parameters: dict[str, float],
        default_value: float = 0.0
    ):
        """
        Adds a new variant to the model family.

        Parameters
        ----------
        name : str
            Name of the variant.
        model : type
            Model class implementing simulate(params: dict, batch_size: int) -> np.ndarray
        free_parameters : dict
            Sampling functions for free parameters.
        fixed_parameters : dict
            Fixed values for some parameters.
        default_value : float
            Default value for unspecified parameters.
        """
        parameter_manager = ParameterManager(
            parameter_names=self.parameter_names,
            free_parameters=free_parameters,
            fixed_parameters=fixed_parameters,
            default_value=default_value,
        )

        self.variants[name] = ModelVariant(
            name=name,
            model=model,
            parameter_manager=parameter_manager
        )


    def sample(
        self,
        variant_name: str,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        """
        Samples a batch of simulations from a specified variant.

        Parameters
        ----------
        variant_name : str
            Name of the variant to use.

        batch_size : int
            Number of simulations to run.

        Returns
        -------
        sim_outputs : np.ndarray
            Simulated data from the model.

        sampled_parameters : dict of {str: np.ndarray}
            Values of sampled free parameters.

        mask : np.ndarray
            Binary mask over parameters (1.0 = free, 0.0 = fixed/defaulted).

        conditions : np.ndarray
            Fixed/defaulted parameter values in schema order.
        """

        return self.variants[variant_name].sample(batch_size)


    def get_mask(self, variant_name: str, batch_size: int) -> np.ndarray:
        """
        Returns the binary parameter mask for a given variant.

        Parameters
        ----------
        variant_name : str
            Name of the variant.
        batch_size : int
            Number of rows in the returned batch.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Mask where 1.0 indicates a free parameter.
        """
        return self.variants[variant_name].get_mask(batch_size)


    def get_condition_vector(self, variant_name: str, batch_size: int) -> np.ndarray:
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
        variant_names = self.variant_names
        idx = variant_names.index(variant_name)
        cond = np.zeros((batch_size, len(variant_names)), dtype=np.float32)
        cond[:, idx] = 1.0
        return cond


    @property
    def variant_names(self) -> list[str]:
        """
        Returns
        -------
        list[str]
            List of available variant names.
        """
        return list(self.variants.keys())
