import numpy as np
from .variant_simulator import VariantSimulator


class ModelFamilySimulator:
    """
    A collection of related model variants sharing a common interface.

    Useful for flexible simulation, inference, and benchmarking workflows
    where each variant represents a different configuration of the same model class.

    Parameters
    ----------
    variants : list of ModelVariant
        List of model variants to include in the family.
    """

    def __init__(self, variants: list[VariantSimulator]):
        self.variants = variants
        self.name_to_index = {v.name: i for i, v in enumerate(variants)}


    def sample(self, variant_idx: int, batch_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Samples a batch of simulations from a specified variant.

        Parameters
        ----------
        variant_idx : int
            Index of the variant to use.

        batch_size : int
            Number of samples to generate.

        Returns
        -------
        x : np.ndarray
            Simulated outputs of shape (batch_size, ...)

        theta : dict of {str: np.ndarray}
            Sampled free parameters of shape (batch_size,)
        """
        return self.variants[variant_idx].sample(batch_size)


    def list_variants(self) -> list[str]:
        """
        Returns a list of all variant names in the family.
        """
        return [variant.name for variant in self.variants]


    def get_variant(self, name: str) -> VariantSimulator:
        """
        Retrieves a variant by name.

        Parameters
        ----------
        name : str
            Name of the variant.

        Returns
        -------
        ModelVariant
            The corresponding model variant.
        """
        return self.variants[self.name_to_index[name]]
