import numpy as np
from .model import Model
from ..utils.simulator_utils import Tokenizer


class ModelVariant:
    """
    Encapsulates a model variant with free and fixed parameters,
    providing simulation and inference interface with parameter masking.

    Parameters
    ----------
    name : str
        Name or identifier for this variant.
    model : type
        A callable class implementing a `.simulate(params: dict, batch_size: int)` method.
    tokenizer : Tokenizer
        Configuration object that manages sampling, value fixing, and masking of parameters.
    """

    def __init__(
        self,
        name: str,
        model: type[Model],
        tokenizer: Tokenizer
    ):
        self.name = name
        self.model: Model = model()
        self.tokenizer = tokenizer
        self.parameter_names = list(tokenizer.parameter_names)


    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Simulate data and return full parameter vectors.

        Returns
        -------
        dict with keys:
        - "sim_data" : np.ndarray of shape (batch_size, ...)
            Simulation outputs.
        - "full_params" : np.ndarray of shape(batch_size, num_parameters)
            Sampled set of parameters.
        """

        # Sample and collect params
        sampled_parameters = self.tokenizer.sample(batch_size)
        params_dicts = [
            self.tokenizer.combine(sampled_parameters, i) for i in range(batch_size)
        ]

        # Run simulator
        sims = []

        for params in params_dicts:
            out = self.model.simulate(params, batch_size=1)

            if isinstance(out, np.ndarray) and out.ndim >= 1 and out.shape[0] == 1:
                sims.append(out.squeeze(axis=0))
            else:
                sims.append(out)

        sim_data = np.stack(sims, axis=0)

        # Build full set of parameters

        base_values = np.tile(self.tokenizer.get_base_values(), (batch_size, 1)).astype(np.float32)

        full_params = base_values.copy()

        for j, name in enumerate(self.parameter_names):
            if self.tokenizer.infer_mask[j] == 1.0:
                full_params[:, j] = sampled_parameters[name].astype(np.float32)

        inference_conditions = self.tokenizer.build_inference_conditions(
            batch_size=batch_size,
            include_variant=False,
            include_context=False
        )

        return {
            "sim_data": sim_data,
            "full_params": full_params,
            "inference_conditions": inference_conditions
        }


    def get_infer_mask(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated binary mask for which parameters are free.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Binary mask indicating which parameters are free.
        """
        mask = self.tokenizer.get_infer_mask()
        return np.tile(mask, (batch_size, 1)).astype(np.float32)


    def get_active_mask(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated binary mask for which parameters are free.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Binary mask indicating which parameters are free.
        """
        mask = self.tokenizer.get_active_mask()
        return np.tile(mask, (batch_size, 1)).astype(np.float32)


    def get_base_values(self, batch_size: int) -> np.ndarray:
        """
        Returns a repeated conditioning vector with fixed/default values.

        Returns
        -------
        np.ndarray of shape (batch_size, num_parameters)
            Conditioning vector with fixed/default values.
        """
        base_values = self.tokenizer.get_base_values()
        return np.tile(base_values, (batch_size, 1)).astype(np.float32)


    def build_inference_conditions(
            self,
            batch_size: int,
            *,
            variant_encoder: np.ndarray = None,
            context_encoder: np.ndarray = None
    ) -> dict[str, np.ndarray]:
        """
        Build inference conditions including optional variant/context encoders.
        """
        return self.tokenizer.build_inference_conditions(
            batch_size=batch_size,
            variant_encoder=variant_encoder,
            context_encoder=context_encoder,
            include_variant=variant_encoder is not None,
            include_context=context_encoder is not None
        )
