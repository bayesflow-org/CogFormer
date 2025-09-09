import numpy as np
from typing import Union, Optional, Callable, Iterable
from ..model import Model
from simulators.benchmarks import simulate_standard_ddm
from simulators import ContextManager


class StandardDDM(Model):
    """
    Simulates response times and choices from a classic Drift Diffusion Model (DDM)
    with static boundaries using Euler-Maruyama discretization.

    Accumulation continues until evidence reaches upper or lower boundary.
    Returns both RT and binary choice (1 = upper, 0 = lower).
    """

    def __init__(self, context_manager: ContextManager, dt: float = 0.001, max_steps: int = 10000):
        """Initialize StandardDDM with simulation parameters.

        Parameters
        ----------
        dt : float, optional
            Time step for simulation in seconds, by default 0.001.
        max_steps : int, optional
            Maximum number of simulation steps, by default 10000.
        """
        self.context_manager = context_manager
        self.dt = dt
        self.max_steps = max_steps

    def simulate(
        self,
        params: dict[str, Union[float, np.ndarray]],
        num_samples: int = 1,
        context: Optional[np.ndarray] = None,
        modulation: Optional[Callable[[dict, np.ndarray], dict]] = None
    ) -> dict[str, np.ndarray]:
        """
        Simulate data for a single model run with specified parameters.

        Parameters
        ----------
        params : dict[str, Union[float, np.ndarray]]
            Parameters for simulation, with arrays of shape (dims,) or scalars.
            Must include:
            - v : float or np.ndarray
                Drift rate, determines evidence accumulation speed.
            - a : float or np.ndarray
                Decision boundary, controls decision caution, > 0.
            - z or z_arr : float or np.ndarray
                Starting point, relative to boundaries (0 < z < 1).
            - tau or tau_arr : float or np.ndarray
                Non-decision time, time before evidence accumulation, >= 0.
            - s_v : float or np.ndarray
                Drift rate noise, standard deviation of drift rate variability, >= 0.
            - sigma : float or np.ndarray
                Diffusion noise, standard deviation of evidence accumulation noise, > 0.
            - s_z : float or np.ndarray
                Starting point noise, standard deviation of starting point variability, >= 0.
            - s_tau : float or np.ndarray
                Non-decision time noise, standard deviation of non-decision time variability, >= 0.
        num_samples : int, optional
            Number of samples (trials) per simulation, by default 1.
        context : np.ndarray, optional
            Array of external conditions for each trial, shape (num_samples,).
        modulation : Callable[[dict, np.ndarray], dict], optional
            Function to adjust model parameters based on context.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing 'rts', 'choices', and optionally 'context'.
            Arrays are of shape (num_samples,) in np.float32.

        Raises
        ------
        ValueError
            If context shape is invalid, required parameters are missing, or parameters violate constraints.
        """

        # Modulate parameters with context if provided
        if context is not None and modulation is not None:
            if context.shape != (num_samples,):
                raise ValueError(f"context must have shape ({num_samples},), got {context.shape}")
            params = modulation(params, context)
            _validate_parameters(params, num_samples)

        # Simulate
        result = simulate_standard_ddm(
            v=params["v"],
            a=params["a"],
            z=params["z"],
            tau=params["tau"],
            s_v=params["s_v"],
            sigma=params["sigma"],
            s_z=params["s_z"],
            s_tau=params["s_tau"],
            dt=self.dt,
            max_steps=self.max_steps,
            num_samples=num_samples
        )

        # Construct output
        output = {"rts": result[:, 0], "choices": result[:, 1]}
        output["context"] = context
        return output


    @staticmethod
    def summarize(
            outputs: dict[str, np.ndarray],
            quantile_levels: Iterable[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
            by_choice: bool = True,
            tau: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Summarize RTs with robust quantiles (and optional decision-time RT−tau).

        Parameters
        ----------
        outputs : dict[str, np.ndarray]
            Dict from `simulate`, must contain 'rts' and 'choices' (float arrays).
        quantile_levels : Iterable[float], optional
            Quantile levels in [0,1], by default (0.1, 0.3, 0.5, 0.7, 0.9).
        by_choice : bool, optional
            If True, compute quantiles separately for choice 0 and 1.
        tau : np.ndarray, optional
            Per-trial non-decision times (same shape as outputs['rts']) to also
            compute decision-time quantiles over max(rts - tau, 0).

        Returns
        -------
        dict[str, np.ndarray]
            {
              'invalid_rate': float32,
              'rt_quantiles': (L,),
              'rt_quantiles_by_choice': (2, L) or all-NaN if by_choice=False,
              'dt_quantiles': (L,) only if tau provided,
              'dt_quantiles_by_choice': (2, L) only if tau provided and by_choice=True
            }
        """
        rts = outputs["rts"].astype(np.float32, copy=False)
        choices = outputs["choices"].astype(np.float32, copy=False)

        q = np.asarray(tuple(quantile_levels), dtype=np.float32)
        num_levels = q.size

        # Valid RT mask (exclude non-terminations)
        valid = ~np.isnan(rts)
        n = rts.size
        n_valid = int(valid.sum())
        invalid_rate = np.float32(1.0 - (n_valid / n if n > 0 else np.nan))

        def _q(xmask: np.ndarray, x: np.ndarray) -> np.ndarray:
            if not np.any(xmask):
                return np.full((num_levels,), np.nan, dtype=np.float32)
            return np.quantile(x[xmask], q, method="linear").astype(np.float32)

        # RT quantiles
        rt_quantiles = _q(valid, rts)

        # RT by choice
        if by_choice:
            rt_quantiles_by_choice = np.vstack([
                _q(valid & (choices == 0), rts),
                _q(valid & (choices == 1), rts),
            ]).astype(np.float32)
        else:
            rt_quantiles_by_choice = np.full((2, num_levels), np.nan, dtype=np.float32)

        out = {
            "invalid_rate": invalid_rate,
            "rt_quantiles": rt_quantiles,
            "rt_quantiles_by_choice": rt_quantiles_by_choice,
        }

        # Optional decision time (RT - tau), clipped at 0
        if tau is not None:
            tau = tau.astype(np.float32, copy=False)
            dt = np.clip(rts - tau, 0.0, None)
            valid_dt = valid & ~np.isnan(tau)
            dt_quantiles = _q(valid_dt, dt)
            out["dt_quantiles"] = dt_quantiles
            if by_choice:
                out["dt_quantiles_by_choice"] = np.vstack([
                    _q(valid_dt & (choices == 0), dt),
                    _q(valid_dt & (choices == 1), dt),
                ]).astype(np.float32)

        return out


def _process_parameters(
    params: dict[str, Union[float, np.ndarray]],
    num_samples: int
) -> dict[str, np.ndarray]:
    """Process parameters to arrays of consistent shape.

    Parameters
    ----------
    params : dict[str, Union[float, np.ndarray]]
        Same as in the simulate method.
    num_samples : int
        Number of trials.

    Returns
    -------
    dict[str, np.ndarray]
        Processed parameters with shapes (num_samples,).
    """
    # Check for required parameters
    required_params = ["v", "a", "s_v", "sigma", "s_z", "s_tau"]
    if not any(k in params for k in ["z", "z_arr"]):
        raise ValueError("One of 'z' or 'z_arr' must be provided")
    if not any(k in params for k in ["tau", "tau_arr"]):
        raise ValueError("One of 'tau' or 'tau_arr' must be provided")
    if not all(k in params for k in required_params):
        raise ValueError(f"Missing parameters: {set(required_params) - set(params)}")

    # Handle parameter aliases
    if "z_arr" in params:
        params["z"] = np.full(num_samples, params["z_arr"]).astype(np.float32) if np.isscalar(params["z_arr"]) \
            else params["z_arr"].astype(np.float32)
        del params["z_arr"]
    if "tau_arr" in params:
        params["tau"] = np.full(num_samples, params["tau_arr"]).astype(np.float32) if np.isscalar(params["tau_arr"]) \
            else params["tau_arr"].astype(np.float32)
        del params["tau_arr"]

    # Broadcast scalar parameters to arrays
    for key in ["v", "a", "z", "tau", "s_v", "sigma", "s_z", "s_tau"]:
        if key in params:
            params[key] = np.full(num_samples, params[key]).astype(np.float32) if np.isscalar(params[key]) \
                else params[key].astype(np.float32)

    return params


def _validate_parameters(
    params_array: dict[str, np.ndarray],
    num_samples: int
) -> None:
    """Validate parameter shapes and values.

    Parameters
    ----------
    params_array : dict[str, np.ndarray]
        Processed parameters to validate.
    num_samples : int
        Number of trials.

    Raises
    ------
    ValueError
        If parameters have invalid shapes or values.
    """
    for key, param in params_array.items():
        if param.shape != num_samples:
            if param.shape[0] == num_samples:
                params_array[key] = param.squeeze()
            else:
                raise ValueError(f"{key} must have shape ({num_samples},), but instead have {param.shape}")
    if np.any(params_array["a"] <= 0) or np.any(params_array["sigma"] <= 0):
        raise ValueError("a, sigma must be > 0")
    if np.any(params_array["s_v"] < 0) or np.any(params_array["s_z"] < 0) or np.any(params_array["s_tau"] < 0):
        raise ValueError("s_v, s_z, s_tau must be >= 0")
    if np.any((params_array["z"] <= 0) | (params_array["z"] >= 1)):
        raise ValueError("0 < z < 1")
