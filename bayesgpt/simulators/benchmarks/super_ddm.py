import numpy as np
from typing import Union, Optional, Callable, Iterable
from ..model import Model
from simulators import ContextManager
from simulators.benchmarks import simulate_mixture_ddm, simulate_schedule_ddm


class SuperDDM(Model):
    """Simulate a flexible drift diffusion model with single, mixture, or scheduled drifts.

    Supports context modulation for any model parameters (e.g., drift rate, decision boundary,
    starting point) and integrates with NestedModelFamily for simulation-based inference workflows.

    Parameters
    ----------
    dt : float, optional
        Time step for simulation (seconds), by default 0.001.
    max_steps : int, optional
        Maximum number of simulation steps, by default 10000.
    """

    def __init__(self, context_manager: ContextManager, dt: float = 0.001, max_steps: int = 10000):
        """Initialize SuperDDM with simulation parameters.

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

        # Run appropriate simulation based on drift type
        if "v_schedule" in params:
            result = simulate_schedule_ddm(
                v_schedule=params["v_schedule"],
                t_schedule=params.get("t_schedule", np.zeros_like(params["v_schedule"])),
                a=params["a"],
                z=params["z"],
                tau=params["tau"],
                s_v=params["s_v"],
                angle=params["angle"],
                s_z=params["s_z"],
                s_tau=params["s_tau"],
                dt=self.dt,
                max_steps=self.max_steps,
                num_samples=num_samples,
            )
        else:
            v = params.get("v_components", params.get("v"))
            p = params.get("p_components", None)
            result = simulate_mixture_ddm(
                v=v,
                p=p,
                a=params["a"],
                z=params["z"],
                tau=params["tau"],
                s_v=params["s_v"],
                angle=params["angle"],
                s_z=params["s_z"],
                s_tau=params["s_tau"],
                dt=self.dt,
                max_steps=self.max_steps,
                num_samples=num_samples,
            )

        # Construct output with optional context
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
        Processed parameters with shapes (num_samples,) or
        (num_samples, num_segments) for multidimensional parameters.

    Raises
    ------
    ValueError
        If required parameters are missing or have invalid values.
    """
    # Check for required parameters
    required_params = ["a", "s_v", "angle", "s_z", "s_tau"]
    if not any(k in params for k in ["v", "v_components", "v_schedule"]):
        raise ValueError("One of 'v', 'v_components', or 'v_schedule' must be provided")
    elif not any(k in params for k in ["z", "z_arr"]):
        raise ValueError("One of 'z' or 'z_arr' must be provided")
    elif not any(k in params for k in ["tau", "tau_arr"]):
        raise ValueError("One of 'tau' or 'tau_arr' must be provided")
    elif not all(k in params for k in required_params):
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
    for key in ["v", "a", "z", "tau", "s_v", "angle", "s_z", "s_tau"]:
        if key in params:
            params[key] = np.full(num_samples, params[key]).astype(np.float32) if np.isscalar(params[key]) \
                else params[key].astype(np.float32)

    # Handle mixture or scheduled drifts
    if "v_components" in params:
        params["v_components"] = (
            np.full((num_samples, params["v_components"].shape[-1]), params["v_components"]).astype(np.float32)
            if params["v_components"].ndim == 1
            else params["v_components"].astype(np.float32)
        )
        if "p_components" in params:
            params["p_components"] = (
                np.full((num_samples, params["v_components"].shape[-1]), params["p_components"]).astype(np.float32)
                if params["p_components"].ndim == 1
                else params["p_components"].astype(np.float32)
            )
    if "v_schedule" in params:
        params["v_schedule"] = (
            np.full((num_samples, params["v_schedule"].shape[-1]), params["v_schedule"]).astype(np.float32)
            if params["v_schedule"].ndim == 1
            else params["v_schedule"].astype(np.float32)
        )
        params["t_schedule"] = (
            np.full((num_samples, params["v_schedule"].shape[-1]), params.get("t_schedule", 0.0)).astype(np.float32)
            if "t_schedule" not in params or params["t_schedule"].ndim == 1
            else params["t_schedule"].astype(np.float32)
        )

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
    # Validate parameter shapes
    for key, param in params_array.items():
        if key in ["v_components", "v_schedule", "t_schedule", "p_components"]:
            if param.shape[0] != num_samples:
                raise ValueError(f"{key} must have shape (num_samples, num_segments)")
        else:
            if param.shape != num_samples:
                if param.shape[0] == num_samples:
                    params_array[key] = param.squeeze()
                else:
                    raise ValueError(f"{key} must have shape ({num_samples},), but instead have {param.shape}")
    if "t_schedule" in params_array and params_array["t_schedule"].shape != params_array["v_schedule"].shape:
        raise ValueError("t_schedule must have same shape as v_schedule")
    if "p_components" in params_array and params_array["p_components"].shape != params_array["v_components"].shape:
        raise ValueError("p_components must have same shape as v_components")

    # Validate parameter values
    if np.any(params_array["a"] <= 0) or np.any(params_array["sigma"] <= 0):
        raise ValueError("a, sigma must be > 0")
    if (np.any(params_array["s_v"] < 0) or np.any(params_array["angle"] < 0)
            or np.any(params_array["s_z"] < 0) or np.any(params_array["s_tau"] < 0)):
        raise ValueError("s_v, angle, s_z, s_tau must be >= 0")
    if "z" in params_array and np.any((params_array["z"] <= 0) | (params_array["z"] >= 1)):
        raise ValueError("0 < z < 1")
    if "p_components" in params_array and np.any(params_array["p_components"] < 0):
        raise ValueError("p_components must be >= 0")
