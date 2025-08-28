import numpy as np
from typing import Union
from numba import njit, prange
from ..model import Model


class SuperDDM(Model):
    """
    Simulate a flexible drift diffusion model with single, mixture, or scheduled drifts.
    """

    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        """
        Initialize SuperDDM with simulation parameters.

        Parameters
        ----------
        dt : float, optional
            Time step for simulation, by default 0.001.
        max_steps : int, optional
            Maximum number of simulation steps, by default 10000.
        """
        self.dt = dt
        self.max_steps = max_steps

    def simulate(
        self,
        params: dict[str, Union[float, np.ndarray]],
        num_trials: int = 1
    ) -> dict[str, np.ndarray]:
        """
        Simulate response times and choices for a single experiment of multiple trials.

        Parameters
        ----------
        params : dict[str, float]
            Dictionary potentially including the following model parameters:
            v : float
                Mean drift rate (required_params if no multiple drifts specified).
            a : float
                Boundary separation (>0).
            z : float
                Mean starting point fraction (0 < z < 1).
            tau : float
                Mean non-decision time (>=0).
            sigma : float
                Diffusion noise (>0).
            angle : float, optional
                Collapse rate (default=0 for fixed bounds).
            s_v : float, optional
                Drift variability (default=0). Applies as additive noise on top of
                the base drift, mixture components, or schedule segments.
            s_z : float, optional
                Starting point variability (default=0).
            s_tau : float, optional
                Non-decision time variability (default=0).
            v_components : array_like, optional
                Drift rates for across-trial mixture (select one per trial).
            p_components : array_like, optional
                Probabilities for mixture components (must sum to 1; default uniform).
            v_schedule : array_like, optional
                Drift rates for within-trial piecewise-constant segments.
            t_schedule : array_like, optional
                Durations (in seconds) for each segment (>0).
        num_trials : int
            Number of trials to simulate.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys 'rts' and 'choices', each mapping to an array
            of shape (num_trials,) containing response times and choices.

        Raises
        ------
        ValueError
            If required_params parameters are missing or have invalid values.
        """
        params = self._process_parameters(params, num_trials)
        self._validate_parameters(params, num_trials)

        if "v_schedule" in params:
            result = simulate_schedule_ddm(
                v_schedule=params["v_schedule"],
                t_schedule=params.get("t_schedule", np.zeros_like(params["v_schedule"])),
                a=params["a"],
                z=params["z"],
                tau=params["tau"],
                s_v=params["s_v"],
                sigma=params["sigma"],
                angle=params["angle"],
                s_z=params["s_z"],
                s_tau=params["s_tau"],
                dt=self.dt,
                max_steps=self.max_steps,
                num_trials=num_trials,
            )
        else:
            v = params.get("v_components", params.get("v"))
            result = simulate_super_ddm(
                v=v,
                a=params["a"],
                z=params["z"],
                tau=params["tau"],
                s_v=params["s_v"],
                sigma=params["sigma"],
                angle=params["angle"],
                s_z=params["s_z"],
                s_tau=params["s_tau"],
                dt=self.dt,
                max_steps=self.max_steps,
                num_trials=num_trials,
            )
        return {"rts": result[:, 0], "choices": result[:, 1]}

    def _process_parameters(
        self,
        params: dict[str, Union[float, np.ndarray]],
        num_trials: int
    ) -> dict[str, np.ndarray]:
        """
        Process parameters in-place to arrays of consistent shape.

        Parameters
        ----------
        params : dict
            Parameters to process (modified in-place).
        num_trials : int
            Number of trials to simulate.

        Returns
        -------
        dict
            Modified params with arrays of shape (num_trials,) or (num_trials, num_segments).

        Raises
        ------
        ValueError
            If required parameters are missing.
        """

        required_params = ["a", "s_v", "sigma", "angle", "s_z", "s_tau"]
        if not any(k in params for k in ["v", "v_components", "v_schedule"]):
            raise ValueError("One of 'v', 'v_components', or 'v_schedule' must be prov_ided")
        if not any(k in params for k in ["z", "z_arr"]):
            raise ValueError("One of 'z' or 'z_arr' must be prov_ided")
        if not any(k in params for k in ["tau", "tau_arr"]):
            raise ValueError("One of 'tau' or 'tau_arr' must be prov_ided")
        if not all(k in params for k in required_params):
            raise ValueError(f"Missing parameters: {set(required_params) - set(params)}")

        # Handle aliases: use z, tau as canonical keys
        if "z_arr" in params:
            params["z"] = np.full(num_trials, params["z_arr"]) if np.isscalar(params["z_arr"]) else params["z_arr"]
            del params["z_arr"]
        if "tau_arr" in params:
            params["tau"] = np.full(num_trials, params["tau_arr"]) if np.isscalar(params["tau_arr"]) else params["tau_arr"]
            del params["tau_arr"]

        # Broadcast standard parameters to (num_trials,)
        for key in ["v", "a", "z", "tau", "s_v", "sigma", "angle", "s_z", "s_tau"]:
            if key in params:
                params[key] = np.full(num_trials, params[key]) if np.isscalar(params[key]) else params[key]

        # Handle v_components or v_schedule
        if "v_components" in params:
            params["v_components"] = (
                np.full((num_trials, params["v_components"].shape[-1]), params["v_components"])
                if params["v_components"].ndim == 1
                else params["v_components"]
            )
        if "v_schedule" in params:
            params["v_schedule"] = (
                np.full((num_trials, params["v_schedule"].shape[-1]), params["v_schedule"])
                if params["v_schedule"].ndim == 1
                else params["v_schedule"]
            )
            params["t_schedule"] = (
                np.full((num_trials, params["v_schedule"].shape[-1]), params.get("t_schedule", 0.0))
                if "t_schedule" not in params or params["t_schedule"].ndim == 1
                else params["t_schedule"]
            )

        return params


    def _validate_parameters(
        self,
        params_array: dict[str, np.ndarray],
        num_trials: int
    ) -> None:
        """
        Validate parameter shapes and values.

        Parameters
        ----------
        params_array : dict
            The arrays of processed parameters to validate.
        num_trials : int
            Number of trials to simulate.

        Raises
        ------
        ValueError
            If parameters have invalid shapes or values.
        """
        # Check shapes
        for key, param in params_array.items():
            if key in ["v_components", "v_schedule", "t_schedule"]:
                if param.shape[0] != num_trials:
                    raise ValueError(f"{key} must have shape (num_trials, num_segments)")
            else:
                if param.shape != (num_trials,):
                    raise ValueError(f"{key} must have shape (num_trials,)")
        if "t_schedule" in params_array and params_array["t_schedule"].shape != params_array["v_schedule"].shape:
            raise ValueError("t_schedule must have same shape as v_schedule")

        # Check values
        if np.any(params_array["a"] <= 0) or np.any(params_array["sigma"] <= 0):
            raise ValueError("a, sigma must be > 0")
        if (np.any(params_array["s_v"] < 0)
        or np.any(params_array["angle"] < 0)
        or np.any(params_array["s_z"] < 0)
        or np.any(params_array["s_tau"] < 0)):
            raise ValueError("s_v, angle, s_z, s_tau must be >= 0")
        if "z" in params_array and np.any((params_array["z"] <= 0) | (params_array["z"] >= 1)):
            raise ValueError("0 < z < 1")

@njit
def simulate_super_ddm(
    v: np.ndarray, 
    a: np.ndarray, 
    z: np.ndarray, 
    tau: np.ndarray, 
    s_v: np.ndarray,
    sigma: np.ndarray, 
    angle: np.ndarray, 
    s_z: np.ndarray, 
    s_tau: np.ndarray,
    dt: float, 
    max_steps: int, 
    num_trials: int
) -> np.ndarray:
    """
    Simulate SuperDDM with single or mixture drift for num_trials trials.

    Parameters
    ----------
    v : np.ndarray
        Drift rate or components, shape (num_trials,) or (num_trials, num_components).
    a, z, tau, s_v, sigma, angle, s_z, s_tau : np.ndarray
        Parameters of shape (num_trials,).
    dt : float
        Time step.
    max_steps : int
        Maximum simulation steps.
    num_trials : int
        Number of trials.

    Returns
    -------
    np.ndarray
        Array of shape (num_trials, 2) with columns [rts, choices].
    """

    result = np.zeros((num_trials, 2))
    rts, choices = result[:, 0], result[:, 1]
    for i in prange(num_trials):
        v_i = np.random.normal(np.mean(v[i]) if v.shape != (num_trials,) else v[i], s_v[i])
        z_i = np.random.normal(z[i], s_z[i])
        z_i = np.clip(z_i, 0.001, 0.999)
        tau_i = np.random.normal(tau[i], s_tau[i])
        tau_i = max(tau_i, 0.0)
        x = z_i * a[i]
        t = 0.0
        for step in range(max_steps):
            bound = a[i] * (1.0 - angle[i] * t)
            bound = max(bound, 0.0)
            x += v_i * dt + sigma[i] * np.sqrt(dt) * np.random.normal()
            t += dt
            if x >= bound:
                rts[i] = t + tau_i
                choices[i] = 1
                break
            elif x <= -bound:
                rts[i] = t + tau_i
                choices[i] = 0
                break
        else:
            rts[i] = np.nan
            choices[i] = np.nan
    return result

@njit
def simulate_schedule_ddm(
    v_schedule: np.ndarray, 
    t_schedule: np.ndarray, 
    a: np.ndarray, 
    z: np.ndarray,
    tau: np.ndarray, 
    s_v: np.ndarray, 
    sigma: np.ndarray, 
    angle: np.ndarray,
    s_z: np.ndarray, 
    s_tau: np.ndarray, 
    dt: float, 
    max_steps: int, 
    num_trials: int
) -> np.ndarray:
    """
    Simulate SuperDDM with scheduled drift for num_trials trials.

    Parameters
    ----------
    v_schedule, t_schedule : np.ndarray
        Drift and time schedules, shape (num_trials, num_segments).
    a, z, tau, s_v, sigma, angle, s_z, s_tau : np.ndarray
        Parameters of shape (num_trials,).
    dt : float
        Time step.
    max_steps : int
        Maximum simulation steps.
    num_trials : int
        Number of trials.

    Returns
    -------
    np.ndarray
        Array of shape (num_trials, 2) with columns [rts, choices].
    """

    result = np.zeros((num_trials, 2))
    rts, choices = result[:, 0], result[:, 1]
    for i in prange(num_trials):
        z_i = np.random.normal(z[i], s_z[i])
        z_i = np.clip(z_i, 1e-3, 1.-1e-3)
        tau_i = np.random.normal(tau[i], s_tau[i])
        tau_i = max(tau_i, 0.0)
        x = z_i * a[i]
        t = 0.0
        step = 0
        v_index = 0
        v = v_schedule[i, v_index]
        t_next = t_schedule[i, v_index] if t_schedule.shape[1] > v_index else np.inf
        while step < max_steps:
            if t >= t_next:
                v_index += 1
                v = v_schedule[i, v_index] if v_index < v_schedule.shape[1] else v
                t_next = t_schedule[i, v_index] if v_index < t_schedule.shape[1] else np.inf
            bound = a[i] * (1.0 - angle[i] * t)
            bound = max(bound, 0.0)
            x += np.random.normal(v, s_v[i]) * dt + sigma[i] * np.sqrt(dt) * np.random.normal()
            t += dt
            step += 1
            if x >= bound:
                rts[i] = t + tau_i
                choices[i] = 1
                break
            elif x <= -bound:
                rts[i] = t + tau_i
                choices[i] = 0
                break
        else:
            rts[i] = np.nan
            choices[i] = np.nan
    return result
