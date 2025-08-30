import numpy as np
from typing import Union, Optional, Callable
from numba import njit, prange
from ..model import Model


class SuperDDM(Model):
    """Simulate a flexible drift diffusion model with single, mixture, or scheduled drifts.

    Supports context modulation for drift rates (e.g., task difficulty) and integrates with
    NestedModelFamily for simulation-based inference workflows.
    """

    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        """Initialize SuperDDM with simulation parameters.

        Parameters
        ----------
        dt : float, optional
            Time step for simulation (seconds), by default 0.001.
        max_steps : int, optional
            Maximum number of simulation steps, by default 10000.
        """
        self.dt = dt
        self.max_steps = max_steps

    def simulate(
        self,
        params: dict[str, Union[float, np.ndarray]],
        num_trials: int = 1,
        context: Optional[np.ndarray] = None,
        context_modulation: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    ) -> dict[str, np.ndarray]:
        """Simulate response times and choices for multiple trials.

        Parameters
        ----------
        params : dict[str, Union[float, np.ndarray]]
            Model parameters:
            - v : float or array, mean drift rate (required if no v_components or v_schedule).
            - a : float or array, boundary separation (>0).
            - z : float or array, starting point fraction (0 < z < 1).
            - tau : float or array, non-decision time (>=0).
            - sigma : float or array, diffusion noise (>0).
            - angle : float or array, collapse rate (>=0, default=0 for fixed bounds).
            - s_v : float or array, drift variability (>=0, default=0).
            - s_z : float or array, starting point variability (>=0, default=0).
            - s_tau : float or array, non-decision time variability (>=0, default=0).
            - v_components : array, drift rates for mixture model (select one per trial).
            - p_components : array, probabilities for mixture components (sums to 1, default uniform).
            - v_schedule : array, drift rates for within-trial piecewise-constant segments.
            - t_schedule : array, durations (seconds) for each segment (>0).
        num_trials : int, optional
            Number of trials to simulate, by default 1.
        context : np.ndarray, optional
            Context array (shape: (num_trials,)) to modulate drift rates, e.g., task difficulty.
        context_modulation : callable, optional
            Function to modulate drift rates: f(v, context) -> modulated_v.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys:
            - 'rts': Response times (shape: (num_trials,)).
            - 'choices': Choices (0 or 1, shape: (num_trials,)).
            - 'context': Input context (if provided, shape: (num_trials,)).

        Raises
        ------
        ValueError
            If parameters are missing, invalid, or have incorrect shapes.

        Examples
        --------
        >>> model = SuperDDM(dt=0.01, max_steps=1000)
        >>> params = {"v": 0.5, "a": 1.0, "z": 0.5, "tau": 0.1, "sigma": 1.0,
        ...           "angle": 0.0, "s_v": 0.1, "s_z": 0.0, "s_tau": 0.0}
        >>> context = np.random.uniform(0.5, 1.5, 100)
        >>> result = model.simulate(params, num_trials=100, context=context,
        ...                         context_modulation=lambda v, c: v * c)
        """
        # Process and validate input parameters
        params = self._process_parameters(params, num_trials)
        self._validate_parameters(params, num_trials)

        # Modulate drift rates with context if provided
        if context is not None and context_modulation is not None:
            if context.shape != (num_trials,):
                raise ValueError(f"Context must have shape ({num_trials},), got {context.shape}")
            if "v_schedule" in params:
                params["v_schedule"] = context_modulation(params["v_schedule"], context[:, None])
            elif "v_components" in params:
                params["v_components"] = context_modulation(params["v_components"], context[:, None])
            else:
                params["v"] = context_modulation(params["v"], context)

        # Run appropriate simulation based on drift type
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

        # Construct output with optional context
        output = {"rts": result[:, 0], "choices": result[:, 1]}
        if context is not None:
            output["context"] = context
        return output


    def _process_parameters(
        self,
        params: dict[str, Union[float, np.ndarray]],
        num_trials: int
    ) -> dict[str, np.ndarray]:
        """Process parameters to arrays of consistent shape.

        Parameters
        ----------
        params : dict[str, Union[float, np.ndarray]]
            Input parameters to process.
        num_trials : int
            Number of trials.

        Returns
        -------
        dict[str, np.ndarray]
            Processed parameters with shapes (num_trials,) or (num_trials, num_segments).

        Raises
        ------
        ValueError
            If required parameters are missing or have invalid values.
        """
        # Check for required parameters
        required_params = ["a", "s_v", "sigma", "angle", "s_z", "s_tau"]
        if not any(k in params for k in ["v", "v_components", "v_schedule"]):
            raise ValueError("One of 'v', 'v_components', or 'v_schedule' must be provided")
        if not any(k in params for k in ["z", "z_arr"]):
            raise ValueError("One of 'z' or 'z_arr' must be provided")
        if not any(k in params for k in ["tau", "tau_arr"]):
            raise ValueError("One of 'tau' or 'tau_arr' must be provided")
        if not all(k in params for k in required_params):
            raise ValueError(f"Missing parameters: {set(required_params) - set(params)}")

        # Handle parameter aliases
        if "z_arr" in params:
            params["z"] = np.full(num_trials, params["z_arr"]) if np.isscalar(params["z_arr"]) else params["z_arr"]
            del params["z_arr"]
        if "tau_arr" in params:
            params["tau"] = np.full(num_trials, params["tau_arr"]) if np.isscalar(params["tau_arr"]) else params["tau_arr"]
            del params["tau_arr"]

        # Broadcast scalar parameters to arrays
        for key in ["v", "a", "z", "tau", "s_v", "sigma", "angle", "s_z", "s_tau"]:
            if key in params:
                params[key] = np.full(num_trials, params[key]) if np.isscalar(params[key]) else params[key]

        # Handle mixture or scheduled drifts
        if "v_components" in params:
            params["v_components"] = (
                np.full((num_trials, params["v_components"].shape[-1]), params["v_components"])
                if params["v_components"].ndim == 1
                else params["v_components"]
            )
            if "p_components" in params:
                params["p_components"] = (
                    np.full((num_trials, params["v_components"].shape[-1]), params["p_components"])
                    if params["p_components"].ndim == 1
                    else params["p_components"]
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
        """Validate parameter shapes and values.

        Parameters
        ----------
        params_array : dict[str, np.ndarray]
            Processed parameters to validate.
        num_trials : int
            Number of trials.

        Raises
        ------
        ValueError
            If parameters have invalid shapes or values.
        """
        # Validate parameter shapes
        for key, param in params_array.items():
            if key in ["v_components", "v_schedule", "t_schedule", "p_components"]:
                if param.shape[0] != num_trials:
                    raise ValueError(f"{key} must have shape (num_trials, num_segments)")
            else:
                if param.shape != (num_trials,):
                    raise ValueError(f"{key} must have shape (num_trials,)")
        if "t_schedule" in params_array and params_array["t_schedule"].shape != params_array["v_schedule"].shape:
            raise ValueError("t_schedule must have same shape as v_schedule")
        if "p_components" in params_array and params_array["p_components"].shape != params_array["v_components"].shape:
            raise ValueError("p_components must have same shape as v_components")

        # Validate parameter values
        if np.any(params_array["a"] <= 0) or np.any(params_array["sigma"] <= 0):
            raise ValueError("a, sigma must be > 0")
        if np.any(params_array["s_v"] < 0) or np.any(params_array["angle"] < 0) or np.any(params_array["s_z"] < 0) or np.any(params_array["s_tau"] < 0):
            raise ValueError("s_v, angle, s_z, s_tau must be >= 0")
        if "z" in params_array and np.any((params_array["z"] <= 0) | (params_array["z"] >= 1)):
            raise ValueError("0 < z < 1")
        if "p_components" in params_array and np.any(params_array["p_components"] < 0):
            raise ValueError("p_components must be >= 0")


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
    """Simulate DDM with single or mixture drift rates.

    Parameters
    ----------
    v : np.ndarray
        Drift rate(s), shape (num_trials,) or (num_trials, num_components).
    a, z, tau, s_v, sigma, angle, s_z, s_tau : np.ndarray
        Parameters, shape (num_trials,).
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
    # Buffer
    result = np.zeros((num_trials, 2))
    rts, choices = result[:, 0], result[:, 1]

    # Simulate
    for i in prange(num_trials):
        # Select drift rate (single or random choice from components)
        v_i = v[i] if v.ndim == 1 else np.random.choice(v[i])
        v_i = np.random.normal(v_i, s_v[i])

        # Sample starting point and non-decision time
        z_i = max(min(np.random.normal(z[i], s_z[i]), 1.0 - 1e-3), 1e-3)
        tau_i = max(np.random.normal(tau[i], s_tau[i]), 0.0)
        x = z_i * a[i]
        t = 0.0

        for step in range(max_steps):
            # Compute time-dependent boundary
            bound = max(a[i] * (1.0 - angle[i] * t), 0.0)
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
    """Simulate DDM with scheduled drift rates.

    Parameters
    ----------
    v_schedule, t_schedule : np.ndarray
        Drift and time schedules, shape (num_trials, num_segments).
    a, z, tau, s_v, sigma, angle, s_z, s_tau : np.ndarray
        Parameters, shape (num_trials,).
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
    # Buffer
    result = np.zeros((num_trials, 2))
    rts, choices = result[:, 0], result[:, 1]

    # Simulate
    for i in prange(num_trials):
        # Sample starting point and non-decision time
        z_i = max(min(np.random.normal(z[i], s_z[i]), 1.0 - 1e-3), 1e-3)
        tau_i = max(np.random.normal(tau[i], s_tau[i]), 0.0)
        x = z_i * a[i]
        t = 0.0

        # Keeping track of schedule
        step = 0
        v_index = 0
        v = v_schedule[i, v_index]
        t_next = t_schedule[i, v_index] if t_schedule.shape[1] > v_index else np.inf

        while step < max_steps:
            # Update drift rate based on schedule
            if t >= t_next:
                v_index += 1
                v = v_schedule[i, v_index] if v_index < v_schedule.shape[1] else v
                t_next = t_schedule[i, v_index] if v_index < t_schedule.shape[1] else np.inf

            # Compute time-dependent boundary
            bound = max(a[i] * (1.0 - angle[i] * t), 0.0)
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
