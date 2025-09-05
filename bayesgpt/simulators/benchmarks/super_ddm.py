import numpy as np
from typing import Union, Optional, Callable
from numba import njit, prange
from ..model import Model


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

    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        """Initialize SuperDDM with simulation parameters.

        Parameters
        ----------
        dt : float, optional
            Time step for simulation in seconds, by default 0.001.
        max_steps : int, optional
            Maximum number of simulation steps, by default 10000.
        """
        self.dt = dt
        self.max_steps = max_steps

    def simulate(
            self,
            params: dict[str, Union[float, np.ndarray]],
            num_samples: int = 1,
            context: Optional[np.ndarray] = None,
            modulation: Optional[Callable[[dict, np.ndarray], dict]] = None
    ) -> dict[str, np.ndarray]:
        """Simulate data for a single model run with specified parameters.

        Parameters
        ----------
        params : dict[str, Union[float, np.ndarray]]
            Parameters for simulation, with arrays of shape (dims,) or scalars.
            Must include:
            - a : float or np.ndarray
                Decision boundary (threshold), controls decision caution, > 0.
            - s_v : float or np.ndarray
                Drift rate noise, standard deviation of drift rate variability, >= 0.
            - sigma : float or np.ndarray
                Diffusion noise, standard deviation of evidence accumulation noise, > 0.
            - angle : float or np.ndarray
                Boundary collapse rate, controls rate of boundary reduction over time, >= 0.
            - s_z : float or np.ndarray
                Starting point noise, standard deviation of starting point variability, >= 0.
            - s_tau : float or np.ndarray
                Non-decision time noise, standard deviation of non-decision time variability, >= 0.
            Must include one of:
            - v : float or np.ndarray
                Single drift rate for all trials, determines evidence accumulation speed.
            - v_components : np.ndarray
                Drift rates for mixture model, shape (num_samples, num_components).
            - v_schedule : np.ndarray
                Drift rates for scheduled model, shape (num_samples, num_segments).
            Optional:
            - z or z_arr : float or np.ndarray
                Starting point, relative to boundaries (0 < z < 1).
            - tau or tau_arr : float or np.ndarray
                Non-decision time, time before evidence accumulation, >= 0.
            - p_components : np.ndarray
                Mixture probabilities for v_components, shape (num_samples, num_components).
            - t_schedule : np.ndarray
                Time points for v_schedule changes, shape (num_samples, num_segments).
        num_samples : int, optional
            Number of samples (trials) per simulation, by default 1.
        context : np.ndarray, optional
            Array of external conditions (e.g., task difficulty, stimulus strength) for each trial,
            shape (num_samples,). Can influence model parameters via modulation and is included
            in the output for use in downstream tasks like neural network training.
        modulation : Callable[[dict, np.ndarray], dict], optional
            Function to adjust model parameters based on context. Takes the params dictionary
            and context array as input and returns an updated params dictionary. Must ensure
            all modified parameters respect model constraints (e.g., a > 0, 0 < z < 1).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing simulated data with keys 'rts' (reaction times),
            'choices' (binary choices), and optionally 'context'. Arrays are of
            shape (num_samples,) in np.float32.

        Raises
        ------
        ValueError
            If context shape is invalid, required parameters are missing, or modulated parameters
            violate model constraints.
        """
        # Process and validate input parameters
        params = self._process_parameters(params, num_samples)
        self._validate_parameters(params, num_samples)

        # Modulate parameters with context if provided
        if context is not None and modulation is not None:
            if context.shape != (num_samples,):
                raise ValueError(f"context must have shape ({num_samples},), got {context.shape}")
            params = modulation(params, context)
            self._validate_parameters(params, num_samples)  # Re-validate after modulation

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
                sigma=params["sigma"],
                angle=params["angle"],
                s_z=params["s_z"],
                s_tau=params["s_tau"],
                dt=self.dt,
                max_steps=self.max_steps,
                num_samples=num_samples,
            )

        # Construct output with optional context
        output = {"rts": result[:, 0], "choices": result[:, 1]}
        if context is not None:
            output["context"] = context
        return output

    def _process_parameters(
        self,
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
            (num_samples, num_segments) for multi-dimensional parameters.

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
            params["z"] = np.full(num_samples, params["z_arr"]).astype(np.float32) if np.isscalar(params["z_arr"]) \
                else params["z_arr"].astype(np.float32)
            del params["z_arr"]
        if "tau_arr" in params:
            params["tau"] = np.full(num_samples, params["tau_arr"]).astype(np.float32) if np.isscalar(params["tau_arr"]) \
                else params["tau_arr"].astype(np.float32)
            del params["tau_arr"]

        # Broadcast scalar parameters to arrays
        for key in ["v", "a", "z", "tau", "s_v", "sigma", "angle", "s_z", "s_tau"]:
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
        self,
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
                if param.shape != (num_samples,):
                    raise ValueError(f"{key} must have shape (num_samples,)")
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

@njit(parallel=True)
def simulate_mixture_ddm(
    v: float | np.ndarray,
    p: float | np.ndarray,
    a: float | np.ndarray,
    z: float | np.ndarray,
    tau: float | np.ndarray,
    s_v: float | np.ndarray,
    sigma: float | np.ndarray,
    angle: float | np.ndarray,
    s_z: float | np.ndarray,
    s_tau: float | np.ndarray,
    dt: float,
    max_steps: int,
    num_samples: int
) -> np.ndarray:
    """Simulate a drift diffusion model with mixture drift rates.

    Parameters
    ----------
    v : float or np.ndarray
        Drift rate for single drift model or component drift rates for mixture model,
        shape (num_samples,) or (num_samples, num_components).
    p : float or np.ndarray, optional
        Mixture probabilities for v_components, shape (num_samples, num_components).
    a : float or np.ndarray
        Decision boundary, shape (num_samples,), > 0.
    z : float or np.ndarray
        Starting point, shape (num_samples,), 0 < z < 1.
    tau : float or np.ndarray
        Non-decision time, shape (num_samples,), >= 0.
    s_v : float or np.ndarray
        Drift rate noise standard deviation, shape (num_samples,), >= 0.
    sigma : float or np.ndarray
        Diffusion noise standard deviation, shape (num_samples,), > 0.
    angle : float or np.ndarray
        Boundary collapse rate, shape (num_samples,), >= 0.
    s_z : float or np.ndarray
        Starting point noise standard deviation, shape (num_samples,), >= 0.
    s_tau : float or np.ndarray
        Non-decision time noise standard deviation, shape (num_samples,), >= 0.
    dt : float
        Time step for simulation (seconds).
    max_steps : int
        Maximum number of simulation steps.
    num_samples : int
        Number of trials to simulate.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 2) containing reaction times (column 0) and choices (column 1).
    """
    # Initialize output arrays
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]

    # Simulate each trial
    for i in prange(num_samples):
        # Extract parameters for the trial
        a_i = a[i]
        sigma_i = sigma[i]
        angle_i = angle[i]
        s_v_i = s_v[i]
        s_z_i = s_z[i]
        s_tau_i = s_tau[i]

        # Sample starting point and non-decision time
        z_i_sample = np.random.normal(z[i], s_z_i)
        z_i = z_i_sample if z_i_sample < 0.999 else 0.999
        z_i = z_i if z_i > 0.001 else 0.001
        tau_i = max(np.random.normal(tau[i], s_tau_i), 0.0)

        # Select drift rate (single or random choice from components)
        if v.ndim == 1:
            v_i = v[i]
        else:
            num_components = v.shape[1]
            p_i = np.ones(num_components).astype(np.float32) / num_components if p is None else p[i]
            p_i = p_i / np.sum(p_i)  # Normalize probabilities
            r = np.random.random()
            cumsum = 0.0
            selected_idx = 0
            for j in range(num_components):
                cumsum += float(p_i[j])
                if r <= cumsum:
                    selected_idx = j
                    break
            v_i = float(v[i, selected_idx])
        v_i = float(np.random.normal(v_i, s_v_i))

        # Initialize decision variable
        x = z_i * a_i
        t = 0.0

        # Run simulation loop
        for step in range(max_steps):
            bound = max(a_i * (1.0 - angle_i * t), 0.0)
            x += v_i * dt + sigma_i * np.sqrt(dt) * np.random.normal()
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

@njit(parallel=True)
def simulate_schedule_ddm(
    v_schedule: np.ndarray,
    t_schedule: np.ndarray,
    a: float | np.ndarray,
    z: float | np.ndarray,
    tau: float | np.ndarray,
    s_v: float | np.ndarray,
    sigma: float | np.ndarray,
    angle: float | np.ndarray,
    s_z: float | np.ndarray,
    s_tau: float | np.ndarray,
    dt: float,
    max_steps: int,
    num_samples: int
) -> np.ndarray:
    """Simulate a drift diffusion model with scheduled drift rates.

    Parameters
    ----------
    v_schedule : np.ndarray
        Drift rates for scheduled model, shape (num_samples, num_segments).
    t_schedule : np.ndarray
        Time points for drift rate changes, shape (num_samples, num_segments).
    a : float or np.ndarray
        Decision boundary, shape (num_samples,), > 0.
    z : float or np.ndarray
        Starting point, shape (num_samples,), 0 < z < 1.
    tau : float or np.ndarray
        Non-decision time, shape (num_samples,), >= 0.
    s_v : float or np.ndarray
        Drift rate noise standard deviation, shape (num_samples,), >= 0.
    sigma : float or np.ndarray
        Diffusion noise standard deviation, shape (num_samples,), > 0.
    angle : float or np.ndarray
        Boundary collapse rate, shape (num_samples,), >= 0.
    s_z : float or np.ndarray
        Starting point noise standard deviation, shape (num_samples,), >= 0.
    s_tau : float or np.ndarray
        Non-decision time noise standard deviation, shape (num_samples,), >= 0.
    dt : float
        Time step for simulation (seconds).
    max_steps : int
        Maximum number of simulation steps.
    num_samples : int
        Number of trials to simulate.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 2) containing reaction times (column 0) and choices (column 1).
    """
    # Initialize output arrays
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]

    # Simulate each trial
    for i in prange(num_samples):
        # Extract parameters for the trial
        a_i = a[i]
        sigma_i = sigma[i]
        angle_i = angle[i]
        s_v_i = s_v[i]
        s_z_i = s_z[i]
        s_tau_i = s_tau[i]

        # Sample starting point and non-decision time
        z_i = max(min(np.random.normal(z[i], s_z[i]), 0.999), 0.001)
        tau_i = max(np.random.normal(tau[i], s_tau[i]), 0.)

        # Initialize decision variable and drift schedule
        x = z_i * a_i
        t = 0.
        step = 0
        v_index = 0
        v = np.random.normal(v_schedule[i, v_index], s_v_i)  # Sample v for first segment
        t_next = t_schedule[i, v_index] if t_schedule.shape[1] > v_index else np.inf

        # Run simulation loop
        while step < max_steps:
            # Update drift rate if time exceeds next schedule point
            if t >= t_next:
                v_index += 1
                if v_index < v_schedule.shape[1]:
                    v = np.random.normal(v_schedule[i, v_index], s_v_i)
                    t_next = t_schedule[i, v_index]
                else:
                    t_next = np.inf

            # Update decision variable and boundary
            bound = max(a_i * (1. - angle_i * t), 0.)
            x += v * dt + sigma_i * np.sqrt(dt) * np.random.normal(0.0, 1.0)
            t += dt
            step += 1

            # Check for boundary crossing
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
