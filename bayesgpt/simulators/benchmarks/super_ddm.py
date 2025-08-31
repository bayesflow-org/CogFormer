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
            num_samples: int = 1,
            context: Optional[np.ndarray] = None,
            context_modulation: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    ) -> dict[str, np.ndarray]:
        # Process and validate input parameters
        params = self._process_parameters(params, num_samples)
        self._validate_parameters(params, num_samples)

        # Modulate drift rates with context if provided
        if context is not None and context_modulation is not None:
            if context.shape != (num_samples,):
                raise ValueError(f"Context must have shape ({num_samples},), got {context.shape}")
            if "v_schedule" in params:
                params["v_schedule"] = context_modulation(params["v_schedule"], context[:, None]).astype(np.float32)
            elif "v_components" in params:
                params["v_components"] = context_modulation(params["v_components"], context[:, None]).astype(np.float32)
            else:
                params["v"] = context_modulation(params["v"], context).astype(np.float32)

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
            result = simulate_super_ddm(
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
            Input parameters to process.
        num_samples : int
            Number of trials.

        Returns
        -------
        dict[str, np.ndarray]
            Processed parameters with shapes (num_samples,) or (num_samples, num_segments).

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

@njit
def simulate_super_ddm(
    v: np.ndarray,
    p: np.ndarray,
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
    num_samples: int
) -> np.ndarray:
    """Simulate DDM with single or mixture drift rates.

    Parameters
    ----------
    v : np.ndarray
        Drift rate(s), shape (num_samples,) or (num_samples, num_components).
    p : np.ndarray
        Probabilities for mixture components, shape (num_samples, num_components) or None.
    a, z, tau, s_v, sigma, angle, s_z, s_tau : np.ndarray
        Parameters, shape (num_samples,).
    dt : float
        Time step.
    max_steps : int
        Maximum simulation steps.
    num_samples : int
        Number of trials.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 2) with columns [rts, choices].
    """
    result = np.zeros((num_samples, 2))
    rts, choices = result[:, 0], result[:, 1]
    for i in range(num_samples):
        # Ensure scalar parameters
        a_i = float(a[i])
        sigma_i = float(sigma[i])
        angle_i = float(angle[i])
        s_v_i = float(s_v[i])
        s_z_i = float(s_z[i])
        s_tau_i = float(s_tau[i])
        z_i = float(max(min(np.random.normal(float(z[i]), s_z_i), 0.999), 0.001))
        tau_i = float(max(np.random.normal(float(tau[i]), s_tau_i), 0.0))

        # Select drift rate (single or random choice from components)
        if v.ndim == 1:
            v_i = float(v[i])
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
        for step in range(max_steps):
            # Compute time-dependent boundary
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
    num_samples: int
) -> np.ndarray:
    """Simulate DDM with scheduled drift rates.

    Parameters
    ----------
    v_schedule, t_schedule : np.ndarray
        Drift and time schedules, shape (num_samples, num_segments).
    a, z, tau, s_v, sigma, angle, s_z, s_tau : np.ndarray
        Parameters, shape (num_samples,).
    dt : float
        Time step.
    max_steps : int
        Maximum simulation steps.
    num_samples : int
        Number of trials.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 2) with columns [rts, choices].
    """
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]
    for i in prange(num_samples):
        # Sample starting point and non-decision time
        z_i = max(min(np.random.normal(z[i], s_z[i]), 0.999), 0.001)
        tau_i = max(np.random.normal(tau[i], s_tau[i]), 0.)
        x = z_i * a[i]
        t = 0.
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
            bound = max(a[i] * (1. - angle[i] * t), 0.)
            x += np.random.normal(v, s_v)
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
