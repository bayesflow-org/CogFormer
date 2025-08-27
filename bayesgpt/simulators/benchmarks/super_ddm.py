import numpy as np
from numba import njit, prange
from ..model import Model


class SuperDDM(Model):
    """
    Drift Diffusion Model (DDM) with across-trial variability in key parameters,
    optional linearly collapsing bounds, and support for multiple drift rates.

    This model simulates human-like decision-making by incorporating trial-to-trial
    variability in drift rate (evidence accumulation), starting point (bias), and
    non-decision time (sensory/motor delays). Boundaries can optionally collapse
    linearly over time to model urgency effects.

    Multiple drift rates can be specified either as an across-trial mixture
    (randomly select one drift per trial) or as a within-trial schedule
    (piecewise-constant drifts switching at fixed times).

    State evolution uses Euler-Maruyama discretization:

        x_{t+dt} = x_t + v(t or i) * dt + sigma * sqrt(dt) * N(0, 1)

    Trials start at a symmetric position:

        x0_i = (2 * z_i - 1) * a

    Symmetric boundaries may collapse linearly:

        B(t) = max(a - angle * t, 1e-3)

    A trial ends when x_t crosses +B(t) (choice=1) or -B(t) (choice=0).
    Reaction time is decision time plus tau_i.

    Notes
    -----
    Precedence for drift specification:
    1. If both v_schedule and t_schedule are provided, use within-trial schedule.
    2. Else if v_components is provided, use across-trial mixture.
    3. Else, use scalar v with optional s_v variability.

    If a trial does not terminate within max_steps, RT and choice are NaN.
    """

    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        self.dt = dt
        self.max_steps = max_steps

    def simulate(
        self, params: dict[str, float], batch_size: int
    ) -> dict[str, np.ndarray]:
        """
        Simulate response times and choices for a batch of trials.

        Parameters
        ----------
        params : dict[str, float]
            Dictionary containing model parameters:
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
        batch_size : int
            Number of trials to simulate.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys 'rts' and 'choices', each mapping to an array
            of shape (batch_size,) containing response times and choices.
        Raises
        ------
        ValueError
            If required_params parameters are missing or have invalid values.
        """
        required_params = ["a", "z", "tau", "sigma"]
        if not all(k in params for k in required_params):
            raise ValueError(f"Missing parameters: {set(required_params) - set(params)}")
        if params["a"] <= 0 or params["sigma"] <= 0 or not (0 < params["z"] < 1):
            raise ValueError("Invalid parameter values: a, sigma must be > 0, 0 < z < 1")
        if params["tau"] < 0:
            raise ValueError("Invalid parameter values: tau must be >= 0")
        
        # required parameters
        a = params["a"]
        z = params["z"]
        tau = params["tau"]
        sigma = params["sigma"]

        # Optional parameters with defaults
        angle = params.get("angle", 0.0)
        s_v = params.get("s_v", 0.0)
        s_z = params.get("s_z", 0.0)
        s_tau = params.get("s_tau", 0.0)

        # Draw per-trial z and tau (as arrays)
        z_arr = (
            np.random.normal(z, s_z, batch_size) if s_z > 0 else np.full(batch_size, z)
        )
        z_arr = np.clip(z_arr, 1e-3, 1 - 1e-3)
        tau_arr = (
            np.random.normal(tau, s_tau, batch_size)
            if s_tau > 0
            else np.full(batch_size, tau)
        )
        tau_arr = np.maximum(tau_arr, 1e-3)

        # Handle drift configuration
        v_schedule = params.get("v_schedule")
        t_schedule = params.get("t_schedule")

        if v_schedule is not None and t_schedule is not None:
            # Within-trial variability with drift schedules
            v_schedule = np.asarray(v_schedule, dtype=np.float32)
            t_schedule = np.asarray(t_schedule, dtype=np.float32)

            if v_schedule.shape != t_schedule.shape:
                raise ValueError("v_schedule and t_schedule must have the same shape.")
            if np.any(t_schedule <= 0):
                raise ValueError("All t_schedule values must be > 0.")

            # Keep track of within-trial switching regime
            K = len(v_schedule)
            segment_switch = np.cumsum(t_schedule)

            if s_v > 0:
                v_segments = np.random.normal(v_schedule[None, :], s_v, (batch_size, K))
            else:
                v_segments = np.tile(v_schedule[None, :], (batch_size, 1))

            result = _simulate_schedule_ddm(
                v_segments,
                segment_switch,
                a,
                z_arr,
                tau_arr,
                sigma,
                angle,
                self.dt,
                self.max_steps,
                batch_size,
                K,
            )
        else:
            # Across-trial variability with multiple drift rates and their probabilities
            v_components = params.get("v_components")

            if v_components is not None:
                v_components = np.asarray(v_components, dtype=float)
                p_components = params.get("p_components")

                if p_components is None:
                    p = np.full(len(v_components), 1.0 / len(v_components))
                else:
                    p = np.asarray(p_components, dtype=float)
                    p = p / p.sum()

                indices = np.random.choice(len(v_components), batch_size, p=p)
                v_arr = v_components[indices]

                if s_v > 0:
                    v_arr += np.random.normal(0, s_v, batch_size)
            else:
                v = params["v"]
                v_arr = (
                    np.random.normal(v, s_v, batch_size)
                    if s_v > 0
                    else np.full(batch_size, v)
                )

            result = _simulate_super_ddm(
                v_arr,
                a,
                z_arr,
                tau_arr,
                sigma,
                angle,
                self.dt,
                self.max_steps,
                batch_size,
            )

        return {"rts": result[:, 0], "choices": result[:, 1]}


@njit
def _simulate_super_ddm(
    v: np.ndarray,
    a: float,
    z: np.ndarray,
    tau: np.ndarray,
    sigma: float,
    angle: float,
    dt: float,
    max_steps: int,
    batch_size: int,
) -> np.ndarray:
    """
    Simulate DDM with across-trial variability.

    Parameters
    ----------
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
    dt : float, optional
        Interval for each time step.
    max_steps : int, optional
        Maximum number of time steps to simulate.
    batch_size : int, optional
        Number of trials to simulate.

    Returns
    -------
    np.ndarray of shape (batch_size, 2) for the following simulated data:
        - rts: reaction time for each trial
        - choices: choices for each trial
    """

    rts = np.zeros(batch_size)
    choices = np.zeros(batch_size)

    for i in prange(batch_size):
        vi = v[i]
        x = (2 * z[i] - 1.0) * a
        t = 0.0
        tau_i = tau[i]

        for _ in range(max_steps):
            t += dt
            bound = max(a - angle * t, 1e-3)

            x += vi * dt + sigma * np.sqrt(dt) * np.random.normal()

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

    result = np.zeros((batch_size, 2))
    result[:, 0] = rts
    result[:, 1] = choices
    return result


@njit
def _simulate_schedule_ddm(
    v: np.ndarray,
    segment_switch: np.ndarray,
    a: float,
    z: np.ndarray,
    tau: np.ndarray,
    sigma: float,
    angle: float,
    dt: float,
    max_steps: int,
    batch_size: int,
    num_segments: int,
) -> np.ndarray:
    """
    Simulate DDM with across-trial variability.

    Parameters
    ----------
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
    dt : float, optional
        Interval for each time step.
    max_steps : int, optional
        Maximum number of time steps to simulate.
    batch_size : int, optional
        Number of trials to simulate.
    num_segments : int, optional
        Number of segments for within-trial variability.

    Returns
    -------
    np.ndarray of shape (batch_size, 2) for the following simulated data:
        - rts: reaction time for each trial
        - choices: choices for each trial
    """

    rts = np.zeros(batch_size)
    choices = np.zeros(batch_size)

    for i in prange(batch_size):
        x = (2 * z[i] - 1.0) * a
        t = 0.0
        tau_i = tau[i]
        seg_idx = 0
        next_switch = segment_switch[0] if num_segments > 0 else np.inf

        for _ in range(max_steps):
            t += dt
            bound = max(a - angle * t, 1e-3)

            if seg_idx < num_segments - 1 and t > next_switch - 1e-6:
                seg_idx += 1
                next_switch = segment_switch[seg_idx]

            vi = v[i, seg_idx]
            x += vi * dt + sigma * np.sqrt(dt) * np.random.normal()

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

    result = np.zeros((batch_size, 2))
    result[:, 0] = rts
    result[:, 1] = choices
    return result
