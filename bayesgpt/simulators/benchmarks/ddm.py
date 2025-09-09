import numpy as np
from numba import njit, prange


@njit(parallel=True)
def simulate_standard_ddm(
    v: np.ndarray,
    a: np.ndarray,
    z: float,
    tau: float,
    s_v: float,
    sigma: float,
    s_z: float,
    s_tau: float,
    dt: float,
    max_steps: int,
    num_samples: int
) -> np.ndarray:
    """
    Internal function to simulate the standard DDM with static boundaries.
    """
    # Initialize output arrays
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]

    # Simulate each trial
    for i in prange(num_samples):
        # Sample parameters with noise
        v_i = np.random.normal(v[i], s_v)
        z_i = max(min(np.random.normal(z, s_z), 0.999), 0.001)
        tau_i = max(np.random.normal(tau, s_tau), 0.0)

        # Initialize decision variable
        x = z_i * a
        t = tau_i

        # Simulation loop
        for step in range(max_steps):
            bound = a[i]  # Static boundary
            x += v_i * dt + sigma * np.sqrt(dt) * np.random.normal()
            t += dt
            if x >= bound:
                rts[i] = t
                choices[i] = 1
                break
            elif x <= -bound:
                rts[i] = t
                choices[i] = 0
                break
        else:
            rts[i] = -1.
            choices[i] = -1.

    return result


@njit(parallel=True)
def simulate_collapsing_bound_ddm(
    v: np.ndarray,
    a: np.ndarray,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    zr: float = 0.5,
    sigma: float = 1.,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:

    num_samples = v.shape[0]
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]

    # Simulate each trial
    for i in prange(num_samples):
        # Sample parameters with noise
        v_i = np.random.normal(v[i], s_v)
        tau_i = max(np.random.normal(tau, s_tau), 0.0)

        # Initialize decision variable (symmetric around 0)
        x = zr * a[i]
        t = tau_i

        # Simulation loop
        for step in range(max_steps):
            t += dt
            bound = max(a[i] - decay * t, 1e-3)
            x += v_i * dt + sigma * np.sqrt(dt) * np.random.normal()
            if x >= bound:
                rts[i] = t
                choices[i] = 1
                break
            elif x <= -bound:
                rts[i] = t
                choices[i] = 0
                break
        else:
            rts[i] = -1.
            choices[i] = -1.

    return result


@njit(parallel=True)
def simulate_mixture_ddm(
    v: np.ndarray,
    p: np.ndarray,
    a: np.ndarray,
    z: float,
    tau: float,
    s_v: float,
    decay: float,
    s_z: float,
    s_tau: float,
    sigma: float,
    dt: float,
    max_steps: int,
    num_samples: int
) -> np.ndarray:
    """
    Simulate a drift diffusion model with mixture drift rates.
    """
    # Initialize output arrays
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]

    # Simulate each trial
    for i in prange(num_samples):

        # Sample starting point and non-decision time
        z_i_sample = np.random.normal(z, s_z)
        z_i = z_i_sample if z_i_sample < 0.999 else 0.999
        z_i = z_i if z_i > 0.001 else 0.001
        tau_i = max(np.random.normal(tau, s_tau), 0.0)

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
        v_i = float(np.random.normal(v_i, s_v))

        # Initialize decision variable
        x = z_i * a
        t = 0.0

        # Run simulation loop
        for step in range(max_steps):
            bound = max(a * (1.0 - decay * t), 0.0)
            x += v_i * dt + sigma * np.sqrt(dt) * np.random.normal()
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
            rts[i] = -1.
            choices[i] = -1.
    return result

@njit(parallel=True)
def simulate_schedule_ddm(
    v_schedule: np.ndarray,
    t_schedule: np.ndarray,
    a: float,
    z: float,
    tau: float,
    s_v: float,
    decay: float,
    s_z: float,
    s_tau: float,
    sigma: float,
    dt: float,
    max_steps: int,
    num_samples: int
) -> np.ndarray:

    # Initialize output arrays
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]

    # Simulate each trial
    for i in prange(num_samples):

        # Sample starting point and non-decision time
        z_i = max(min(np.random.normal(z, s_z), 0.999), 0.001)
        tau_i = max(np.random.normal(tau, s_tau), 0.)

        # Initialize decision variable and drift schedule
        x = z_i * a
        t = 0.
        step = 0
        v_index = 0
        v = np.random.normal(v_schedule[i, v_index], s_v)  # Sample v for first segment
        t_next = t_schedule[i, v_index] if t_schedule.shape[1] > v_index else np.inf

        # Run simulation loop
        while step < max_steps:
            # Update drift rate if time exceeds next schedule point
            if t >= t_next:
                v_index += 1
                if v_index < v_schedule.shape[1]:
                    v = np.random.normal(v_schedule[i, v_index], s_v)
                    t_next = t_schedule[i, v_index]
                else:
                    t_next = np.inf

            # Update decision variable and boundary
            bound = max(a * (1. - decay * t), 0.)
            x += v * dt + sigma * np.sqrt(dt) * np.random.normal(0.0, 1.0)
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
            rts[i] = -1.
            choices[i] = -1.
    return result
