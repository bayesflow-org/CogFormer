import numpy as np
from numba import njit, prange


@njit
def simulate_standard_ddm_trial(
    v: float,
    a: float,
    z: float,
    tau: float,
    s_v: float,
    s_z: float,
    s_tau: float,
    sigma: float
):
    pass

@njit(parallel=True)
def simulate_standard_ddm(
    v: np.ndarray,
    a: np.ndarray,
    z: float,
    tau: float,
    s_v: float,
    s_z: float,
    s_tau: float,
    sigma: float,
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


@njit
def simulate_collapsing_bound_ddm_trial(
    v_base: float,
    a_i: float,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    zr: float,
    sigma: float,
    dt: float,
    max_steps: int,
) -> (float, float):
    v_i = np.random.normal(v_base, s_v)
    tau_i = max(np.random.normal(tau, s_tau), 0.0)

    x = zr * a_i
    t = tau_i

    for _ in range(max_steps):
        t += dt
        bound = max(a_i * np.exp(-decay * t), 1e-3)
        x += v_i * dt + sigma * np.sqrt(dt) * np.random.normal()
        if x >= bound:
            return t, 1.0
        if x <= -bound:
            return t, 0.0
    return -1.0, -1.0


@njit(parallel=True)
def simulate_collapsing_bound_ddm(
    v: np.ndarray,
    a: np.ndarray,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    zr: float = 0.5,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    n = v.shape[0]
    out = np.zeros((n, 2), dtype=np.float32)
    rts, choices = out[:, 0], out[:, 1]

    for i in prange(n):
        rt_i, ch_i = simulate_collapsing_bound_ddm_trial(
            float(v[i]), float(a[i]),
            tau, s_tau, s_v, decay, zr, sigma,
            dt, max_steps
        )
        rts[i] = rt_i
        choices[i] = ch_i
    return out


@njit(inline='always')
def simulate_mixture_ddm_trial(
    v_mean: float,
    a_i: float,
    z: float,
    tau: float,
    s_v: float,
    decay: float,
    s_z: float,
    s_tau: float,
    sigma: float,
    dt: float,
    max_steps: int,
) -> (float, float):
    z_i = min(max(np.random.normal(z, s_z), 0.001), 0.999)
    tau_i = max(np.random.normal(tau, s_tau), 0.0)
    v_i = np.random.normal(v_mean, s_v)

    x = z_i * a_i
    t = 0.0

    for _ in range(max_steps):
        bound = max(a_i * (1.0 - decay * t), 0.0)
        x += v_i * dt + sigma * np.sqrt(dt) * np.random.normal()
        t += dt
        if x >= bound:
            return t + tau_i, 1.0
        if x <= -bound:
            return t + tau_i, 0.0
    return -1.0, -1.0  # non-termination


@njit(parallel=True)
def simulate_mixture_ddm(
    v: np.ndarray,          # (N,) or (N,K) mixture means
    p: np.ndarray,          # (N,K) probs; if empty (shape[0]==0), assume uniform
    a: np.ndarray,          # (N,) boundary (trialwise)
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
    result = np.zeros((num_samples, 2), dtype=np.float32)
    rts, choices = result[:, 0], result[:, 1]

    for i in prange(num_samples):
        # pick drift component (supports v as (N,) or (N,K))
        if v.ndim == 1:
            v_mean = float(v[i])
        else:
            K = v.shape[1]
            # probabilities row (uniform if p is "empty")
            if p.shape[0] == 0:
                r = np.random.random()
                csum = 0.0
                v_mean = float(v[i, K - 1])
                for j in range(K):
                    csum += 1.0 / K
                    if r <= csum:
                        v_mean = float(v[i, j])
                        break
            else:
                # normalize p[i] and sample
                s = 0.0
                for j in range(K):
                    s += p[i, j]
                if s <= 0.0:
                    r = np.random.random()
                    csum = 0.0
                    v_mean = float(v[i, K - 1])
                    for j in range(K):
                        csum += 1.0 / K
                        if r <= csum:
                            v_mean = float(v[i, j])
                            break
                else:
                    r = np.random.random()
                    csum = 0.0
                    v_mean = float(v[i, K - 1])
                    for j in range(K):
                        csum += p[i, j] / s
                        if r <= csum:
                            v_mean = float(v[i, j])
                            break

        rt_i, ch_i = simulate_mixture_ddm_trial(
            v_mean=v_mean,
            a_i=float(a[i]),
            z=z,
            tau=tau,
            s_v=s_v,
            decay=decay,
            s_z=s_z,
            s_tau=s_tau,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        rts[i] = rt_i
        choices[i] = ch_i

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
