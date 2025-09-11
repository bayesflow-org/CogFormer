import numpy as np
from numba import njit, prange


@njit
def simulate_collapsing_bound_ddm_trial(
    v: float,
    a: float,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    zr: float = 0.5,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> (float, float):
    """
    Single trial with collapsing bounds; v_mean and a_i are already per-trial scalars.
    """
    v_i = np.random.normal(v, s_v)
    tau_i = tau
    if s_tau > 0.0:
        tau_i = np.random.normal(tau, s_tau)
        if tau_i < 0.0:
            tau_i = 0.0

    # initialize
    a0 = a if a > 1e-6 else 1e-6
    d  = decay if decay > 0.0 else 0.0

    x = zr * a0
    t = tau_i

    for _ in range(max_steps):
        t += dt
        bound = a0 * np.exp(-d * t)
        if bound < 1e-3:
            bound = 1e-3
        x += v_i * dt + sigma * np.sqrt(dt) * np.random.normal()
        if x >= bound:
            return t, 1.0
        if x <= -bound:
            return t, 0.0
    return -1.0, -1.0

@njit(parallel=True)
def simulate_collapsing_bound_ddm(
    v_intercept: float,
    v_slope: float,
    a_intercept: float,
    a_slope: float,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    x_v: np.ndarray,   # shape (N,)
    x_a: np.ndarray,   # shape (N,)
    zr: float = 0.5,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    """
    Collapsing-bound DDM with **regression-style** v and a:
        v_i = v_intercept + v_slope * x_v[i]
        a_i = a_intercept + a_slope * x_a[i]
    Masking is naturally supported by passing zeros for masked coefficients.
    """
    n = x_v.shape[0]
    out = np.zeros((n, 2), dtype=np.float32)
    rts, choices = out[:, 0], out[:, 1]

    for i in prange(n):
        # Per-trial drift and bound from coefficients + covariates
        v_i = v_intercept + v_slope * float(x_v[i])
        a_i = a_intercept + a_slope * float(x_a[i])
        if a_i < 1e-6:
            a_i = 1e-6  # keep positive bound

        rt_i, ch_i = simulate_collapsing_bound_ddm_trial(
            v=v_i,
            a=a_i,
            tau=tau,
            s_tau=s_tau,
            s_v=s_v,
            decay=decay,
            zr=zr,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        rts[i] = rt_i
        choices[i] = ch_i

    return out
