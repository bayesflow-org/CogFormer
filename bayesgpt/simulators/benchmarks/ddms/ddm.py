import numpy as np
from numba import njit, prange
from simulators import Model


@njit
def simulate_ddm_trial(
    v: float,
    a: float,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    z: float = 0.5,
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

    x = z * a0
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
def simulate_ddm(
    v: np.ndarray,
    a: np.ndarray,
    tau: float,
    s_tau: float,
    s_v: float,
    decay: float,
    z: float = 0.5,
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
    n = v.shape[0]
    out = np.zeros((n, 2), dtype=np.float32)
    rts, choices = out[:, 0], out[:, 1]

    for i in prange(n):
        rt_i, ch_i = simulate_ddm_trial(
            v=v[i],
            a=a[i],
            tau=tau,
            s_tau=s_tau,
            s_v=s_v,
            decay=decay,
            z=z,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        rts[i] = rt_i
        choices[i] = ch_i

    return out

@njit
def sample_ddm_prior():
    v_intercept = np.random.gamma(3.0, 0.8)
    v_slope     = np.random.normal(0.0, 3.0)
    s_v         = np.random.gamma(1.0, 0.2)
    a_intercept = np.random.gamma(10.0, 0.3)
    a_slope     = np.random.normal(0.0, 1.0)
    decay       = np.random.gamma(1.0, 0.4)
    tau         = np.random.gamma(3.0, 0.2)
    s_tau       = np.random.uniform(0.0, tau * 2.0)
    return np.array([v_intercept, v_slope, s_v, a_intercept, a_slope, decay, tau, s_tau], dtype=np.float32)

class DDM(Model):
    def prepare_params(self, params: dict, num_samples: int):
        # no-op; masking already applied upstream
        return params

    def simulate(self, params: dict, num_samples: int, context=None):
        if context is None or "x_v" not in context or "x_a" not in context:
            raise ValueError("context must be a dict with 'x_v' and 'x_a' of shape (num_samples,)")

        x_v = np.asarray(context["x_v"], dtype=np.float32).reshape(-1)
        x_a = np.asarray(context["x_a"], dtype=np.float32).reshape(-1)
        if x_v.shape[0] != num_samples or x_a.shape[0] != num_samples:
            raise ValueError("x_v and x_a must have length == num_samples")

        # coefficients (masked entries may be 0.0)
        v_intercept = params["v_intercept"]
        v_slope = params["v_slope"]
        a_intercept = params["a_intercept"]
        a_slope = params["a_slope"]

        # per-trial arrays via regression
        v = v_intercept + v_slope * x_v
        a = a_intercept + a_slope * x_a
        a = np.maximum(a, 1e-6).astype(np.float32)

        # scalars
        tau = max(params["tau"], 0.0)
        s_tau = max(params["s_tau"], 0.0)
        s_v = max(params["s_v"], 0.0)
        decay = max(params["decay"], 0.0)

        # run collapsing-bound ddm (vectorized driver)
        results = simulate_ddm(
            v=v.astype(np.float32, copy=False),
            a=a.astype(np.float32, copy=False),
            tau=tau,
            s_tau=s_tau,
            s_v=s_v,
            decay=decay,
            z=0.5,
            sigma=1.0,
            dt=0.001,
            max_steps=10000,
        )

        rts = results[:, 0]
        choices = results[:, 1]
        return {"rts": rts, "choices": choices}
