import numpy as np
from ..model import Model


class SuperDDM(Model):
    """
    Drift Diffusion Model (DDM) with full parameter variability, optional
    linearly collapsing boundaries, and support for multiple drift rates.

    This model extends a standard DDM by allowing across‑trial variability in
    *all* core parameters. Variability is modeled as Gaussian noise around a
    base (mean) value for each trial.

    The model also supports optional *multiple drifts* either as an
    across‑trial mixture (choose one component per trial) or as a
    within‑trial piecewise‑constant schedule (switch drifts during the
    trial at specified times).

    State evolution (Euler–Maruyama discretization)
    ----------------------------------------------
    The latent decision variable evolves as:

        x_{t+dt} = x_t + v(t or i) * dt + sigma_i * sqrt(dt) * N(0, 1)

    where `v(t or i)` is either the per‑trial drift, a per‑trial mixture
    component, or the current segment’s drift in a within‑trial schedule.
    Trials start at a symmetric position:

        x0 = (2*z_i - 1) * a_i

    Boundaries
    ----------
    Symmetric boundaries may collapse linearly over time:

        B_i(t) = max(a_i - angle_i * t, eps)

    A trial ends when `x_t` crosses `+B_i(t)` (choice = 1) or `-B_i(t)`
    (choice = 0). The reported reaction time (RT) is the decision time plus
    the non‑decision component `tau_i`.

    Parameters
    ----------
    dt : float, optional
        Time step for numerical integration. Default is 0.001.
    max_steps : int, optional
        Maximum number of Euler steps per trial. Default is 10000.

    Notes
    -----
    Required base parameters in ``params`` (scalars):
        v : float
            Mean drift rate.
        a : float
            Boundary separation (must be > 0).
        z : float
            Starting point fraction in (0, 1).
        tau or t : float
            Non‑decision time (must be >= 0).
        sigma : float
            Diffusion noise scale (must be > 0).

    Optional base parameter in ``params``:
        angle : float, default 0
            Linear collapse slope for the boundaries. Positive values produce
            narrowing bounds; zero recovers fixed bounds.

    Across‑trial standard deviations in ``params`` (all optional, default 0):
        s_v or eta : float
            SD for drift rate across trials.
        s_z or sz : float
            SD for starting point across trials.
        s_tau or st : float
            SD for non‑decision time across trials.
        s_a : float
            SD for boundary separation across trials.
        s_sigma : float
            SD for diffusion noise across trials.
        s_angle : float
            SD for collapse slope across trials.

    Multiple‑drift options in ``params`` (all optional)
    --------------------------------------------------
    Across‑trial mixture (choose a single component per trial):
        v_components : array_like, shape (K,)
            Drift components.
        p_components : array_like, shape (K,), optional
            Mixture weights (non‑negative, sum to 1). If omitted, uses a
            uniform mixture.

    Within‑trial piecewise‑constant schedule:
        v_schedule : array_like, shape (K,)
            Drifts for K sequential segments.
        t_schedule : array_like, shape (K,)
            Durations (in seconds) for each segment. Must be > 0.

    Precedence (if multiple options supplied):
        1) If both ``v_schedule`` and ``t_schedule`` are provided, use the
           within‑trial schedule (piecewise‑constant drift).
        2) Else, if ``v_components`` is provided, use the across‑trial mixture.
        3) Else, use the scalar drift ``v`` with across‑trial SD ``s_v``.

    Implementation notes
    --------------------
    * All across‑trial SDs are applied via Gaussian draws and then clipped
      where necessary to maintain valid domains (e.g., z in (0, 1), sigma > 0).
    * If a trial does not hit a bound within ``max_steps``, the row is
      returned as ``[nan, nan]``.
    """

    def __init__(self, dt=0.001, max_steps=10000, eps=1e-6):
        """
        Construct a SuperDDM simulator.

        Parameters
        ----------
        dt : float, optional
            Time step for numerical integration. Default is 0.001.
        max_steps : int, optional
            Maximum number of Euler steps per trial. Default is 10000.
        eps : float, optional
            Numerical tolerance for computing the
        """
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.eps = eps


    def _param(self, d, key, default=None, aliases=()):
        """
        Fetch a parameter from a dict with optional aliases and default.

        Parameters
        ----------
        d : dict
            Source dictionary.
        key : str
            Primary key to look up.
        default : object, optional
            Value to return if neither ``key`` nor any alias exists.
        aliases : tuple or list of str, optional
            Alternative keys to try (in order).

        Returns
        -------
        object
            Found value or the provided default.

        Raises
        ------
        KeyError
            If none of the keys are present and ``default`` is None.
        """
        if key in d:
            return d[key]
        for a in aliases:
            if a in d:
                return d[a]
        if default is None:
            raise KeyError(f"Missing required parameter '{key}'")
        return default


    def _draw_scalar_or_array(self, mean, std, size, clip_min=None, clip_max=None):
        """
        Draw an array with optional Gaussian noise and clipping.

        Parameters
        ----------
        mean : float
            Base value.
        std : float
            Across‑trial standard deviation. If 0, returns a constant array.
        size : int
            Number of draws (trials).
        clip_min : float, optional
            Minimum allowed value. If provided, values are clipped below this.
        clip_max : float, optional
            Maximum allowed value. If provided, values are clipped above this.

        Returns
        -------
        np.ndarray, shape (size,)
            Array of draws as float64.
        """
        if std > 0:
            x = np.random.normal(mean, std, size=size)
        else:
            x = np.full(size, mean)
        if clip_min is not None:
            x = np.maximum(clip_min, x)
        if clip_max is not None:
            x = np.minimum(clip_max, x)
        return x.astype(np.float64)


    def simulate(self, params, batch_size):
        """
        Simulate multiple trials of the SuperDDM.

        Parameters
        ----------
        params : dict
            Dictionary of parameter values. See the class docstring for the
            full list of supported keys, aliases, and meanings.
        batch_size : int
            Number of trials to simulate.

        Returns
        -------
        np.ndarray, shape (batch_size, 2)
            Columns are ``[RT, choice]``.
            * RT (float): reaction time in seconds. May be ``nan`` if the trial
              did not terminate within ``max_steps``.
            * choice (float): 1.0 for upper bound, 0.0 for lower bound, or
              ``nan`` if the trial did not terminate.
        """
        B = int(batch_size)
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        eps = self.eps

        # Base means (support aliases)
        v_mu = float(self._param(params, "v", default=0.0))
        a_mu = float(self._param(params, "a"))
        z_mu = float(self._param(params, "z"))
        tau_mu = float(self._param(params, "tau", aliases=("t",)))
        sigma_mu = float(self._param(params, "sigma"))
        angle_mu = float(self._param(params, "angle", default=0.0))

        # Across‑trial SDs (support common aliases)
        s_v = float(self._param(params, "s_v",
                                default=self._param(params, "eta", default=0.0)))
        s_z = float(self._param(params, "s_z",
                                default=self._param(params, "sz", default=0.0)))
        s_tau = float(self._param(params, "s_tau",
                                  default=self._param(params, "st", default=0.0)))
        s_a = float(self._param(params, "s_a", default=0.0))
        s_sigma = float(self._param(params, "s_sigma", default=0.0))
        s_angle = float(self._param(params, "s_angle", default=0.0))

        # Configure multiple drift rates
        v_components = params.get("v_components", None)
        p_components = params.get("p_components", None)
        v_schedule = params.get("v_schedule", None)
        t_schedule = params.get("t_schedule", None)

        # Draw per-trial parameters
        z = self._draw_scalar_or_array(
            z_mu, s_z, B, clip_min=eps, clip_max=1.0 - eps
        )
        tau = self._draw_scalar_or_array(tau_mu, s_tau, B, clip_min=eps)
        a = self._draw_scalar_or_array(a_mu, s_a, B, clip_min=eps)
        sigma = self._draw_scalar_or_array(sigma_mu, s_sigma, B, clip_min=eps)
        angle = self._draw_scalar_or_array(angle_mu, s_angle, B)

        # Output buffers
        rts = np.empty(B, dtype=np.float64)
        choices = np.empty(B, dtype=np.float64)

        # Within‑trial schedule (takes precedence)
        if v_schedule is not None and t_schedule is not None:
            v_schedule = np.asarray(v_schedule, dtype=float).ravel()
            t_schedule = np.asarray(t_schedule, dtype=float).ravel()

            if v_schedule.size != t_schedule.size:
                raise ValueError(
                    "v_schedule and t_schedule must have the same length."
                )
            if np.any(t_schedule <= 0):
                raise ValueError("All entries of t_schedule must be > 0.")

            K = v_schedule.size
            # Optional across‑trial jitter around each segment’s drift via s_v
            if s_v > 0:
                Vseg = np.random.normal(
                    loc=v_schedule[None, :], scale=s_v, size=(B, K)
                ).astype(np.float64)
            else:
                Vseg = np.tile(v_schedule[None, :], (B, 1)).astype(np.float64)

            seg_switch = np.cumsum(t_schedule.astype(np.float64))

            for i in range(B):
                x = (2.0 * z[i] - 1.0) * a[i]
                t_acc = 0.0
                seg_idx = 0
                next_switch = seg_switch[seg_idx]

                for _ in range(self.max_steps):
                    t_acc += dt

                    # Collapsing symmetric bounds
                    bound = a[i] - angle[i] * t_acc
                    if bound < eps:
                        bound = eps

                    # Segment switch check
                    if seg_idx < K - 1 and t_acc > (next_switch - eps):
                        seg_idx += 1
                        next_switch = seg_switch[seg_idx]

                    # Euler–Maruyama step
                    v_now = Vseg[i, seg_idx]
                    x += v_now * dt + sigma[i] * sqrt_dt * np.random.randn()

                    if x >= bound:
                        rts[i] = t_acc + tau[i]
                        choices[i] = 1.0
                        break
                    if x <= -bound:
                        rts[i] = t_acc + tau[i]
                        choices[i] = 0.0
                        break
                else:
                    rts[i] = np.nan
                    choices[i] = np.nan

            return np.stack([rts, choices], axis=1)


        # Across‑trial mixture of drifts
        if v_components is not None:
            vc = np.asarray(v_components, dtype=float).ravel()
            if vc.size == 0:
                raise ValueError("v_components must be a non‑empty array.")

            if p_components is None:
                pk = np.full(vc.size, 1.0 / vc.size, dtype=float)
            else:
                pk = np.asarray(p_components, dtype=float).ravel()
                if pk.size != vc.size:
                    raise ValueError(
                        "p_components must have the same length as v_components."
                    )
                if np.any(pk < 0):
                    raise ValueError("p_components must be non‑negative.")
                s = pk.sum()
                if s <= 0:
                    raise ValueError("Sum of p_components must be > 0.")
                pk = pk / s

            comps = np.random.choice(vc.size, size=B, p=pk)
            if s_v > 0:
                v = np.random.normal(vc[comps], s_v, size=B)
            else:
                v = vc[comps]
            v = v.astype(np.float64)

        else: # Scalar drift with across‑trial SD
            v = self._draw_scalar_or_array(v_mu, s_v, B)

        # Simulate each trial
        for i in range(B):
            x = (2.0 * z[i] - 1.0) * a[i]
            t_acc = 0.0

            for _ in range(self.max_steps):
                t_acc += dt

                bound = a[i] - angle[i] * t_acc
                if bound < eps:
                    bound = eps

                x += v[i] * dt + sigma[i] * sqrt_dt * np.random.randn()

                if x >= bound:
                    rts[i] = t_acc + tau[i]
                    choices[i] = 1.0
                    break
                if x <= -bound:
                    rts[i] = t_acc + tau[i]
                    choices[i] = 0.0
                    break
            else:
                rts[i] = np.nan
                choices[i] = np.nan

        return np.stack([rts, choices], axis=1)
