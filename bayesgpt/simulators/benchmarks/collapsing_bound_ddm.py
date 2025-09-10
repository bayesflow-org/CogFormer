import numpy as np
from typing import Optional, Iterable
from simulators import Model
from .ddm import simulate_collapsing_bound_ddm

class CollapsingBoundDDM(Model):
    def __init__(self, dt: float = 0.001, max_steps: int = 10000):
        # Initialize with context manager and simulation parameters
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, np.ndarray | float], num_samples: int = 1) -> dict[str, np.ndarray]:

        v = params['v']
        a= params['a']
        zr = params['zr']
        tau = params['tau']
        s_v = params['s_v']
        s_tau = params['s_tau']
        decay = params['decay']
        sigma = params['sigma']

        # Simulate using regressed and scalar parameters
        result = simulate_collapsing_bound_ddm(
            v=v,
            a=a,
            zr=zr,
            tau=tau,
            s_v=s_v,
            s_tau=s_tau,
            decay=decay,
            sigma=sigma,
            dt=self.dt,
            max_steps=self.max_steps
        )

        # Construct output
        return {
            "rts": result[:, 0],
            "choices": result[:, 1]
        }

    @staticmethod
    def summarize(
        outputs: dict[str, np.ndarray],
        quantile_levels: Iterable[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        by_choice: bool = True,
        tau: Optional[np.ndarray] = None
    ) -> dict[str, np.ndarray]:
        # Summarize response times with quantiles
        rts = outputs["rts"].astype(np.float32, copy=False)
        choices = outputs["choices"].astype(np.float32, copy=False)
        q = np.asarray(tuple(quantile_levels), dtype=np.float32)
        num_levels = q.size

        # Valid RT mask (exclude non-terminations)
        valid = ~np.isnan(rts)
        n = rts.size
        n_valid = int(valid.sum())
        invalid_rate = np.float32(1.0 - (n_valid / n if n > 0 else np.nan))

        def _q(xmask: np.ndarray, x: np.ndarray) -> np.ndarray:
            if not np.any(xmask):
                return np.full((num_levels,), np.nan, dtype=np.float32)
            return np.quantile(x[xmask], q, method="linear").astype(np.float32)

        # RT quantiles
        rt_quantiles = _q(valid, rts)

        # RT by choice
        rt_quantiles_by_choice = np.vstack([
            _q(valid & (choices == 0), rts),
            _q(valid & (choices == 1), rts)
        ]) if by_choice else np.full((2, num_levels), np.nan, dtype=np.float32)

        out = {
            "invalid_rate": invalid_rate,
            "rt_quantiles": rt_quantiles,
            "rt_quantiles_by_choice": rt_quantiles_by_choice
        }

        # Decision time quantiles if tau provided
        if tau is not None:
            tau = tau.astype(np.float32, copy=False)
            dt = np.clip(rts - tau, 0.0, None)
            valid_dt = valid & ~np.isnan(tau)
            out["dt_quantiles"] = _q(valid_dt, dt)
            if by_choice:
                out["dt_quantiles_by_choice"] = np.vstack([
                    _q(valid_dt & (choices == 0), dt),
                    _q(valid_dt & (choices == 1), dt)
                ])

        return out
