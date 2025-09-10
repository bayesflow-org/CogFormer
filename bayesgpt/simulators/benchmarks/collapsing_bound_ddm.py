import numpy as np
from typing import Optional, Iterable
from simulators import Model, ContextManager
from .ddm import simulate_collapsing_bound_ddm

class CollapsingBoundDDM(Model):
    def __init__(self, context_manager: ContextManager, dt: float = 0.001, max_steps: int = 10000):
        # Initialize with context manager and simulation parameters
        self.context_manager = context_manager
        self.dt = dt
        self.max_steps = max_steps

    def simulate(self, params: dict[str, np.ndarray | float], num_samples: int = 1) -> dict[str, np.ndarray]:

        # Validate parameters
        # self._validate_parameters(regressed_params, num_samples)


        # Simulate using regressed and scalar parameters
        result = simulate_collapsing_bound_ddm(
            v=params["v"],
            a=params["a"],
            zr=params["zr"],
            tau=params["tau"],
            s_v=params["s_v"],
            decay=params["decay"],
            s_tau=params["s_tau"],
            sigma=params["sigma"],
            dt=self.dt,
            max_steps=self.max_steps
        )

        # Construct output
        return {
            "rts": result[:, 0],
            "choices": result[:, 1]
        }

    # def _validate_parameters(self, params: Dict[str, np.ndarray], num_samples: int) -> None:
    #     # Check required parameters
    #     required_params = ["v", "a", "zr", "tau", "s_v", "decay", "s_tau", "sigma"]
    #     missing = set(required_params) - set(params)
    #     if missing:
    #         raise ValueError(f"Missing parameters: {missing}")
    #
    #     # Validate shapes
    #     for key in params:
    #         if params[key].shape != (num_samples,):
    #             raise ValueError(f"{key} must have shape ({num_samples},), got {params[key].shape}")
    #
    #     # Enforce constraints
    #     if np.any(params["a"] <= 0) or np.any(params["sigma"] <= 0):
    #         raise ValueError("a, sigma must be > 0")
    #     if np.any(params["zr"] <= 0) or np.any(params["zr"] >= 1):
    #         raise ValueError("0 < zr < 1")
    #     if np.any(params["s_v"] < 0) or np.any(params["decay"] < 0) or np.any(params["s_tau"] < 0):
    #         raise ValueError("s_v, decay, s_tau must be >= 0")

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