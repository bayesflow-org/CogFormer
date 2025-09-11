import numpy as np
from simulators import Model
from simulators.benchmarks.ddms.ddm import simulate_collapsing_bound_ddm


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
        v_intercept = float(params["v_intercept"])
        v_slope = float(params["v_slope"])
        a_intercept = float(params["a_intercept"])
        a_slope = float(params["a_slope"])

        # per-trial arrays via regression
        v = v_intercept + v_slope * x_v
        a = a_intercept + a_slope * x_a
        a = np.maximum(a, 1e-6).astype(np.float32)

        # scalars
        tau = max(float(params["tau"]), 0.0)
        s_tau = max(float(params["s_tau"]), 0.0)
        s_v = max(float(params["s_v"]), 0.0)
        decay = max(float(params["decay"]), 0.0)

        # run collapsing-bound ddm (vectorized driver)
        results = simulate_collapsing_bound_ddm(
            v=v.astype(np.float32, copy=False),
            a=a.astype(np.float32, copy=False),
            tau=np.float32(tau),
            s_tau=np.float32(s_tau),
            s_v=np.float32(s_v),
            decay=np.float32(decay),
            zr=np.float32(0.5),
            sigma=np.float32(1.0),
            dt=np.float32(0.001),
            max_steps=10000,
        )

        rts = results[:, 0]
        choices = results[:, 1]
        return {"rts": rts, "choices": choices} # shape (N, 2) → [RT, choice]
