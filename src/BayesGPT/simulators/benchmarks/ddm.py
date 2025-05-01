import numpy as np

from BayesGPT.simulators.simulator_family import SimulatorFamily


class DDMFamily(SimulatorFamily):
    def __init__(self):
        super().__init__()
        self._add_variants()

    def _add_variants(self):
        self.add(
            lambda drift, boundary: {
                "RT": np.abs(boundary / drift) + np.random.normal(0, 0.1),
                "choice": int(drift > 0),
            },
            variable_names=["v", "a"],
            simulator_name="ddm_basic",
        )

        self.add(
            lambda drift, boundary, ndt: {
                "RT": np.abs(boundary / drift) + ndt + np.random.normal(0, 0.1),
                "choice": int(drift > 0),
                "ndt": ndt,
            },
            variable_names=["v", "a", "t"],
            simulator_name="ddm_with_ndt",
        )

        def ddm_collapsing_bound(drift, initial_boundary, ndt, collapse_rate):
            dt = 0.01
            x = 0
            trajectory = [x]
            current_boundary = initial_boundary
            for t in range(100):
                current_boundary = max(0.1, initial_boundary - collapse_rate * t * dt)
                x += drift * dt + np.random.normal(0, 0.1)
                trajectory.append(x)
                if np.abs(x) >= current_boundary:
                    break
            rt = len(trajectory) * dt + ndt
            return {
                "RT": rt,
                "choice": int(x > 0),
                "trajectory": trajectory,
                "final_bound": current_boundary,
            }

        self.add(
            ddm_collapsing_bound,
            variable_names=["v", "a", "t", "b"],
            simulator_name="ddm_collapsing_bound",
        )
