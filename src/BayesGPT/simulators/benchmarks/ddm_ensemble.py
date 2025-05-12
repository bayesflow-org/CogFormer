import numpy as np

from BayesGPT.simulators.ensemble_simulator import EnsembleSimulator


class DDMEnsemble(EnsembleSimulator):
    def __init__(self):
        super().__init__()
        self._add_variants()

    def _add_variants(self):
        def ddm_basic(drift, boundary):
            rt = np.abs(boundary / drift) + np.random.normal(0, 0.1)
            choice = int(drift > 0)
            return {"RT": rt, "choice": choice}

        def ddm_with_ndt(drift, boundary, ndt):
            rt = np.abs(boundary / drift) + ndt + np.random.normal(0, 0.1)
            choice = int(drift > 0)
            return {"RT": rt, "choice": choice, "ndt": ndt}

        def ddm_with_traj(drift, boundary, ndt):
            dt = 0.01
            x = 0
            trajectory = [x]
            for _ in range(100):
                x += drift * dt + np.random.normal(0, 0.1)
                trajectory.append(x)
                if np.abs(x) >= boundary:
                    break
            rt = len(trajectory) * dt + ndt
            return {"RT": rt, "choice": int(x > 0), "trajectory": trajectory}

        def ddm_collapsing_bound(drift, initial_boundary, ndt, collapse_rate):
            dt = 0.01
            x = 0
            trajectory = [x]
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
            ddm_basic, variable_names=["drift", "boundary"], simulator_name="basic"
        )
        self.add(
            ddm_with_ndt,
            variable_names=["drift", "boundary", "ndt"],
            simulator_name="with_ndt",
        )
        self.add(
            ddm_with_traj,
            variable_names=["drift", "boundary", "ndt"],
            simulator_name="trajectory",
        )
        self.add(
            ddm_collapsing_bound,
            variable_names=["drift", "initial_boundary", "ndt", "collapse_rate"],
            simulator_name="collapsing_bound",
        )
