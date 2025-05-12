import numpy as np

from BayesGPT.simulators.ensemble_simulator import EnsembleSimulator


class DDMEnsemble(EnsembleSimulator):
    """
    An ensemble of Drift Diffusion Model (DDM) variants for simulating simple decision-making tasks.

    This class wraps several DDM variants differing in their complexity:

    Variants
    --------
    - 'basic': Basic DDM with drift and fixed boundary.
    - 'with_ndt': Adds non-decision time to the basic DDM.
    - 'trajectory': Tracks the trajectory of evidence accumulation.
    - 'collapsing_bound': Implements a time-collapsing boundary.
    """

    def __init__(self):
        """
        Initializes the DDM ensemble by adding each variant to the simulator registry.
        """
        super().__init__()
        self._add_variants()

    def _add_variants(self):
        """
        Defines and registers multiple DDM variants with different parameter sets.
        Each variant is added with a name and its required input variables.
        """

        def ddm_basic(drift, boundary):
            """
            Basic DDM with a fixed decision boundary.

            Parameters
            ----------
            drift : float
                Drift rate of evidence accumulation.
            boundary : float
                Decision threshold.

            Returns
            -------
            dict
                A dictionary with keys 'RT' (reaction time) and 'choice' (0 or 1).
            """

            rt = np.abs(boundary / drift) + np.random.normal(0, 0.1)
            choice = int(drift > 0)
            return {"RT": rt, "choice": choice}

        def ddm_with_ndt(drift, boundary, ndt):
            """
            DDM with added non-decision time (e.g., encoding and motor delay).

            Parameters
            ----------
            drift : float
                Drift rate.
            boundary : float
                Decision boundary.
            ndt : float
                Non-decision time added to RT.

            Returns
            -------
            dict
                Dictionary with keys 'RT', 'choice', and 'ndt'.
            """

            rt = np.abs(boundary / drift) + ndt + np.random.normal(0, 0.1)
            choice = int(drift > 0)
            return {"RT": rt, "choice": choice, "ndt": ndt}

        def ddm_with_trajectory(drift, boundary, ndt):
            """
            DDM variant that returns the evidence trajectory over time.

            Parameters
            ----------
            drift : float
                Drift rate.
            boundary : float
                Decision boundary.
            ndt : float
                Non-decision time.

            Returns
            -------
            dict
                Contains 'RT', 'choice', and 'trajectory' (list of positions over time).
            """
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
            """
            DDM with a collapsing boundary that shrinks linearly over time.

            Parameters
            ----------
            drift : float
                Drift rate.
            initial_boundary : float
                The initial decision threshold at time 0.
            ndt : float
                Non-decision time.
            collapse_rate : float
                Rate at which the boundary decreases per unit time.

            Returns
            -------
            dict
                Contains 'RT', 'choice', 'trajectory', and 'final_bound'.
            """
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
            ddm_with_trajectory,
            variable_names=["drift", "boundary", "ndt"],
            simulator_name="trajectory",
        )
        self.add(
            ddm_collapsing_bound,
            variable_names=["drift", "initial_boundary", "ndt", "collapse_rate"],
            simulator_name="collapsing_bound",
        )
