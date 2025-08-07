import numpy as np
from BayesGPT.deprecated.lightweight_ensemble_simulator import (
    LightweightEnsembleSimulator,
)


class LightweightDDMEnsemble(LightweightEnsembleSimulator):
    """
    Ensemble of Drift Diffusion Model (DDM) variants for simulating decision-making.

    This class defines multiple DDM variants with differing parameterization:

    - 'basic': Basic DDM with drift and boundary.
    - 'with_ndt': Adds non-decision time (NDT).
    - 'trajectory': Tracks the trajectory of evidence accumulation.
    - 'collapsing_bound': Models a boundary that collapses over time.
    """

    def __init__(self):
        """
        Initializes the DDM ensemble by registering each variant.
        """
        super().__init__()
        self._add_variants()

    def _add_variants(self):
        """
        Defines and registers each DDM variant simulator with its required parameters.
        """

        def ddm_basic(drift, boundary):
            """
            Basic DDM with constant decision boundary.

            Parameters
            ----------
            drift : float
                The drift rate of evidence accumulation.
            boundary : float
                The static decision threshold.

            Returns
            -------
            dict
                Contains 'RT' (reaction time) and 'choice' (binary decision).
            """
            rt = np.abs(boundary / drift) + np.random.normal(0, 0.1)
            choice = int(drift > 0)
            return {"RT": rt, "choice": choice}

        def ddm_with_ndt(drift, boundary, ndt):
            """
            DDM that includes a non-decision time component.

            Parameters
            ----------
            drift : float
                Drift rate.
            boundary : float
                Decision threshold.
            ndt : float
                Non-decision time added to the RT.

            Returns
            -------
            dict
                Contains 'RT', 'choice', and 'ndt'.
            """
            rt = np.abs(boundary / drift) + ndt + np.random.normal(0, 0.1)
            choice = int(drift > 0)
            return {"RT": rt, "choice": choice, "ndt": ndt}

        def ddm_with_trajectory(drift, boundary, ndt):
            """
            DDM that returns the trajectory of evidence accumulation.

            Parameters
            ----------
            drift : float
                Drift rate.
            boundary : float
                Decision threshold.
            ndt : float
                Non-decision time.

            Returns
            -------
            dict
                Contains 'RT', 'choice', and 'trajectory' (list of evidence values over time).
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
            DDM with a boundary that collapses linearly over time.

            Parameters
            ----------
            drift : float
                Drift rate.
            initial_boundary : float
                The initial decision threshold at time zero.
            ndt : float
                Non-decision time.
            collapse_rate : float
                Rate at which the boundary collapses per time step.

            Returns
            -------
            dict
                Contains 'RT', 'choice', 'trajectory', and 'final_bound' (value of boundary at decision).
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
