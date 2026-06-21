"""High-level entry point for amortized estimation with a trained CogFormer.

``estimate`` is the one call that turns a *design configuration* into
*diagnostics-ready results*. Given a model name and a design config it:

1. resolves and loads the matching trained checkpoint,
2. **simulates ground truth** — draws ``n_datasets`` parameter sets and
   synthetic datasets under the chosen design (no user data needed),
3. runs the trained network to draw an amortized posterior per dataset,
4. computes the requested diagnostics (recovery / coverage / ECDF).

The name mirrors BayesFlow's ``BasicWorkflow.estimate`` so it reads naturally
to the same audience. Note: it returns *posterior draws*, not a point estimate.

This is the keystone of the interactive web demo: ``EstimateResult.to_dict()``
is exactly the JSON payload the frontend renders, so the Python API and the HTTP
contract never drift. See ``EstimateResult.to_dict`` for the wire format.

Heavy dependencies (torch, the simulators) are imported lazily inside the
functions so ``import cogformer`` stays cheap.

----------------------------------------------------------------------------
JSON CONTRACT (the ``/estimate`` response; what the frontend consumes)
----------------------------------------------------------------------------
{
  "model": "ddm",
  "design_config": {"1": ["v","a","z","tau"], "u_1": ["v","a"]},
  "param_labels": ["v", "a", "z", "tau", "v|u_1", "a|u_1"],
  "settings": {"n_datasets": 100, "num_obs": 500, "num_samples": 100,
               "steps": 50, "seed": 0},
  "elapsed_s": 1.84,
  "diagnostics": {
    "recovery": {
      "<param>": {"true": [...], "mean": [...], "q_lo": [...], "q_hi": [...],
                  "r2": 0.97}          # per dataset; q_* = central CI for error bars
    },
    "coverage": {
      "<param>": {"nominal": [...], "empirical": [...]}   # calibration curve
    },
    "ecdf": {
      "<param>": {"x": [...], "y": [...], "band_lo": [...], "band_hi": [...]}
    }                                  # SBC rank ECDF + simultaneous band
  },
  # present only when return_samples=True (large; off by default for the demo):
  "posterior_samples": [[[...]]],      # (n_datasets, num_samples, n_params)
  "true_params": [[...]]               # (n_datasets, n_params)
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # avoid importing numpy at module import time for type hints only
    import numpy as np

__all__ = [
    "estimate",
    "EstimateResult",
    "RecoveryDiagnostic",
    "CoverageDiagnostic",
    "EcdfDiagnostic",
    "DEFAULT_DIAGNOSTICS",
    "SUPPORTED_MODELS",
]

# Models the demo exposes. Intrinsic-parameter lists are the source of truth for
# parameter ordering/labels and are ported from FAMILY_REGISTRY in
# experiments/model_family/family_cf_validate.py. ddm/rdm confirmed; cdm and the
# joint model_class are TODO (port from the same registry to avoid inventing names).
SUPPORTED_MODELS: dict[str, dict] = {
    "ddm": {"intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"]},
    "rdm": {"intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"]},
    "cdm": {"intrinsics": None},          # TODO: port from FAMILY_REGISTRY
    "model_class": {"intrinsics": None},  # TODO: joint multi-family model
}

DEFAULT_DIAGNOSTICS: tuple[str, ...] = ("recovery", "coverage", "ecdf")


# ---------------------------------------------------------------------------
# Result types — final/stable. These define the JSON contract via to_dict().
# ---------------------------------------------------------------------------
@dataclass
class RecoveryDiagnostic:
    """Per-parameter parameter-recovery data (true vs. posterior, across datasets)."""

    true: list[float]      # ground-truth value per dataset
    mean: list[float]      # posterior mean per dataset
    q_lo: list[float]      # lower bound of central credible interval (error bars)
    q_hi: list[float]      # upper bound of central credible interval
    r2: float              # coefficient of determination (true vs. mean)

    def to_dict(self) -> dict:
        return {"true": self.true, "mean": self.mean, "q_lo": self.q_lo,
                "q_hi": self.q_hi, "r2": self.r2}


@dataclass
class CoverageDiagnostic:
    """Per-parameter calibration curve: empirical vs. nominal CI coverage."""

    nominal: list[float]    # nominal central-interval levels in [0, 1]
    empirical: list[float]  # fraction of datasets whose true value falls inside

    def to_dict(self) -> dict:
        return {"nominal": self.nominal, "empirical": self.empirical}


@dataclass
class EcdfDiagnostic:
    """Per-parameter SBC rank ECDF with a simultaneous confidence band."""

    x: list[float]        # evaluation points (fractional ranks) in [0, 1]
    y: list[float]        # empirical CDF of fractional ranks
    band_lo: list[float]  # lower simultaneous confidence band (under uniformity)
    band_hi: list[float]  # upper simultaneous confidence band

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "band_lo": self.band_lo, "band_hi": self.band_hi}


@dataclass
class EstimateResult:
    """Everything ``estimate`` produces. ``to_dict()`` is the ``/estimate`` payload."""

    model: str
    design_config: dict[str, list[str]]
    param_labels: list[str]
    settings: dict
    elapsed_s: float
    recovery: dict[str, RecoveryDiagnostic] = field(default_factory=dict)
    coverage: dict[str, CoverageDiagnostic] = field(default_factory=dict)
    ecdf: dict[str, EcdfDiagnostic] = field(default_factory=dict)
    # Raw arrays, included only when estimate(return_samples=True).
    posterior_samples: "np.ndarray | None" = None  # (n_datasets, num_samples, n_params)
    true_params: "np.ndarray | None" = None         # (n_datasets, n_params)

    def to_dict(self, *, include_samples: bool | None = None) -> dict:
        """Serialize to the JSON contract.

        ``include_samples`` defaults to whether raw samples were retained. The
        demo keeps it ``False`` (diagnostics only) to keep payloads small.
        """
        diagnostics: dict[str, dict] = {}
        if self.recovery:
            diagnostics["recovery"] = {k: v.to_dict() for k, v in self.recovery.items()}
        if self.coverage:
            diagnostics["coverage"] = {k: v.to_dict() for k, v in self.coverage.items()}
        if self.ecdf:
            diagnostics["ecdf"] = {k: v.to_dict() for k, v in self.ecdf.items()}

        payload: dict = {
            "model": self.model,
            "design_config": self.design_config,
            "param_labels": self.param_labels,
            "settings": self.settings,
            "elapsed_s": self.elapsed_s,
            "diagnostics": diagnostics,
        }

        want_samples = include_samples if include_samples is not None else (
            self.posterior_samples is not None
        )
        if want_samples and self.posterior_samples is not None:
            payload["posterior_samples"] = self.posterior_samples.tolist()
            if self.true_params is not None:
                payload["true_params"] = self.true_params.tolist()
        return payload


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def estimate(
    model: str,
    design_config: dict[str, list[str]],
    *,
    n_datasets: int = 100,
    num_obs: int = 500,
    num_samples: int = 100,
    steps: int = 50,
    seed: int | None = None,
    diagnostics: Sequence[str] = DEFAULT_DIAGNOSTICS,
    return_samples: bool = False,
    checkpoint: "str | None" = None,
    device: "str | None" = None,
) -> EstimateResult:
    """Amortized estimation under a design config, with diagnostics.

    Simulates ground truth, runs the trained CogFormer, and returns
    diagnostics-ready results. Returns posterior *draws*, not a point estimate.

    Parameters
    ----------
    model
        One of ``SUPPORTED_MODELS`` (``"ddm"``, ``"rdm"``, ``"cdm"``,
        ``"model_class"``).
    design_config
        Maps each regressor key to the intrinsic parameters it modulates, e.g.
        ``{"1": ["v","a","z","tau"], "u_1": ["v","a"], "u_2": [], "u_1:u_2": []}``.
        ``"1"`` is the intercept (always-on) block.
    n_datasets
        Number of simulated datasets / ground truths (the recovery/coverage/ECDF
        sample size). Default 100 — enough for meaningful calibration curves.
    num_obs
        Trials per simulated dataset.
    num_samples
        Posterior draws per dataset.
    steps
        Flow-matching ODE integration steps. Lower = faster, fuzzier; the demo's
        "draft vs. high-quality" lever. Cost scales as n_datasets * num_samples * steps.
    seed
        RNG seed; fix it for reproducible/cacheable results.
    diagnostics
        Subset of ``{"recovery", "coverage", "ecdf"}`` to compute.
    return_samples
        If True, retain raw posterior samples + true params on the result
        (and in ``to_dict``). Off by default to keep web payloads small.
    checkpoint, device
        Override checkpoint path / compute device; both auto-resolve when None.

    Returns
    -------
    EstimateResult
        ``.to_dict()`` yields the ``/estimate`` JSON contract.
    """
    import time

    _validate_request(model, design_config, diagnostics)
    settings = {
        "n_datasets": n_datasets, "num_obs": num_obs, "num_samples": num_samples,
        "steps": steps, "seed": seed,
    }
    param_labels = active_param_labels(model, design_config)

    t0 = time.perf_counter()
    # --- Phase 0 wiring (port from experiments/model_family/family_cf_validate.py) ---
    # bundle, net = _load_model(model, checkpoint, device)        # registry + state_dict
    # true_params, adapted = _simulate_ground_truth(bundle, design_config, settings)
    # posterior = _sample_posterior(net, adapted, num_samples, steps)  # CogFormer.sample
    # diags = _compute_diagnostics(true_params, posterior, param_labels, diagnostics)
    raise NotImplementedError(
        "estimate() pipeline not yet wired. The result/serialization types and the "
        "JSON contract above are final; next step ports the simulate->infer->diagnose "
        "core from family_cf_validate.py and reuses cogformer/diagnostics/metric/*."
    )
    elapsed_s = time.perf_counter() - t0  # noqa: F841  (reached once wired)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def active_param_labels(model: str, design_config: dict[str, list[str]]) -> list[str]:
    """Flattened labels for the active parameters, in posterior-output order.

    Intercept (``"1"``) params are bare intrinsic names; other regressors are
    ``"<intrinsic>|<regressor>"`` (e.g. ``"v|u_1"``). Order follows design_config
    key order, then intrinsic order within each key.

    NOTE: must match the column order the Adapter/ContextManager emit when
    flattening the parameter matrix; verify against ContextManager during wiring.
    """
    labels: list[str] = []
    for key, intrinsics in design_config.items():
        for name in intrinsics:
            labels.append(name if key == "1" else f"{name}|{key}")
    return labels


def _validate_request(
    model: str, design_config: dict[str, list[str]], diagnostics: Sequence[str]
) -> None:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {sorted(SUPPORTED_MODELS)}.")
    unknown = set(diagnostics) - set(DEFAULT_DIAGNOSTICS)
    if unknown:
        raise ValueError(f"Unknown diagnostics {sorted(unknown)}; supported: {DEFAULT_DIAGNOSTICS}.")
    if not design_config:
        raise ValueError("design_config must be non-empty.")
