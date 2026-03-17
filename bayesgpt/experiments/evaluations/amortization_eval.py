"""
Amortization gap evaluation script.

Computes the amortization gap between BayesFlow (baseline), BayesGPT model family,
and BayesGPT model class for NRMSE, Calibration Error, and Posterior Contraction,
across all model families (DDM, RDM, CDM) and design configs.

Gap formula:
    Gap = 100 * (Reference - Baseline) / Baseline

where Baseline = BayesFlow.

Outputs:
    amortization_gap_detailed.csv  — per-parameter gaps
    amortization_gap_summary.csv   — mean ± std across parameters
"""

from pathlib import Path

import pandas as pd
import numpy as np


BASE = Path("./bayesgpt/experiments/figures")
OUT_DIR = Path("./bayesgpt/experiments/evaluations")

MODELS = ["DDM", "RDM", "CDM"]
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction", "full"]
METRICS = ["NRMSE", "Calibration Error", "Posterior Contraction"]


def bf_path(model: str, case: str) -> Path:
    m = model.lower()
    return BASE / "bf" / m / case / f"{m}_family_{case}_bf_metrics.csv"


def fm_family_path(model: str, case: str) -> Path:
    m = model.lower()
    return BASE / "fm" / m / case / f"{m}_family_{case}_fm_mixed_metrics.csv"


def fm_model_class_path(model: str, case: str) -> Path:
    m = model.lower()
    return BASE / "fm" / "model_class" / m / case / f"{m}_{case}_fm_metrics.csv"


def load_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    # Keep only the three metrics of interest
    available = [m for m in METRICS if m in df.columns]
    return df[available]


def compute_gap(reference: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """100 * (reference - baseline) / baseline, aligned on index."""
    ref, base = reference.align(baseline, join="inner")
    return 100.0 * (ref - base) / base.abs()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_rows = []
    summary_rows = []

    scenarios = {
        "BayesGPT_Family": fm_family_path,
        "BayesGPT_ModelClass": fm_model_class_path,
    }

    for model in MODELS:
        for case in CASES:
            baseline_df = load_metrics(bf_path(model, case))
            if baseline_df is None:
                print(f"[skip] BayesFlow {model}/{case} — metrics not found")
                continue

            for scenario_name, path_fn in scenarios.items():
                ref_df = load_metrics(path_fn(model, case))
                if ref_df is None:
                    print(f"[skip] {scenario_name} {model}/{case} — metrics not found")
                    continue

                gap_df = compute_gap(ref_df, baseline_df)

                # Detailed: one row per parameter
                for param in gap_df.index:
                    row = {
                        "model": model,
                        "case": case,
                        "scenario": scenario_name,
                        "parameter": param,
                    }
                    for metric in METRICS:
                        if metric in gap_df.columns:
                            row[metric] = gap_df.loc[param, metric]
                        else:
                            row[metric] = np.nan
                    detailed_rows.append(row)

                # Summary: mean and std across parameters
                summary_row = {
                    "model": model,
                    "case": case,
                    "scenario": scenario_name,
                }
                for metric in METRICS:
                    if metric in gap_df.columns:
                        summary_row[f"{metric}_mean"] = gap_df[metric].mean()
                        summary_row[f"{metric}_std"] = gap_df[metric].std()
                    else:
                        summary_row[f"{metric}_mean"] = np.nan
                        summary_row[f"{metric}_std"] = np.nan
                summary_rows.append(summary_row)

                print(f"[done] {scenario_name} {model}/{case}")

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)

    detailed_path = OUT_DIR / "amortization_gap_detailed.csv"
    summary_path = OUT_DIR / "amortization_gap_summary.csv"

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved: {detailed_path}")
    print(f"Saved: {summary_path}")
    print(f"\nSummary preview:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
