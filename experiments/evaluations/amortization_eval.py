"""
Amortization gap evaluation script.

Computes the amortization gap for NRMSE, Calibration Error, and Posterior Contraction,
across all model families (DDM, RDM, CDM) and design configs.

Gap formula:
    Gap = 100 * (Reference - Baseline) / Baseline

Baselines:
    CogFormer_Family     vs BayesFlow         (baseline = BayesFlow)
    CogFormer_ModelClass vs CogFormer_Family   (baseline = CogFormer_Family)

Outputs:
    amortization_gap_detailed.csv  — per-parameter gaps
    amortization_gap_summary.csv   — mean ± std across parameters, per case
    amortization_gap_table.csv     — mean ± SEM averaged across configs and parameters
    amortization_gap_table.tex     — LaTeX table
"""

from cogformer.utils import paths
from pathlib import Path

import pandas as pd
import numpy as np


BASE = paths.FIGURES
OUT_DIR = paths.tables_dir("evaluations")

MODELS = ["DDM", "RDM", "CDM"]
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
METRICS = ["NRMSE", "Calibration Error", "Posterior Contraction"]

SCENARIO_LABELS = {
    "CogFormer_Family": r"\textsc{CogFormer}\textsubscript{Family} vs.\ \textsc{BayesFlow}",
    "CogFormer_ModelClass": r"\textsc{CogFormer}\textsubscript{Class} vs.\ \textsc{CogFormer}\textsubscript{Family}",
}
METRIC_LABELS = {
    "NRMSE": "NRMSE",
    "Calibration Error": "Cal.\ Error",
    "Posterior Contraction": "Post.\ Contr.",
}


def bf_path(model: str, case: str) -> Path:
    m = model.lower()
    return paths.figures_dir("model_family", "bf", m, case) / f"{m}_family_{case}_bf_metrics.csv"


def fm_family_path(model: str, case: str) -> Path:
    m = model.lower()
    return paths.figures_dir("model_family", "cf", m, case) / f"{m}_family_{case}_fm_mixed_metrics.csv"


def fm_model_class_path(model: str, case: str) -> Path:
    m = model.lower()
    return paths.figures_dir("model_class", "cf", m, case) / f"{m}_{case}_fm_metrics.csv"


def load_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    available = [m for m in METRICS if m in df.columns]
    return df[available]


def compute_gap(reference: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """100 * (reference - baseline) / baseline, aligned on index."""
    ref, base = reference.align(baseline, join="inner")
    return 100.0 * (ref - base) / base.abs()


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """
    Build a LaTeX table with rows = Model × Scenario, columns = metrics (mean ± SEM).
    agg_df must have columns: model, scenario, {metric}_mean, {metric}_sem.
    """
    scenarios = ["CogFormer_Family", "CogFormer_ModelClass"]
    n_metrics = len(METRICS)

    col_spec = "ll" + "r" * n_metrics
    header_metrics = " & ".join(METRIC_LABELS[m] for m in METRICS)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Amortization gaps (\%) averaged across design configurations and parameters.")
    lines.append(r"Values are mean $\pm$ SEM. Negative values indicate improvement over baseline.}")
    lines.append(r"\label{tab:amortization-gap}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Comparison} & " + header_metrics + r" \\")
    lines.append(r"\midrule")

    for m_idx, model in enumerate(MODELS):
        if m_idx > 0:
            lines.append(r"\midrule")
        for s_idx, scenario in enumerate(scenarios):
            row = agg_df[(agg_df["model"] == model) & (agg_df["scenario"] == scenario)]
            if row.empty:
                continue
            row = row.iloc[0]
            model_cell = r"\textbf{" + model + "}" if s_idx == 0 else ""
            scenario_cell = SCENARIO_LABELS[scenario]
            cells = [model_cell, scenario_cell]
            for metric in METRICS:
                mean_val = row.get(f"{metric}_mean", np.nan)
                sem_val = row.get(f"{metric}_sem", np.nan)
                if np.isnan(mean_val):
                    cells.append("---")
                else:
                    cells.append(f"${mean_val:+.1f} \\pm {sem_val:.1f}$")
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_rows = []
    summary_rows = []

    for model in MODELS:
        for case in CASES:
            bf_df = load_metrics(bf_path(model, case))
            fm_family_df = load_metrics(fm_family_path(model, case))
            fm_class_df = load_metrics(fm_model_class_path(model, case))

            scenario_pairs = []
            if bf_df is not None and fm_family_df is not None:
                scenario_pairs.append(("CogFormer_Family", fm_family_df, bf_df))
            else:
                if bf_df is None:
                    print(f"[skip] BayesFlow {model}/{case} — metrics not found")
                if fm_family_df is None:
                    print(f"[skip] CogFormer_Family {model}/{case} — metrics not found")

            if fm_family_df is not None and fm_class_df is not None:
                scenario_pairs.append(("CogFormer_ModelClass", fm_class_df, fm_family_df))
            else:
                if fm_family_df is None:
                    print(f"[skip] CogFormer_Family {model}/{case} — needed as baseline for ModelClass")
                if fm_class_df is None:
                    print(f"[skip] CogFormer_ModelClass {model}/{case} — metrics not found")

            for scenario_name, ref_df, baseline_df in scenario_pairs:
                gap_df = compute_gap(ref_df, baseline_df)

                for param in gap_df.index:
                    row = {
                        "model": model,
                        "case": case,
                        "scenario": scenario_name,
                        "parameter": param,
                    }
                    for metric in METRICS:
                        row[metric] = gap_df.loc[param, metric] if metric in gap_df.columns else np.nan
                    detailed_rows.append(row)

                summary_row = {"model": model, "case": case, "scenario": scenario_name}
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

    # Aggregate across configs and parameters: mean ± SEM per (model, scenario)
    agg_rows = []
    for model in MODELS:
        for scenario in ["CogFormer_Family", "CogFormer_ModelClass"]:
            subset = detailed_df[(detailed_df["model"] == model) & (detailed_df["scenario"] == scenario)]
            if subset.empty:
                continue
            agg_row = {"model": model, "scenario": scenario}
            for metric in METRICS:
                vals = subset[metric].dropna()
                n = len(vals)
                agg_row[f"{metric}_mean"] = vals.mean()
                agg_row[f"{metric}_sem"] = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            agg_rows.append(agg_row)

    agg_df = pd.DataFrame(agg_rows)
    agg_path = OUT_DIR / "amortization_gap_table.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")

    latex = generate_latex_table(agg_df)
    tex_path = OUT_DIR / "amortization_gap_table.tex"
    tex_path.write_text(latex)
    print(f"Saved: {tex_path}")

    print("\nAggregated table (mean ± SEM across configs and parameters):")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
