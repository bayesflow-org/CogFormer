from cogformer.utils import paths
import argparse
import numpy as np
import pandas as pd

from pathlib import Path


DATA_DIR = paths.tables_dir("ablations", "component")

CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]

CONDITIONS = ["baseline", "no_sab", "no_mab", "no_film", "no_fourier"]

CONDITION_LABELS = {
    "baseline":   "Baseline",
    "no_sab":     "No SAB",
    "no_mab":     "No MAB",
    "no_film":    "No FiLM",
    "no_fourier": "No Fourier",
}

CASE_LABELS = {
    "intercept_only":  "Intercept Only",
    "fixed":           "Fixed",
    "regressed":       "Regressed",
    "fixed_regressed": "Fixed Regressed",
    "interaction":     "Interaction",
}

# For each metric: True = higher is better, False = lower is better
METRIC_DIRECTION = {
    "NRMSE":                False,
    "Calibration Error":    False,
    "Posterior Contraction": True,
}


def load_metrics(data_dir: Path, case: str) -> dict[str, pd.DataFrame]:
    metrics = {}
    for condition in CONDITIONS:
        path = data_dir / f"{condition}_{case}_metrics.csv"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue
        df = pd.read_csv(path, sep=",", index_col=0)
        metrics[condition] = df
    return metrics


def summarise(metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean of each metric across all active parameters per condition."""
    rows = []
    for condition, df in metrics.items():
        row = {"Condition": CONDITION_LABELS[condition]}
        for col in METRIC_DIRECTION:
            if col in df.columns:
                row[col] = df[col].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Condition")


def compute_delta(summary: pd.DataFrame) -> pd.DataFrame:
    """Absolute change vs. baseline for each metric."""
    if CONDITION_LABELS["baseline"] not in summary.index:
        raise ValueError("Baseline metrics not found — run eval for 'baseline' first.")
    baseline = summary.loc[CONDITION_LABELS["baseline"]]
    delta = summary - baseline
    return delta.drop(index=CONDITION_LABELS["baseline"])


def to_latex(summary: pd.DataFrame, delta: pd.DataFrame, case: str) -> str:
    ablation_rows = [r for r in summary.index if r != CONDITION_LABELS["baseline"]]
    all_rows = [CONDITION_LABELS["baseline"]] + ablation_rows

    col_header = " & ".join([""] + all_rows) + r" \\"
    metric_lines = []

    for metric, higher_is_better in METRIC_DIRECTION.items():
        if metric not in summary.columns:
            continue

        values = summary[metric]
        best = values.max() if higher_is_better else values.min()

        cells = []
        for cond in all_rows:
            v = values[cond]
            cell = f"{v:.4f}"
            if np.isclose(v, best):
                cell = r"\textbf{" + cell + "}"
            if cond != CONDITION_LABELS["baseline"] and cond in delta.index:
                d = delta.loc[cond, metric]
                sign = "+" if d > 0 else ""
                cell += rf" \small{{({sign}{d:.4f})}}"
            cells.append(cell)

        arrow = r"$\uparrow$" if higher_is_better else r"$\downarrow$"
        short = {
            "NRMSE": r"NRMSE",
            "Calibration Error": r"Cal.\ Error",
            "Posterior Contraction": r"Post.\ Contr.",
        }.get(metric, metric)
        metric_lines.append(f"{short} {arrow} & " + " & ".join(cells) + r" \\")

    metric_block = "\n".join(metric_lines)
    col_spec = "l" + "c" * len(all_rows)
    case_label = CASE_LABELS[case]

    return rf"""\begin{{table}}[t]
\centering
\caption{{Component ablation study — {case_label} case. Mean metrics across
all active parameters. $\Delta$ (in parentheses) is the change relative to the
baseline CogFormer. NRMSE and Calibration Error: lower is better ($\downarrow$).
Posterior Contraction: higher is better ($\uparrow$). Best result per row is
\textbf{{bolded}}.}}
\label{{tab:component_ablation_{case}}}
\begin{{tabular}}{{{col_spec}}}
\toprule
{col_header}
\midrule
{metric_block}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def run_case(data_dir: Path, case: str):
    print(f"\n=== Case: {CASE_LABELS[case]} ===")
    print("Loading per-condition metrics...")
    metrics = load_metrics(data_dir, case)

    if not metrics:
        print("  No metrics files found for this case. Skipping.")
        return

    print("Computing summary and Δ from baseline...")
    summary = summarise(metrics)
    print(f"\n--- Mean metrics per condition ({case}) ---")
    print(summary.to_string())

    summary.to_csv(data_dir / f"component_ablation_summary_{case}.csv")
    print(f"\nSaved summary to {data_dir / f'component_ablation_summary_{case}.csv'}")

    if CONDITION_LABELS["baseline"] in summary.index:
        delta = compute_delta(summary)
        print(f"\n--- Δ from baseline ({case}) ---")
        print(delta.to_string())
        delta.to_csv(data_dir / f"component_ablation_delta_{case}.csv")
        print(f"Saved delta to {data_dir / f'component_ablation_delta_{case}.csv'}")

        latex = to_latex(summary, delta, case)
        latex_path = data_dir / f"component_ablation_table_{case}.tex"
        latex_path.write_text(latex)
        print(f"Saved LaTeX table to {latex_path}")
    else:
        print("\n  Baseline not found — skipping Δ and LaTeX table.")


def main(data_dir: Path, cases: list[str]):
    for case in cases:
        run_case(data_dir, case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate component ablation metrics into per-case comparison tables")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="Directory containing per-condition metrics CSVs")
    parser.add_argument("--cases", nargs="+", default=CASES, choices=CASES,
                        help="Which cases to compare (default: all)")
    args = parser.parse_args()
    main(Path(args.data_dir), args.cases)
