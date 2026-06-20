import argparse
import numpy as np
import pandas as pd

from pathlib import Path


MODELS = ["DDM", "RDM", "CDM"]

CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]

CASE_LABELS = {
    "intercept_only":  "Intercept Only",
    "fixed":           "Fixed",
    "regressed":       "Regressed",
    "fixed_regressed": "Fixed Regressed",
    "interaction":     "Interaction",
}

CONDITION_LABELS = {
    "with_embed": "With Embedding",
    "no_embed":   "No Embedding",
}

METRIC_DIRECTION = {
    "NRMSE":                False,
    "Calibration Error":    False,
    "Posterior Contraction": True,
}


def load_metrics(with_embed_dir: Path, no_embed_dir: Path, model: str, case: str) -> dict[str, pd.DataFrame]:
    metrics = {}
    for condition, base_dir in [("with_embed", with_embed_dir), ("no_embed", no_embed_dir)]:
        path = base_dir / model.lower() / case / f"{model.lower()}_{case}_fm_metrics.csv"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue
        metrics[condition] = pd.read_csv(path, index_col=0)
    return metrics


def summarise(metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for condition, df in metrics.items():
        row = {"Condition": CONDITION_LABELS[condition]}
        for col in METRIC_DIRECTION:
            if col in df.columns:
                row[col] = df[col].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Condition")


def compute_delta(summary: pd.DataFrame) -> pd.DataFrame:
    baseline_label = CONDITION_LABELS["with_embed"]
    if baseline_label not in summary.index:
        raise ValueError("with_embed metrics not found — run validate for the with-embedding checkpoint first.")
    baseline = summary.loc[baseline_label]
    delta = summary - baseline
    return delta.drop(index=baseline_label)


def to_latex(summary: pd.DataFrame, delta: pd.DataFrame, model: str, case: str) -> str:
    all_rows = list(summary.index)
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
            if cond != CONDITION_LABELS["with_embed"] and cond in delta.index:
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
\caption{{Model embedding ablation — {model} / {case_label}. Mean metrics across
all active parameters. $\Delta$ (in parentheses) is the change relative to the
with-embedding baseline. NRMSE and Calibration Error: lower is better ($\downarrow$).
Posterior Contraction: higher is better ($\uparrow$). Best result per row is
\textbf{{bolded}}.}}
\label{{tab:embed_ablation_{model.lower()}_{case}}}
\begin{{tabular}}{{{col_spec}}}
\toprule
{col_header}
\midrule
{metric_block}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def run(with_embed_dir: Path, no_embed_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model in MODELS:
        for case in CASES:
            print(f"\n=== {model} / {CASE_LABELS[case]} ===")
            metrics = load_metrics(with_embed_dir, no_embed_dir, model, case)
            if not metrics:
                print("  No metrics found, skipping.")
                continue

            summary = summarise(metrics)
            print(summary.to_string())

            stem = f"{model.lower()}_{case}"
            summary.to_csv(out_dir / f"{stem}_embedding_summary.csv")

            if CONDITION_LABELS["with_embed"] in summary.index and CONDITION_LABELS["no_embed"] in summary.index:
                delta = compute_delta(summary)
                print(f"\nΔ from with-embedding baseline:")
                print(delta.to_string())
                delta.to_csv(out_dir / f"{stem}_embedding_delta.csv")

                latex = to_latex(summary, delta, model, case)
                (out_dir / f"{stem}_embedding_table.tex").write_text(latex)
                print(f"Saved LaTeX to {out_dir / f'{stem}_embedding_table.tex'}")

                # Accumulate for aggregate table
                for metric in METRIC_DIRECTION:
                    if metric in summary.columns:
                        all_summaries.append({
                            "Model": model,
                            "Case": CASE_LABELS[case],
                            "Metric": metric,
                            "With Embedding": summary.loc[CONDITION_LABELS["with_embed"], metric],
                            "No Embedding": summary.loc[CONDITION_LABELS["no_embed"], metric],
                            "Delta": delta.loc[CONDITION_LABELS["no_embed"], metric],
                        })

    if all_summaries:
        agg = pd.DataFrame(all_summaries)
        agg.to_csv(out_dir / "embedding_ablation_aggregate.csv", index=False)
        print(f"\nSaved aggregate to {out_dir / 'embedding_ablation_aggregate.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Compare model class with/without embedding ablation")
    parser.add_argument(
        "--with_embed_dir",
        type=str,
        default="./experiments/figures/fm/model_class",
        help="Output directory from validate run with model embedding",
    )
    parser.add_argument(
        "--no_embed_dir",
        type=str,
        default="./experiments/figures/fm/model_class_no_embed",
        help="Output directory from validate run without model embedding",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./experiments/model_classes/embedding_ablation_data",
        help="Where to write summary CSVs and LaTeX tables",
    )
    args = parser.parse_args()
    run(Path(args.with_embed_dir), Path(args.no_embed_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
