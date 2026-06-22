"""
C2ST evaluation script.

Aggregates per-case C2ST summary CSVs across DDM, RDM, CDM and produces:
    c2st_table.csv  — mean ± SEM across cases per (model, scenario)
    c2st_table.tex  — LaTeX summary table
"""
from cogformer.utils import paths

import numpy as np
import pandas as pd


C2ST_DIR = paths.figures_dir("mf", "c2st")
OUT_DIR = paths.tables_dir("evaluations")

MODELS = ["DDM", "RDM", "CDM"]
SCENARIOS = ["Family_vs_BF", "Class_vs_Family"]
SCENARIO_LABELS = {
    "Family_vs_BF": r"\textsc{CogFormer}\textsubscript{Family} vs.\ \textsc{BayesFlow}",
    "Class_vs_Family": r"\textsc{CogFormer}\textsubscript{Class} vs.\ \textsc{CogFormer}\textsubscript{Family}",
}


def load_summary(model: str) -> pd.DataFrame | None:
    path = C2ST_DIR / model.lower() / f"{model.lower()}_c2st_summary.csv"
    if not path.exists():
        print(f"[missing] {path}")
        return None
    return pd.read_csv(path)


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{C2ST results (mean $\pm$ SEM across design configurations)."
        r" Mean Accuracy and Joint C2ST near 0.5 indicate indistinguishable posteriors.}",
        r"\label{tab:c2st}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Comparison} & Mean Accuracy & Joint C2ST \\",
        r"\midrule",
    ]

    for m_idx, model in enumerate(MODELS):
        if m_idx > 0:
            lines.append(r"\midrule")
        for s_idx, scenario in enumerate(SCENARIOS):
            row = agg_df[(agg_df["model"] == model) & (agg_df["scenario"] == scenario)]
            if row.empty:
                continue
            row = row.iloc[0]
            model_cell = r"\textbf{" + model + "}" if s_idx == 0 else ""
            acc_str = f"${row['mean_accuracy_mean']:.3f} \\pm {row['mean_accuracy_sem']:.3f}$"
            joint_str = f"${row['joint_c2st_mean']:.3f} \\pm {row['joint_c2st_sem']:.3f}$"
            lines.append(f"{model_cell} & {SCENARIO_LABELS[scenario]} & {acc_str} & {joint_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model in MODELS:
        df = load_summary(model)
        if df is not None:
            all_rows.append(df)

    if not all_rows:
        print("No summary CSVs found. Run the C2ST scripts first.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Aggregate mean ± SEM across cases per (model, scenario)
    agg_rows = []
    for model in MODELS:
        for scenario in SCENARIOS:
            sub = combined[(combined["model"] == model) & (combined["scenario"] == scenario)]
            if sub.empty:
                continue
            n = len(sub)
            agg_rows.append({
                "model": model,
                "scenario": scenario,
                "mean_accuracy_mean": sub["mean_accuracy"].mean(),
                "mean_accuracy_sem": sub["mean_accuracy"].std(ddof=1) / np.sqrt(n),
                "joint_c2st_mean": sub["joint_c2st"].mean(),
                "joint_c2st_sem": sub["joint_c2st"].std(ddof=1) / np.sqrt(n),
            })

    agg_df = pd.DataFrame(agg_rows)
    csv_path = OUT_DIR / "c2st_table.csv"
    agg_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    tex = generate_latex_table(agg_df)
    tex_path = OUT_DIR / "c2st_table.tex"
    tex_path.write_text(tex)
    print(f"Saved: {tex_path}")

    print("\nC2ST aggregated table:")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()