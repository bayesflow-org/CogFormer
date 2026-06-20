"""Generate absolute metrics summary table (mean ± SEM) for all three methods."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("./experiments/figures")
OUT_DIR = Path("./experiments/evaluations")
MODELS = ["DDM", "RDM", "CDM"]
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
METRICS = ["NRMSE", "Calibration Error", "Posterior Contraction"]
SOURCES = ["BayesFlow", "CogFormer_Family", "CogFormer_ModelClass"]

SOURCE_LABELS = {
    "BayesFlow": r"\textsc{BayesFlow}",
    "CogFormer_Family": r"\textsc{CogFormer}\textsubscript{Family}",
    "CogFormer_ModelClass": r"\textsc{CogFormer}\textsubscript{Class}",
}
METRIC_LABELS = {
    "NRMSE": "NRMSE",
    "Calibration Error": r"Cal.\ Error",
    "Posterior Contraction": r"Post.\ Contr.",
}


def load(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    return df[[m for m in METRICS if m in df.columns]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        m = model.lower()
        for case in CASES:
            bf = load(BASE / "bf" / m / case / f"{m}_family_{case}_bf_metrics.csv")
            fm_fam = load(BASE / "fm" / m / case / f"{m}_family_{case}_fm_mixed_metrics.csv")
            fm_cls = load(BASE / "fm" / "model_class" / m / case / f"{m}_{case}_fm_metrics.csv")
            for source, df in [("BayesFlow", bf), ("CogFormer_Family", fm_fam), ("CogFormer_ModelClass", fm_cls)]:
                if df is None:
                    continue
                for metric in METRICS:
                    if metric in df.columns:
                        for val in df[metric].values:
                            rows.append({"model": model, "source": source, "metric": metric, "value": val})

    data = pd.DataFrame(rows)
    agg = data.groupby(["model", "source", "metric"])["value"].agg(
        mean="mean",
        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)),
    ).reset_index()

    # Save CSV
    csv_rows = []
    for model in MODELS:
        for source in SOURCES:
            row = {"model": model, "method": source}
            sub = agg[(agg["model"] == model) & (agg["source"] == source)]
            for metric in METRICS:
                r = sub[sub["metric"] == metric]
                if r.empty:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_sem"] = np.nan
                else:
                    row[f"{metric}_mean"] = r.iloc[0]["mean"]
                    row[f"{metric}_sem"] = r.iloc[0]["sem"]
            csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "absolute_metrics_table.csv", index=False)

    # Build LaTeX
    metric_header = " & ".join(METRIC_LABELS[m] for m in METRICS)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Absolute metric values (mean $\pm$ SEM) averaged across design configurations and parameters,"
        r" providing context for the amortization gap percentages.}",
        r"\label{tab:absolute-metrics}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Method} & " + metric_header + r" \\",
        r"\midrule",
    ]

    for m_idx, model in enumerate(MODELS):
        if m_idx > 0:
            lines.append(r"\midrule")
        for s_idx, source in enumerate(SOURCES):
            sub = agg[(agg["model"] == model) & (agg["source"] == source)]
            model_cell = r"\textbf{" + model + "}" if s_idx == 0 else ""
            cells = [model_cell, SOURCE_LABELS[source]]
            for metric in METRICS:
                r = sub[sub["metric"] == metric]
                if r.empty:
                    cells.append("---")
                else:
                    mean_val = r.iloc[0]["mean"]
                    sem_val = r.iloc[0]["sem"]
                    cells.append(f"${mean_val:.3f} \\pm {sem_val:.3f}$")
            lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    (OUT_DIR / "absolute_metrics_table.tex").write_text(tex)

    print(tex)
    print(f"\nSaved: {OUT_DIR / 'absolute_metrics_table.csv'}")
    print(f"Saved: {OUT_DIR / 'absolute_metrics_table.tex'}")


if __name__ == "__main__":
    main()
