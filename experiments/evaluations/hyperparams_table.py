"""Generate LaTeX hyperparameter summary table for CogFormer models."""
from cogformer.utils import paths
from pathlib import Path

OUT_DIR = paths.tables_dir("evaluations")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        # (description, family_value, model_class_value)
        (r"\textbf{Layers}",                          "",    ""),
        (r"\quad Encoder layers",                     "8",   "8"),
        (r"\quad Decoder layers",                     "8",   "8"),
        (r"\textbf{Attention}",                       "",    ""),
        (r"\quad Heads (encoder \& decoder)",         "8",   "8"),
        (r"\quad Projection dim $d_{\text{model}}$",  "256", "256"),
        (r"\quad QKV dim per head",                   "32",  "32"),
        (r"\textbf{Embeddings}",                      "",    ""),
        (r"\quad Time embedding dim",                 "32",  "32"),
        (r"\quad Positional embedding dim",           "32",  "32"),
        (r"\quad Seed tokens",                        "32",  "32"),
        (r"\quad Seed dim",                           "64",  "128"),
        (r"\quad Model embedding dim",                r"$-$", "8"),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{CogFormer hyperparameters for the family-level models (DDM, RDM, CDM) "
        r"and the model-class model. QKV dim per head = projection dim $\div$ heads.}",
        r"\label{tab:hyperparams}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Hyperparameter} & \textbf{Family} & \textbf{Model Class} \\",
        r"\midrule",
    ]

    for desc, fam, mc in rows:
        if fam == "" and mc == "":
            # Section header row
            lines.append(r"\addlinespace[2pt]")
            lines.append(f"{desc} & & \\\\")
        else:
            lines.append(f"{desc} & {fam} & {mc} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)

    tex_path = OUT_DIR / "hyperparams_table.tex"
    tex_path.write_text(tex)
    print(f"Saved: {tex_path}")
    print()
    print(tex)


if __name__ == "__main__":
    main()