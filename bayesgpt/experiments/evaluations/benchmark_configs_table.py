"""Generate LaTeX summary table for benchmark design configurations."""
from pathlib import Path

OUT_DIR = Path("./bayesgpt/experiments/evaluations")

CONFIGS = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
CONFIG_LABELS = {
    "intercept_only":  "Intercept Only",
    "fixed":           "Fixed Variability",
    "regressed":       "Regressed",
    "fixed_regressed": "Fixed + Regressed",
    "interaction":     "With Interaction",
}

BENCHMARKS = {
    "DDM": {
        "intercept_only":  {"1": ["v","a","z",r"\tau","s_v",r"s_\tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
        "fixed":           {"1": ["v","a","z",r"\tau"],                 "u_1": [], "u_2": [], "u_1:u_2": []},
        "regressed":       {"1": ["v","a","z",r"\tau","s_v",r"s_\tau"], "u_1": ["v","a","z"], "u_2": ["v","a","z"], "u_1:u_2": []},
        "fixed_regressed": {"1": ["v","a","z",r"\tau"],                 "u_1": ["v","a","z"], "u_2": ["v","a","z"], "u_1:u_2": []},
        "interaction":     {"1": ["v","a","z",r"\tau","s_v",r"s_\tau"], "u_1": ["v","a","z",r"\tau","s_v"], "u_2": ["v","a","z",r"\tau"], "u_1:u_2": ["v","a","z"]},
    },
    "RDM": {
        "intercept_only":  {"1": ["v","v_{\\mathrm{d}}","a",r"\tau","s_v",r"s_\tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
        "fixed":           {"1": ["v","v_{\\mathrm{d}}","a",r"\tau"],                 "u_1": [], "u_2": [], "u_1:u_2": []},
        "regressed":       {"1": ["v","v_{\\mathrm{d}}","a",r"\tau","s_v",r"s_\tau"], "u_1": ["v_{\\mathrm{d}}","a"], "u_2": ["v_{\\mathrm{d}}","a"], "u_1:u_2": []},
        "fixed_regressed": {"1": ["v","v_{\\mathrm{d}}","a",r"\tau"],                 "u_1": ["v_{\\mathrm{d}}","a"], "u_2": ["v_{\\mathrm{d}}","a"], "u_1:u_2": []},
        "interaction":     {"1": ["v","v_{\\mathrm{d}}","a",r"\tau","s_v",r"s_\tau"], "u_1": ["v","v_{\\mathrm{d}}","a",r"\tau","s_v"], "u_2": ["v","v_{\\mathrm{d}}","a",r"\tau"], "u_1:u_2": ["v","v_{\\mathrm{d}}","a"]},
    },
    "CDM": {
        "intercept_only":  {"1": ["v",r"v_\theta","a",r"\tau","s_v",r"s_\tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
        "fixed":           {"1": ["v",r"v_\theta","a",r"\tau"],                 "u_1": [], "u_2": [], "u_1:u_2": []},
        "regressed":       {"1": ["v",r"v_\theta","a",r"\tau","s_v",r"s_\tau"], "u_1": ["v","a"], "u_2": ["v","a"], "u_1:u_2": []},
        "fixed_regressed": {"1": ["v",r"v_\theta","a",r"\tau"],                 "u_1": ["v","a"], "u_2": ["v","a"], "u_1:u_2": []},
        "interaction":     {"1": ["v",r"v_\theta","a",r"\tau","s_v",r"s_\tau"], "u_1": ["v",r"v_\theta","a",r"\tau","s_v"], "u_2": ["v",r"v_\theta","a",r"\tau"], "u_1:u_2": ["v",r"v_\theta","a"]},
    },
}

MODELS = ["DDM", "RDM", "CDM"]


def fmt(params):
    if not params:
        return r"$-$"
    return "$" + ", ".join(p if p.startswith("\\") else p for p in params) + "$"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Benchmark design configurations. Each cell lists the parameters "
        r"assigned to the corresponding effect type. A dash ($-$) indicates no parameters "
        r"are assigned. $v_{\mathrm{d}}$ abbreviates $v_{\mathrm{diff}}$ (RDM).}",
        r"\label{tab:benchmark-configs}",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Model} & \textbf{Intercept} ($\mathbf{1}$) "
        r"& \textbf{Slope} ($u_1$) & \textbf{Slope} ($u_2$) & \textbf{Interaction} ($u_1{\times}u_2$) \\",
        r"\midrule",
    ]

    for cfg in CONFIGS:
        cfg_label = CONFIG_LABELS[cfg]
        for m_idx, model in enumerate(MODELS):
            dc = BENCHMARKS[model][cfg]
            config_cell = r"\textbf{" + cfg_label + "}" if m_idx == 0 else ""
            row = " & ".join([
                config_cell,
                model,
                fmt(dc["1"]),
                fmt(dc["u_1"]),
                fmt(dc["u_2"]),
                fmt(dc["u_1:u_2"]),
            ]) + r" \\"
            lines.append(row)
        if cfg != CONFIGS[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)

    tex_path = OUT_DIR / "benchmark_configs_table.tex"
    tex_path.write_text(tex)
    print(f"Saved: {tex_path}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
