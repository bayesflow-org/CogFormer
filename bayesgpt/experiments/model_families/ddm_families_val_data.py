import argparse
import numpy as np
from pathlib import Path

from bayesgpt.simulators.model_family import NestedModelFamily
from bayesgpt.simulators.benchmarks.ddms import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors2
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun

def get_benchmark_design_configs():
    free_params = ["v", "a", "tau"]
    fixed_params = ["s_v", "s_tau"]
    intrinsic_params = free_params + fixed_params

    intercept_only = {
        "1": intrinsic_params,
        "u_1": [],
        "u_2": [],
        "u_1:u_2": []
    }

    av_regressed = {
        "1": intrinsic_params,
        "u_1": ["v", "a"],
        "u_2": ["v", "a"],
        "u_1:u_2": []
    }

    fixed = {
        "1": free_params,
        "u_1": [],
        "u_2": [],
        "u_1:u_2": []
    }

    fixed_regressed = {
        "1": free_params,
        "u_1": ["v", "a"],
        "u_2": ["v", "a"],
        "u_1:u_2": []
    }

    interaction = {
        "1": intrinsic_params,
        "u_1": ["v", "a", "tau", "s_v"],
        "u_2": ["v", "a", "tau"],
        "u_1:u_2": ["v", "a"]
    }

    names = ["intercept_only", "av_regressed", "fixed", "fixed_regressed", "interaction"]
    configs = [intercept_only, av_regressed, fixed, fixed_regressed, interaction]
    return list(zip(names, configs))

def infer_free_fixed_intrinsics(
    design_config: dict[str, list[str]],
    all_intrinsics: list[str],
    default_fixed_values: dict[str, float],
):
    # If a param never appears in the design_config value-lists, treat it as fixed.
    used = set()
    for _, plist in design_config.items():
        used.update(plist)

    free_intrinsics = [p for p in all_intrinsics if p in used]
    fixed_intrinsics = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed_intrinsics if p in default_fixed_values}
    return free_intrinsics, fixed_intrinsics, fixed_values

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="./bayesgpt/experiments/data", help="Output directory")

    # Validation settings
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)


    return p.parse_args()

def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Intrinsics / display names
    intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

    # Rebuild model-family exactly like training
    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = NestedModelFamily(
        model=DDM(),
        name="DDM",
        prior_fun=ddm_priors2(),# ddm_baseline_priors(),
        mask_randomizer_kwargs={  # default; per-config we’ll override
            "free_intrinsics": intrinsic_params,
            "fixed_intrinsics": [],
            "fixed_values": {},
        },
    )

    benchmark = get_benchmark_design_configs()

    default_fixed_values = {"s_v": 0.0, "s_tau": 0.0}

    for cfg_name, design_config in benchmark:
        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config=design_config,
            all_intrinsics=intrinsic_params,
            default_fixed_values=default_fixed_values,
        )

        # Per-config mask randomizer settings
        val_params_kwargs = {
            "free_intrinsics": free_intr,
            "fixed_intrinsics": fixed_intr,
            "fixed_values": fixed_vals,
        }

        val_sample_config = {
            "mask_randomizer_kwargs": val_params_kwargs,
            "min_num_regressors": 0,
            "fixed_config": True,  # make intent explicit
        }

        # Simulate
        test_samples = model_family.batch_sample(
            **model_family_config,
            **val_sample_config,
            batch_size=args.batch_size,
            flatten_param_outputs=True,
            design_config=design_config,
            link_fun=ddm_link_fun()
        )

        # save
        save_path = outdir / f"ddm_{cfg_name}_data.npz"
        np.savez(save_path, **test_samples)

if __name__ == "__main__":
    main()