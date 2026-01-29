import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bayesgpt.simulators.model_family import NestedModelFamily
from bayesgpt.simulators.context_manager import ContextManager
from bayesgpt.simulators.benchmarks.ddms.ddm import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_baseline_priors


def prior_predictive_check(
    design_config: dict,
    intrinsic_params: list[str],
    variable_names: list[str],
    fig_size: tuple = None,
    title_fontsize: int = 18,
    label_fontsize: int = 14,
):
    raise NotImplementedError


def prior_pushforward_check(
    design_config: dict,
    intrinsic_params: list[str],
    variable_names: list[str],
    figsize: tuple[int, int] = None,
    title_fontsize=18,
):
    num_rows = len(list(design_config.keys()))
    num_params = len(intrinsic_params)
    if figsize is None:
        figsize = (3 * num_params, 3 * num_rows)

    fig, axarr = plt.subplots(1, num_params, figsize=figsize)

    for i, ax in enumerate(axarr):
        ax.set_title(variable_names[i], fontsize=title_fontsize)
    fig.tight_layout()
    return fig


if __name__ == "__main__":

    num_batches = 10
    batch_size = 10

    intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]
    regressed_params = ["v", "a"]

    design_config = {
        "1": intrinsic_params,
        "u_1": regressed_params,
        "u_2": regressed_params,
        "u_1:u_2": regressed_params,
    }
    ddm_family = NestedModelFamily(
        name="ddm",
        model=DDM(),
        prior_fun=ddm_baseline_priors(),
        mask_randomizer_kwargs=dict(
            free_intrinsics=intrinsic_params,
            fixed_intrinsics=[],
            fixed_values={}
        )
    )

    ddm_samples = ddm_family.batch_sample(
        design_config=design_config,
        num_obs=500,
        batch_size=1000,
    )
    rts = ddm_samples["sim_data"]["rts"]
    rts_min, rts_mean, rts_max = rts.min(axis=1), rts.mean(axis=1), rts.max(axis=1)
    rts_stat = np.hstack([rts_min, rts_mean, rts_max])
    print(rts_stat.shape)

    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, a in enumerate(ax):
        sns.histplot(rts_stat[:,i], bins=50, kde=True, ax=a)
        a.set_xlabel("RT (s)")
        a.set_ylabel("Count" if i == 0 else None)
    f.savefig("rts_test.pdf")

    variable_names = [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$\tau$"]

    fig = prior_pushforward_check(
        design_config = design_config,
        intrinsic_params = intrinsic_params,
        variable_names = variable_names
    )
    fig.savefig("test_pushforward_check.pdf")
    print("groovy")