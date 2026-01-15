import numpy as np

from simulators.model_family import NestedModelFamily
from simulators.benchmarks.ddms.ddm import DDM
from simulators.benchmarks.ddms.ddm_priors import ddm_baseline_priors


def test():
    ddm_family = NestedModelFamily(
        model=DDM(),
        name="DDM",
        prior_fun=ddm_baseline_priors(),
        regressed_params=["v", "a"],
        mask_randomizer_kwargs=dict(
            free_intrinsics=["v", "a", "tau", "s_v", "s_tau"],
            fixed_intrinsics=[],
            fixed_values={}
        ),
    )

    random_sample_kwargs = {
        'min_num_regressors': 1,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": False
    }

    fixed_sample_kwargs = {
        'min_num_regressors': 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": True
    }

    ddm_random_samples = ddm_family.batch_sample(
        batch_size=10,
        flatten_param_outputs=True,
        num_obs=200,
        **random_sample_kwargs
    )

    ddm_fixed_samples = ddm_family.batch_sample(
        batch_size=10,
        flatten_param_outputs=True,
        num_obs=200,
        **fixed_sample_kwargs
    )
    for i in range(10):
        print(ddm_random_samples["design_configs"][i])

    for i in range(10):
        print(ddm_fixed_samples["design_configs"][i])

if __name__ == '__main__':
    test()
