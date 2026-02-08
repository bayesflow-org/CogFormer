import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import bayesflow as bf

from bayesgpt.simulators.model_family import NestedModelFamily
from bayesgpt.simulators.benchmarks.ddms.ddm import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors2
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from bayesgpt.utils.simulator_utils import inspect


class DDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self):
        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_priors2(),
            mask_randomizer_kwargs=dict(
                free_intrinsics=["v", "a", "tau", "s_v", "s_tau"],
                fixed_intrinsics=[],
                fixed_values={}
            )
    )

    def sample(self, batch_size, num_obs=500, flatten_param_outputs=True, check=False, **kwargs):
        intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]

        design_config = {
            "1": intrinsic_params,
            "u_1": ["v", "a"],
            "u_2": ["v", "a"]
        }

        if isinstance(batch_size, tuple):
             batch_size = batch_size[0]

        sample_kwargs = {
            "design_config": design_config,
            "max_num_regressors": 2,
            "max_num_categories": 2,
        }

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=flatten_param_outputs,
            link_fun=ddm_link_fun(),
            **sample_kwargs,
            **kwargs
        )

        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        params = samples["param_matrices"]

        params = params[:, params[0] != 0]

        # Special treatment for BF:
        # Trim away zeros from non-regressed params
        if check:
            f = self.model_family.check_regressed_priors(
                design_config=design_config,
                num_obs=num_obs,
                link_fun=ddm_link_fun()
            )

        return {"rts": rts, "choices": choices, "params": params}


def test():
    ddm_family_simulator = DDMModelFamilyBF()
    ddm_samples = ddm_family_simulator.sample(4)

    for k, v in ddm_samples.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        elif isinstance(v, dict):
            print(k, v.keys())
        else:
            print(k, v)
    print(ddm_samples["params"])


def main():
    # Define simulator
    ddm_family_simulator = DDMModelFamilyBF()

    # Define adapter
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )

    # define networks
    summary_net = bf.networks.SetTransformer(
        summary_dim=32,
        seed_dim=64,
        num_heads=(4, 4, 4, 4),
        mlp_depths=(1, 1, 1, 1),
        embed_dims=(128, 128, 128, 128),
        mlp_widths=(128, 128, 128, 128),
        num_seeds=4,
    )
    inference_net = bf.networks.FlowMatching()

    # define checkpoint filepath
    checkpoint_path = "./bayesgpt/experiments/checkpoints/ddm_families_bf_regressed"

    # Set up workflow
    workflow = bf.BasicWorkflow(
        simulator=ddm_family_simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=checkpoint_path
    )

    history = workflow.fit_online(
        epochs=1000,
        steps_per_epoch=100,
        batch_size=64
    )

if __name__ == '__main__':
    debug = False
    if debug:
        test()
    else:
        main()
