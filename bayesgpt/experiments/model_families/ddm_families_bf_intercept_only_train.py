import os
os.environ["KERAS_BACKEND"] = "torch"

import bayesflow as bf

from bayesgpt.simulators.model_family import NestedModelFamily
from bayesgpt.simulators.benchmarks.ddms.ddm import DDM
from bayesgpt.simulators.benchmarks.ddms.ddm_priors import ddm_priors2
from bayesgpt.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun


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

    def sample(self, batch_size, num_obs=500, flatten_param_outputs=False, **kwargs):

        if isinstance(batch_size, tuple):
             batch_size = batch_size[0]

        sample_kwargs = {
            "max_num_regressors": 0,
            "max_num_categories": 0
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
        params = samples["param_matrices"].squeeze(axis=1)

        return {"rts": rts, "choices": choices, "params": params}

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
        seed_dim=128,
        num_heads=(8, 8),
        mlp_depths=(8, 8),
        # embed_dims=(128, 128),
        num_seeds=32,
        dropout=0.05,
        layer_norm=True
    )
    inference_net = bf.networks.FlowMatching()

    # define checkpoint filepath
    checkpoint_path = "./experiments/checkpoints/ddm_families_bf_intercept_only"

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

    if not debug:
        main()
