import os

import numpy as np

os.environ["KERAS_BACKEND"] = "torch"

import logging
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from simulators.model_family import NestedModelFamily
from simulators.benchmarks.ddms.ddm import DDM
from simulators.benchmarks.ddms.ddm_priors import ddm_baseline_priors


class DDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self):
        self.model_family = NestedModelFamily(
            model=DDM(),
            prior_fun=ddm_baseline_priors(),
            mask_randomizer_kwargs=dict(
                free_intrinsics=["v", "a", "tau"],
                fixed_intrinsics=["s_v", "s_tau"],
                fixed_values={"s_v": 0, "s_tau": 0},
            )
    )

    def sample(self, batch_size, num_obs=500, flatten_param_outputs=False, **kwargs):

        if isinstance(batch_size, tuple):
             batch_size = batch_size[0]

        sample_kwargs = {
            'min_num_regressors': 0,
            "max_num_regressors": 0,
            "max_num_categories": 0,
            "fixed_config": True
        }

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=flatten_param_outputs,
            **sample_kwargs,
            **kwargs
        )

        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        params = samples["param_matrices"]

        # Special treatment for BF:
        # Trim away zeros from non-regressed params
        params = params[:, params[0] != 0]

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
    summary_net = bf.networks.SetTransformer()
    inference_net = bf.networks.FlowMatching()

    # define checkpoint filepath
    checkpoint_path = "./experiments/checkpoints/ddm_families_bf_fixed"

    # Set up workflow
    workflow = bf.BasicWorkflow(
        simulator=ddm_family_simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=checkpoint_path
    )

    history = workflow.fit_online(
        epochs=500,
        steps_per_epoch=100,
        batch_size=32
    )

    param_names = [r"$v$", r"$a$", r"$\tau$"]

    evals = workflow.compute_default_diagnostics(test_data=200, variable_names=param_names)

    evals_dir = Path("./experiments/evaluations")
    evals_dir.mkdir(parents=True, exist_ok=True)
    evals.to_csv(evals_dir / "ddm_families_bf_fixed_evaluations.csv", sep=";")
    logging.info("Metric evaluation is now finished.")


if __name__ == '__main__':
    debug = False

    if debug:
        test()
    else:
        main()
