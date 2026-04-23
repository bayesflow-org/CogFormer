import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import bayesflow as bf

from cogformer.simulators.model_family import NestedModelFamily
from cogformer.simulators.benchmarks.cdms.cdm import CDM
from cogformer.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from cogformer.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun


CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 0,
        "max_num_regressors": 0,
        "max_num_categories": 0,
        "fixed_config": False,
        "design_config": None,
        "flatten_param_outputs": False,
        "squeeze_outputs": True,
        "expected_num_active": None,  # varies: random subset of 6 intrinsics
    },
    "fixed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"],
        "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0,
        "max_num_regressors": 0,
        "max_num_categories": 0,
        "fixed_config": True,
        "design_config": None,
        "flatten_param_outputs": False,
        "squeeze_outputs": True,
        "expected_num_active": 4,  # v, v_theta, a, tau
    },
    "regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": True,
        "design_config": None,
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "expected_num_active": 10,  # 6 intercepts + 2 u_1 + 2 u_2
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"],
        "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": True,
        "design_config": None,
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "expected_num_active": 8,  # 4 intercepts (v, v_theta, a, tau) + 2 u_1 + 2 u_2
    },
    "interaction": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": False,
        "design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v"],
            "u_2": ["v", "v_theta", "a", "tau"],
            "u_1:u_2": ["v", "v_theta", "a"],
        },
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "expected_num_active": 18,  # 6 + 5 + 4 + 3 from design_config
    },
    "full": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [],
        "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2,
        "max_num_regressors": 2,
        "max_num_categories": 2,
        "fixed_config": True,
        "design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        },
        "flatten_param_outputs": True,
        "squeeze_outputs": False,
        "expected_num_active": 24,  # 6 * 4 columns
    },
}


class CDMModelFamilyBF(bf.simulators.Simulator):

    def __init__(self, case: str):
        if case not in CASE_CONFIGS:
            raise ValueError(f"Unknown case: {case}. Must be one of {list(CASE_CONFIGS.keys())}")

        self.case = case
        self.config = CASE_CONFIGS[case]

        self.model_family = NestedModelFamily(
            model=CDM(),
            prior_fun=cdm_priors(),
            regressed_params=self.config["regressed_params"],
            mask_randomizer_kwargs=dict(
                free_intrinsics=self.config["free_intrinsics"],
                fixed_intrinsics=self.config["fixed_intrinsics"],
                fixed_values=self.config["fixed_values"],
            )
        )

    def sample(self, batch_size, num_obs=500, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        sample_kwargs = {
            "min_num_regressors": self.config["min_num_regressors"],
            "max_num_regressors": self.config["max_num_regressors"],
            "max_num_categories": self.config["max_num_categories"],
        }

        if self.config["fixed_config"]:
            sample_kwargs["fixed_config"] = True

        if self.config["design_config"] is not None:
            sample_kwargs["design_config"] = self.config["design_config"]

        flatten_param_outputs = self.config["flatten_param_outputs"]

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=flatten_param_outputs,
            link_fun=cdm_link_fun(),
            **sample_kwargs,
            **kwargs
        )

        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        param_matrices = samples["param_matrices"]
        param_masks = samples["param_masks"]

        if self.config["squeeze_outputs"]:
            param_matrices = param_matrices.squeeze(axis=1)
            param_masks = param_masks.squeeze(axis=1)

        return {
            "design_matrices": design_matrices,
            "rts": rts,
            "choices": choices,
            "params": param_matrices,
            "masks": param_masks,
        }


def test(case: str):
    config = CASE_CONFIGS[case]
    cdm_family_simulator = CDMModelFamilyBF(case=case)
    cdm_samples = cdm_family_simulator.sample(4)

    print(f"Case: {case}")
    for k, v in cdm_samples.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: {v.keys()}")
        else:
            print(f"  {k}: {v}")

    masks = cdm_samples["masks"]
    params = cdm_samples["params"]

    # Check 1: active param count matches expected
    num_active = int(masks[0].astype(bool).sum())
    expected = config["expected_num_active"]
    if expected is not None:
        status = "OK" if num_active == expected else f"FAIL (expected {expected})"
        print(f"\n  [Check] Active params: {num_active}  →  {status}")
        assert num_active == expected, f"Active param count: got {num_active}, expected {expected}"
    else:
        print(f"\n  [Check] Active params: {num_active}  →  (variable, no assertion)")

    # Check 2: mask is identical across batch for fixed designs
    if config["fixed_config"] or config["design_config"] is not None:
        consistent = np.all(masks == masks[0])
        print(f"  [Check] Mask consistent across batch: {'OK' if consistent else 'FAIL'}")
        assert consistent, "Mask differs across batch items for a fixed-config case"

    # Check 3: params are zero wherever mask is zero
    zero_where_masked = np.allclose(params * (1 - masks), 0.0)
    print(f"  [Check] Params zero where masked: {'OK' if zero_where_masked else 'FAIL'}")
    assert zero_where_masked, "Non-zero param values found at masked-out positions"

    adapter = (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )

    adapted_sims = adapter(cdm_family_simulator.sample(4))
    print("\nAdapted samples:")
    for k, v in adapted_sims.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: {v.keys()}")
        else:
            print(f"  {k}: {v}")


def main(
    case: str,
    epochs: int = 1000,
    steps_per_epoch: int = 100,
    batch_size: int = 64,
):
    cdm_family_simulator = CDMModelFamilyBF(case=case)

    adapter = (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )

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

    checkpoint_path = f"./cogformer/experiments/checkpoints/cdm_families_bf_{case}"

    workflow = bf.BasicWorkflow(
        simulator=cdm_family_simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=checkpoint_path,
    )

    history = workflow.fit_online(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
    )

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CDM Model Family Training")
    parser.add_argument(
        "--case",
        type=str,
        default="intercept_only",
        choices=list(CASE_CONFIGS.keys()),
        help="Training case to run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Number of steps per epoch",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (check simulator output)",
    )
    args = parser.parse_args()

    if args.test:
        test(case=args.case)
    else:
        main(
            case=args.case,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
        )
