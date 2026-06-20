from cogformer.utils import paths
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import logging
import numpy as np
import pandas as pd
import bayesflow as bf
import matplotlib.pyplot as plt


from cogformer.simulators.model_family import NestedModelFamily
from cogformer.simulators.benchmarks.ddms.ddm import DDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.simulators.benchmarks.rdms.rdm import RDM
from cogformer.simulators.benchmarks.rdms.rdm_priors import rdm_priors
from cogformer.simulators.benchmarks.rdms.rdm_link_fun import rdm_link_fun
from cogformer.simulators.benchmarks.cdms.cdm import CDM
from cogformer.simulators.benchmarks.cdms.cdm_priors import cdm_priors
from cogformer.simulators.benchmarks.cdms.cdm_link_fun import cdm_link_fun
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import bf_colors


FAMILY_REGISTRY = {
    "ddm": {
        "name": "DDM",
        "model_cls": DDM,
        "prior_fun": ddm_priors,
        "link_fun": ddm_link_fun,
        "checkpoint_prefix": "ddm_families_bf",
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "a", "z", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "a", "z", "tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
            "full":           {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"]},
        },
        "case_configs": {
            "intercept_only": {
                "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
                "fixed_config": False, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"], "expected_num_active": None,
            },
            "fixed": {
                "free_intrinsics": ["v", "a", "z", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
                "regressed_params": None, "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": False, "squeeze_outputs": True,
                "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$"], "expected_num_active": 4,
            },
            "regressed": {
                "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": ["v", "a", "z"], "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$", r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$"],
                "expected_num_active": 12,
            },
            "fixed_regressed": {
                "free_intrinsics": ["v", "a", "z", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
                "regressed_params": ["v", "a", "z"], "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$", r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$"],
                "expected_num_active": 10,
            },
            "interaction": {
                "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": False,
                "design_config": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
                "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$", r"$u_{2, \tau}$", r"$u_1:u_{2, v}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, z}$"],
                "expected_num_active": 18,
            },
            "full": {
                "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True,
                "design_config": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"]},
                "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{1, s_\tau}$", r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$", r"$u_{2, \tau}$", r"$u_{2, s_v}$", r"$u_{2, s_\tau}$", r"$u_1:u_{2, v}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, z}$", r"$u_1:u_{2, \tau}$", r"$u_1:u_{2, s_v}$", r"$u_1:u_{2, s_\tau}$"],
                "expected_num_active": 24,
            },
        },
    },
    "rdm": {
        "name": "RDM",
        "model_cls": RDM,
        "prior_fun": rdm_priors,
        "link_fun": rdm_link_fun,
        "checkpoint_prefix": "rdm_families_bf",
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "v_diff", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
            "full":           {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"]},
        },
        "case_configs": {
            "intercept_only": {
                "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
                "fixed_config": True,
                "design_config": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"]},
                "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
                "expected_num_active": 6,
            },
            "fixed": {
                "free_intrinsics": ["v", "v_diff", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
                "regressed_params": None, "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": False, "squeeze_outputs": True,
                "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$"], "expected_num_active": 4,
            },
            "regressed": {
                "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": ["v_diff", "a"], "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$", r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$"],
                "expected_num_active": 10,
            },
            "fixed_regressed": {
                "free_intrinsics": ["v", "v_diff", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
                "regressed_params": ["v_diff", "a"], "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$", r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$"],
                "expected_num_active": 8,
            },
            "interaction": {
                "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": False,
                "design_config": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
                "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{2, v}$", r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$", r"$u_{2, \tau}$", r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_{\mathrm{diff}}}$", r"$u_1:u_{2, a}$"],
                "expected_num_active": 18,
            },
            "full": {
                "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True,
                "design_config": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"]},
                "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{1, s_\tau}$", r"$u_{2, v}$", r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$", r"$u_{2, \tau}$", r"$u_{2, s_v}$", r"$u_{2, s_\tau}$", r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_{\mathrm{diff}}}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, \tau}$", r"$u_1:u_{2, s_v}$", r"$u_1:u_{2, s_\tau}$"],
                "expected_num_active": 24,
            },
        },
    },
    "cdm": {
        "name": "CDM",
        "model_cls": CDM,
        "prior_fun": cdm_priors,
        "link_fun": cdm_link_fun,
        "checkpoint_prefix": "cdm_families_bf",
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "benchmark_design_configs": {
            "intercept_only": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed":      {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "fixed":          {"1": ["v", "v_theta", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed":{"1": ["v", "v_theta", "a", "tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "interaction":    {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v"], "u_2": ["v", "v_theta", "a", "tau"], "u_1:u_2": ["v", "v_theta", "a"]},
            "full":           {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"]},
        },
        "case_configs": {
            "intercept_only": {
                "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
                "fixed_config": False, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"], "expected_num_active": None,
            },
            "fixed": {
                "free_intrinsics": ["v", "v_theta", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
                "regressed_params": None, "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": False, "squeeze_outputs": True,
                "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$"], "expected_num_active": 4,
            },
            "regressed": {
                "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": ["v", "a"], "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{2, v}$", r"$u_{2, a}$"],
                "expected_num_active": 10,
            },
            "fixed_regressed": {
                "free_intrinsics": ["v", "v_theta", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
                "regressed_params": ["v", "a"], "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True, "design_config": None, "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{2, v}$", r"$u_{2, a}$"],
                "expected_num_active": 8,
            },
            "interaction": {
                "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": False,
                "design_config": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v"], "u_2": ["v", "v_theta", "a", "tau"], "u_1:u_2": ["v", "v_theta", "a"]},
                "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, v_\theta}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{2, v}$", r"$u_{2, v_\theta}$", r"$u_{2, a}$", r"$u_{2, \tau}$", r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_\theta}$", r"$u_1:u_{2, a}$"],
                "expected_num_active": 18,
            },
            "full": {
                "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "fixed_intrinsics": [], "fixed_values": {},
                "regressed_params": None, "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
                "fixed_config": True,
                "design_config": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"]},
                "flatten_param_outputs": True, "squeeze_outputs": False,
                "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$u_{1, v}$", r"$u_{1, v_\theta}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{1, s_\tau}$", r"$u_{2, v}$", r"$u_{2, v_\theta}$", r"$u_{2, a}$", r"$u_{2, \tau}$", r"$u_{2, s_v}$", r"$u_{2, s_\tau}$", r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_\theta}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, \tau}$", r"$u_1:u_{2, s_v}$", r"$u_1:u_{2, s_\tau}$"],
                "expected_num_active": 24,
            },
        },
    },
}


class ModelFamilyBF(bf.simulators.Simulator):

    def __init__(self, family: str, case: str):
        reg = FAMILY_REGISTRY[family]
        if case not in reg["case_configs"]:
            raise ValueError(f"Unknown case: {case}. Must be one of {list(reg['case_configs'].keys())}")

        self.case = case
        self.config = reg["case_configs"][case]
        self.link_fun = reg["link_fun"]

        self.model_family = NestedModelFamily(
            model=reg["model_cls"](),
            prior_fun=reg["prior_fun"](),
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

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=self.config["flatten_param_outputs"],
            link_fun=self.link_fun(),
            **sample_kwargs,
            **kwargs
        )

        param_matrices = samples["param_matrices"]
        param_masks = samples["param_masks"]

        if self.config["squeeze_outputs"]:
            param_matrices = param_matrices.squeeze(axis=1)
            param_masks = param_masks.squeeze(axis=1)

        return {
            "design_matrices": samples["design_matrices"],
            "rts": samples["sim_data"]["rts"],
            "choices": samples["sim_data"]["choices"],
            "params": param_matrices,
            "masks": param_masks,
        }


def reshape_bf_to_gpt(bf_samples, design_config, intrinsic_params):
    *leading, num_active = bf_samples.shape
    num_rows = len(design_config)
    num_cols = len(intrinsic_params)
    col_idx = {p: j for j, p in enumerate(intrinsic_params)}
    result = np.zeros((*leading, num_rows, num_cols))
    flat_pos = 0
    for row_i, active_params in enumerate(design_config.values()):
        ordered = [p for p in intrinsic_params if p in active_params]
        for p in ordered:
            result[..., row_i, col_idx[p]] = bf_samples[..., flat_pos]
            flat_pos += 1
    assert flat_pos == num_active, (
        f"Expected {num_active} active params but mapped {flat_pos}."
    )
    return result


def test(family: str, case: str):
    reg = FAMILY_REGISTRY[family]
    config = reg["case_configs"][case]
    simulator = ModelFamilyBF(family=family, case=case)

    samples = simulator.sample(4)
    masks = samples["masks"]
    params = samples["params"]

    num_active = int(masks[0].astype(bool).sum())
    expected = config["expected_num_active"]
    if expected is not None:
        status = "OK" if num_active == expected else f"FAIL (expected {expected})"
        print(f"  [Check] Active params: {num_active}  →  {status}")
        assert num_active == expected, f"Active param count: got {num_active}, expected {expected}"
    else:
        print(f"  [Check] Active params: {num_active}  →  (variable, no assertion)")

    if config["fixed_config"] or config["design_config"] is not None:
        consistent = np.all(masks == masks[0])
        print(f"  [Check] Mask consistent across batch: {'OK' if consistent else 'FAIL'}")
        assert consistent, "Mask differs across batch items for a fixed-config case"

    zero_where_masked = np.allclose(params * (1 - masks), 0.0)
    print(f"  [Check] Params zero where masked: {'OK' if zero_where_masked else 'FAIL'}")
    assert zero_where_masked, "Non-zero param values found at masked-out positions"

    checkpoint_path = str(paths.checkpoints_dir("bf", f"{reg['checkpoint_prefix']}_{case}", "model.keras"))
    approximator = keras.saving.load_model(checkpoint_path)
    print("Loaded model")

    val_sims = simulator.sample(100)
    conditions = {
        "rts": val_sims["rts"],
        "choices": val_sims["choices"],
        "design_matrices": val_sims["design_matrices"],
    }
    targets = val_sims["params"]
    post_draws = approximator.sample(conditions=conditions, num_samples=100)
    estimates = post_draws["params"]
    print(f"Targets shape: {targets.shape}, Estimates shape: {estimates.shape}")

    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)
    true_params = targets[:, active_idx]
    pred_params = estimates[:, :, active_idx]
    print(f"Active true_params shape: {true_params.shape}, pred_params shape: {pred_params.shape}")


def main(family: str, case: str, batch_size: int = 200, num_samples: int = 200, skip_posteriors: bool = False, skip_log_gamma: bool = True):
    reg = FAMILY_REGISTRY[family]
    config = reg["case_configs"][case]
    param_names = config["param_names"]
    fam_lower = reg["name"].lower()

    simulator = ModelFamilyBF(family=family, case=case)

    checkpoint_path = str(paths.checkpoints_dir("bf", f"{reg['checkpoint_prefix']}_{case}", "model.keras"))
    approximator = keras.saving.load_model(checkpoint_path)

    data_dir = paths.data_dir("predictions")
    figures_dir = paths.figures_dir("model_family", "bf", fam_lower, case)
    evals_dir = paths.tables_dir("evaluations")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    val_sims = simulator.sample(batch_size)
    conditions = {
        "rts": val_sims["rts"],
        "choices": val_sims["choices"],
        "design_matrices": val_sims["design_matrices"],
    }
    targets = val_sims["params"]
    post_draws = approximator.sample(conditions=conditions, num_samples=num_samples)
    estimates = post_draws["params"]

    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)
    true_params = targets[:, active_idx]
    pred_params = estimates[:, :, active_idx]
    if len(param_names) == len(active_idx):
        param_names = [name for name, active in zip(param_names, active_idx) if active]

    true_set = val_sims["params"]
    pred_set = post_draws["params"]

    np.savez(
        data_dir / f"{fam_lower}_{case}_data.npz",
        rts=val_sims["rts"],
        choices=val_sims["choices"],
        true_set=true_set,
        pred_set=pred_set,
        design_matrices=val_sims["design_matrices"],
        param_masks=val_sims["masks"],
    )
    logging.info(f"Saved data to {data_dir}")

    rmse = bf.diagnostics.metrics.root_mean_squared_error(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )
    calibration_errors = bf.diagnostics.metrics.calibration_error(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )
    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=pred_params, targets=true_params, variable_names=param_names
    )

    metrics_dict = {
        rmse["metric_name"]: rmse["values"],
        calibration_errors["metric_name"]: calibration_errors["values"],
        contraction["metric_name"]: contraction["values"],
    }
    if not skip_log_gamma:
        log_gamma = bf.diagnostics.metrics.calibration_log_gamma(
            estimates=pred_params, targets=true_params, variable_names=param_names
        )
        metrics_dict[log_gamma["metric_name"]] = log_gamma["values"]

    metrics = pd.DataFrame(metrics_dict)
    metrics.to_csv(evals_dir / f"{fam_lower}_families_bf_{case}_evaluations.csv", sep=";")
    logging.info("Metric evaluation is now finished.")

    intrinsic_params_all = reg["intrinsic_params"]
    variable_names_all = reg["variable_names"]
    design_config = reg["benchmark_design_configs"][case]
    adaptive_colors = bf_colors()

    true_grid = reshape_bf_to_gpt(true_params, design_config, intrinsic_params_all)
    pred_grid = reshape_bf_to_gpt(pred_params, design_config, intrinsic_params_all)
    params_mask = reshape_bf_to_gpt(
        np.ones((1, true_params.shape[-1])), design_config, intrinsic_params_all
    )[0]

    recovery_fig = adaptive_recovery(
        true=true_grid, pred=pred_grid, design_config=design_config,
        intrinsic_params=intrinsic_params_all, max_num_categories=2,
        parameter_mask=params_mask, variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"], main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
    )
    recovery_fig.savefig(figures_dir / f"{fam_lower}_family_{case}_bf_recovery.pdf", bbox_inches="tight")
    plt.close(recovery_fig)

    coverage_fig = adaptive_coverage(
        true=true_grid, pred=pred_grid, design_config=design_config,
        intrinsic_params=intrinsic_params_all, max_num_categories=2,
        parameter_mask=params_mask, variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"], main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
    )
    coverage_fig.savefig(figures_dir / f"{fam_lower}_family_{case}_bf_coverage.pdf", bbox_inches="tight")
    plt.close(coverage_fig)

    ecdf_fig = adaptive_ecdf(
        true=true_grid, pred=pred_grid, design_config=design_config,
        intrinsic_params=intrinsic_params_all, max_num_categories=2,
        parameter_mask=params_mask, variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"], main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"], difference=True,
    )
    ecdf_fig.savefig(figures_dir / f"{fam_lower}_family_{case}_bf_ecdf.pdf", bbox_inches="tight")
    plt.close(ecdf_fig)

    metrics_fig = plot_adaptive_metrics(
        true=true_grid, pred=pred_grid, design_config=design_config,
        intrinsic_params=intrinsic_params_all, max_num_categories=2,
        parameter_mask=params_mask, variable_names=variable_names_all,
        intercept_color=adaptive_colors["intercept"], main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"], skip_log_gamma=skip_log_gamma,
    )
    metrics_fig.savefig(figures_dir / f"{fam_lower}_family_{case}_bf_metrics.pdf", bbox_inches="tight")
    plt.close(metrics_fig)

    metrics_df = compute_adaptive_metrics(
        true=true_grid, pred=pred_grid, design_config=design_config,
        intrinsic_params=intrinsic_params_all, max_num_categories=2,
        parameter_mask=params_mask, variable_names=variable_names_all,
        skip_log_gamma=skip_log_gamma,
    )
    metrics_df.to_csv(paths.metrics_mirror(figures_dir, make=True) / f"{fam_lower}_family_{case}_bf_metrics.csv")

    if not skip_posteriors:
        for i in range(10):
            posterior_fig = adaptive_posterior(
                samples=pred_grid[i], design_config=design_config,
                intrinsic_params=intrinsic_params_all, max_num_categories=2, unfold=False,
                intercept_color=adaptive_colors["intercept"], main_effect_color=adaptive_colors["main_effect"],
                interaction_color=adaptive_colors["interaction"],
            )
            posterior_fig.savefig(
                figures_dir / f"{fam_lower}_family_{case}_bf_posterior_{i}.pdf", bbox_inches="tight"
            )
            plt.close(posterior_fig.fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Family BayesFlow Validation")
    parser.add_argument("--model_family", type=str, required=True, choices=list(FAMILY_REGISTRY.keys()))
    parser.add_argument("--case", type=str, default="intercept_only", help="Validation case to run")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_posteriors", action="store_true")
    parser.add_argument("--skip_log_gamma", action="store_true", default=True)
    parser.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")
    args = parser.parse_args()

    reg = FAMILY_REGISTRY[args.model_family]
    valid_cases = list(reg["case_configs"].keys())
    if args.case not in valid_cases:
        parser.error(f"Invalid case '{args.case}' for {args.model_family}. Choose from: {valid_cases}")

    if args.test:
        test(family=args.model_family, case=args.case)
    else:
        main(
            family=args.model_family,
            case=args.case,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            skip_posteriors=args.skip_posteriors,
            skip_log_gamma=args.skip_log_gamma,
        )
