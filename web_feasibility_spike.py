"""
Latency-knee sweep for the BayesGPT web demo backend.

Loads the DDM checkpoint once, simulates a single dataset (one web request),
then sweeps (ODE steps x posterior draws) to find a config whose wall time is
acceptable for a live web UX. Reports recovery quality at each point so we can
see the speed/accuracy tradeoff.

Run: python web_feasibility_spike.py
"""
import os
import sys
import time
import contextlib
import numpy as np
import torch

from cogformer.simulators import NestedModelFamily
from cogformer.simulators.benchmarks.ddms.ddm import DDM
from cogformer.simulators.benchmarks.ddms.ddm_priors import ddm_priors
from cogformer.simulators.benchmarks.ddms.ddm_link_fun import ddm_link_fun
from cogformer.adapters import Adapter
from cogformer.networks.transformers.cf.cogformer import CogFormer

CKPT = "outputs/checkpoints/model_family/ddm/cogformer_mf_l8_h8_p256_s32_d64_o500_b64_e5000_t100.pt"
INTRINSIC = ["v", "a", "z", "tau", "s_v", "s_tau"]
DESIGN_CONFIG = {
    "1":       ["v", "a", "z", "tau", "s_v", "s_tau"],
    "u_1":     ["v", "a", "z"],
    "u_2":     ["v", "a", "z"],
    "u_1:u_2": [],
}
MAX_NUM_REGRESSORS, MAX_NUM_CATEGORIES, KEEP_INTERCEPT = 2, 2, True

# (steps, draws) operating points to probe, cheapest first
SWEEP = [(30, 100), (50, 100), (50, 200), (100, 100), (100, 200), (200, 200)]


def build_encoder_input_dim(mr, mc, ki):
    return (mr * (mr + 1) // 2) * (mc - 1) + (3 if ki else 2)


@torch.no_grad()
def main():
    torch.set_num_threads(os.cpu_count())
    print(f"CPU cores={os.cpu_count()} | torch threads={torch.get_num_threads()}\n")
    device = torch.device("cpu")
    torch.manual_seed(0); np.random.seed(0)

    model_family = NestedModelFamily(
        model=DDM(), name="DDM", prior_fun=ddm_priors(),
        mask_randomizer_kwargs={"free_intrinsics": INTRINSIC, "fixed_intrinsics": [], "fixed_values": {}},
    )
    adapter = Adapter()
    cogformer = CogFormer(
        encoder_input_dim=build_encoder_input_dim(MAX_NUM_REGRESSORS, MAX_NUM_CATEGORIES, KEEP_INTERCEPT),
        encoder_num_layers=8, decoder_num_layers=8, encoder_num_heads=8, decoder_num_heads=8,
        num_seeds=32, seed_dim=64, proj_dim=256, dropout=0.05, layer_dropout=0.05,
        decoder_layer_design="mixed_attention", decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=32, pos_embedding_dim=32,
    ).to(device)
    cogformer.load_state_dict(torch.load(CKPT, map_location=device))
    cogformer.eval()

    # one dataset = one web request
    samples = model_family.batch_sample(
        batch_size=1, num_obs=500,
        max_num_regressors=MAX_NUM_REGRESSORS, max_num_categories=MAX_NUM_CATEGORIES,
        keep_intercept=KEEP_INTERCEPT, add_interaction=True,
        design_config=DESIGN_CONFIG, fixed_config=True,
        flatten_param_outputs=True, link_fun=ddm_link_fun(),
        mask_randomizer_kwargs={"free_intrinsics": ["v", "a", "z", "tau"],
                                "fixed_intrinsics": ["s_v", "s_tau"],
                                "fixed_values": {"s_v": 0.0, "s_tau": 0.0}},
        min_num_regressors=0,
    )
    adapted = adapter.adapt(samples, intrinsic_params=INTRINSIC)
    for k, v in adapted.items():
        if torch.is_tensor(v):
            adapted[k] = v.to(device)
    n_cols = len(INTRINSIC)
    true_set = adapted["param_matrices"].cpu().numpy()
    n_rows = true_set.shape[1] // n_cols
    true_set = true_set.reshape(1, n_rows, n_cols)
    pmask = adapted["param_masks"].cpu().numpy().reshape(1, n_rows, n_cols)[0].astype(bool)

    print(f"{'steps':>6} {'draws':>6} {'wall(s)':>9} {'recov.corr':>11}")
    print("-" * 36)
    for steps, draws in SWEEP:
        t0 = time.perf_counter()
        with contextlib.redirect_stderr(open(os.devnull, "w")):  # silence tqdm
            pred = cogformer.sample(
                adapted["input_data"], adapted["param_indices"],
                adapted["regressor_indices"], adapted["param_masks"],
                steps=steps, num_samples=draws,
            )
        wall = time.perf_counter() - t0
        pred = pred.reshape(1, draws, n_rows, n_cols)
        post_mean = pred.mean(axis=1)
        corr = np.corrcoef(true_set[:, pmask].ravel(), post_mean[:, pmask].ravel())[0, 1]
        print(f"{steps:>6} {draws:>6} {wall:>9.2f} {corr:>11.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
