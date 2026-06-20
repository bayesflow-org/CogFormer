"""Smoke tests: verify the package imports and the core model runs a forward pass.

These are intentionally lightweight (CPU, tiny dimensions) so they can run in CI
without data, GPUs, or checkpoints. They exist to catch packaging/import breakage
and gross API regressions (e.g. a cleanup that removes a module still in use).
"""

import torch


def test_public_api_imports():
    # The package's advertised entry point.
    from cogformer.networks import CogFormer
    from cogformer.networks.transformers.cf.cogformer import CogFormer as CogFormerDirect

    assert CogFormer is CogFormerDirect


def test_subpackages_import():
    import cogformer.simulators  # noqa: F401
    import cogformer.diagnostics  # noqa: F401
    import cogformer.adapters  # noqa: F401
    import cogformer.utils  # noqa: F401


def _tiny_model():
    from cogformer.networks import CogFormer

    return CogFormer(
        encoder_input_dim=4,
        proj_dim=16,
        encoder_num_layers=1,
        decoder_num_layers=1,
        encoder_num_heads=2,
        decoder_num_heads=2,
        num_seeds=2,
        seed_dim=16,
        time_embedding_dim=8,
        pos_embedding_dim=8,
    )


def test_cogformer_instantiates():
    model = _tiny_model()
    assert isinstance(model, torch.nn.Module)


def test_cogformer_forward_and_loss():
    torch.manual_seed(0)
    model = _tiny_model()

    batch, num_obs, num_tokens = 2, 5, 3
    input_data = torch.randn(batch, num_obs, 4)            # (B, N, encoder_input_dim)
    params = torch.randn(batch, num_tokens, 1)            # theta: (B, T, 1)
    param_indices = torch.randn(batch, num_tokens, 1)     # (B, T, 1)
    regressor_indices = torch.randn(batch, num_tokens, 1)  # (B, T, 1)
    params_mask = torch.ones(batch, num_tokens)           # (B, T)

    pred_velocity, target_velocity = model(
        params, input_data, param_indices, regressor_indices, params_mask=params_mask
    )

    assert pred_velocity.shape == target_velocity.shape
    assert pred_velocity.shape[0] == batch

    loss = model.compute_loss(pred_velocity, target_velocity, params_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
