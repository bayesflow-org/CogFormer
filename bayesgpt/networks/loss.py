import torch


def nll_loss(true_params, mu, logvar, param_masks) -> torch.Tensor:
    """
    Heteroskedastic Gaussian NLL with masking.
    """

    mu = mu.squeeze(dim=-1)
    logvar = logvar.squeeze(dim=-1)

    inv_var = torch.exp(-logvar)
    nll = 0.5 * ((true_params - mu)**2 * inv_var + logvar)

    nll_masked = nll * param_masks
    denom = param_masks.sum(dim=-1)
    return torch.mean(nll_masked.sum(dim=-1) / denom)


def mse_loss(true_params, mu, param_masks) -> torch.Tensor:
    """
    Masked mean squared error (homoskedastic).
    Ignores log_var, uses only mu.
    """
    mu = mu.squeeze(dim=-1)
    true_params = true_params.squeeze(dim=-1)

    sq_err = (true_params - mu) ** 2

    # optional: normalize by param scale to avoid huge/small params dominating
    # eps = 1e-8
    # scale = true_params.abs().mean(dim=0, keepdim=True) + eps
    # sq_err = sq_err / (scale ** 2)

    sq_err_masked = sq_err * param_masks
    num_active_params = param_masks.sum(dim=-1)
    return torch.mean(sq_err_masked.sum(dim=-1) / num_active_params)