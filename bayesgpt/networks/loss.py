import torch


def loss(true_params, mu, logvar, param_masks) -> torch.Tensor:
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
