import torch


def expand_right(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Expand x by appending n singleton dimensions on the right.
    Equivalent to x[..., None, None, ..., None] (n times).
    """
    if n < 0:
        raise ValueError(f"Cannot expand {n} times.")
    if n == 0:
        return x
    # build indexing tuple: (Ellipsis, None, None, ..., None)
    return x[(...,) + (None,) * n]


def expand_right_to(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Expand x so that x.ndim == dim, by appending singleton dims on the right.
    """
    return expand_right(x, dim - x.ndim)


def expand_right_as(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Expand x so that it has the same number of dimensions as y,
    by appending singleton dims on the right.
    """
    return expand_right_to(x, y.ndim)


def broadcast_right(x, shape):
    """Same semantics as torch.broadcast_to, but instead of automatically inserting unit dimensions on the left, it does so on the right."""
    x = torch.permute(x, list(range(x.ndim))[::-1])
    x = torch.broadcast_to(x, list(shape)[::-1])
    x = torch.permute(x, list(range(x.ndim))[::-1])
    return x
