"""Explicit inverse-consistency objective for caller-owned joint GAN hosts.

Categorical reconstructions enter as logits, so confident wrong predictions keep
a corrective cross-entropy derivative. Targets never receive gradients. This
training objective does not change a host's generator or sampling law.
"""
import torch
from torch.nn import functional as F


def joint_reconstruction_loss(reconstruction, target, *, categorical_dim=None):
    """Mean squared error, or mean categorical cross entropy from raw logits."""
    if (not isinstance(reconstruction, torch.Tensor) or not isinstance(target, torch.Tensor)
            or reconstruction.shape != target.shape or reconstruction.ndim < 2
            or len(reconstruction) < 1 or not reconstruction.is_floating_point()
            or reconstruction.device != target.device or reconstruction.dtype != target.dtype):
        raise ValueError("joint reconstruction requires matching nonempty floating panels")
    if categorical_dim is None:
        return F.mse_loss(reconstruction, target.detach())
    if (type(categorical_dim) is not int or categorical_dim == 0
            or not -reconstruction.ndim < categorical_dim < reconstruction.ndim):
        raise ValueError("categorical dimension must name a non-batch axis")
    axis = categorical_dim % reconstruction.ndim
    if axis == 0:
        raise ValueError("categorical dimension cannot be the batch axis")
    return -(target.detach() * reconstruction.log_softmax(axis)).sum(axis).mean()
