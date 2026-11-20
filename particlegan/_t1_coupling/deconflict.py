"""Stateless activation-gram whitened gradient deconfliction for generator layers.

When disconnected modes share dense generator layers ``y_i = W h_i + b``,
post-LeakyReLU activations ``h_i`` carry a large common DC component and
cross-mode correlation, so ``G_W = sum_i delta_i h_i^T`` suffers destructive
interference across opposing mode pulls and drags already-settled modes off
target. Preconditioning in the whitened activation metric ``P = (Sigma_H +
lambda * mean_eig * I)^{-1/2}`` via ``polar(G_W P) P`` orthogonalizes mode
subspaces using pure FP32 coupled Newton-Schulz iterations with zero persistent
state, zero checkpoint overhead, and zero RNG draws.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn


def _exact_spd_inv_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """Exact deterministic ``mat^{-1/2}`` for small SPD ``mat`` (``d <= 128``) in FP32."""
    d = mat.shape[0]
    sym = 0.5 * (mat + mat.mT)
    evals, evecs = torch.linalg.eigh(sym)
    inv_sqrt = evals.clamp_min(1e-6).rsqrt()
    z = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.mT
    z = 0.5 * (z + z.mT)
    rms = (z.square().sum() / float(d)).clamp_min(1e-12).sqrt()
    return z / rms


def attach_generator_deconflict_hooks(module: nn.Module | None, *, max_in_features: int = 128) -> None:
    """Attach idempotent stateless activation capture hooks to eligible ``nn.Linear`` layers."""
    if not isinstance(module, nn.Module):
        return
    max_in = int(max_in_features)
    for child in module.modules():
        if (
            isinstance(child, nn.Linear)
            and 2 <= child.in_features <= max_in
            and child.weight.requires_grad
            and not getattr(child, "_t1c_deconflict_hook_registered", False)
        ):
            def _hook(mod: nn.Module, args, _output):
                if not mod.training or not torch.is_grad_enabled() or not args:
                    return
                if getattr(mod.weight, "_t1c_last_act", None) is not None:
                    # Keep primary training forward batch; ignore subsequent chord/probe passes.
                    return
                inp = args[0]
                if isinstance(inp, torch.Tensor) and inp.is_floating_point() and inp.numel() > 0:
                    h = inp.detach().reshape(-1, mod.in_features)
                    if h.shape[0] >= 8:
                        mod.weight._t1c_last_act = h

            child.register_forward_hook(_hook)
            child._t1c_deconflict_hook_registered = True


def clear_generator_deconflict_captures(parameters: Iterable[torch.Tensor]) -> None:
    """Clear transient per-step activation captures on generator weight tensors."""
    for p in parameters:
        if hasattr(p, "_t1c_last_act"):
            p._t1c_last_act = None


def deconflicted_polar_step(
    parameter: torch.Tensor,
    direction: torch.Tensor,
    polar_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    blend: float = 0.5,
    damping: float = 0.10,
) -> torch.Tensor | None:
    """Return ``polar(direction @ P_tilde) @ P_tilde`` if activation capture exists, else None."""
    h = getattr(parameter, "_t1c_last_act", None)
    if h is not None:
        parameter._t1c_last_act = None
    if blend <= 0.0 or parameter.ndim != 2 or h is None:
        return None
    if h.shape[1] != parameter.shape[1] or h.shape[0] < 8 or h.device != parameter.device:
        return None
    hf = h.to(dtype=torch.float32)
    b_n, d_in = hf.shape
    gram = (hf.mT @ hf) / float(b_n)
    mean_diag = torch.diagonal(gram).mean().clamp_min(1e-12)
    if not bool(torch.isfinite(mean_diag)):
        return None
    eye = torch.eye(d_in, device=parameter.device, dtype=torch.float32)
    lam = max(1e-3, float(damping))
    a_mat = (gram + lam * mean_diag * eye) / ((1.0 + lam) * mean_diag)
    p_inv = _exact_spd_inv_sqrt(a_mat)
    b_val = min(1.0, max(0.0, float(blend)))
    p_tilde = (1.0 - b_val) * eye + b_val * p_inv
    rms = (p_tilde.square().sum() / float(d_in)).clamp_min(1e-12).sqrt()
    p_tilde = (p_tilde / rms).to(dtype=direction.dtype)
    pre_dir = direction @ p_tilde
    u_mat = polar_fn(pre_dir)
    out = u_mat @ p_tilde
    if not bool(torch.isfinite(out).all()):
        return None
    return out
