"""Pure 1st-Order Spatial-Frame Forward-Tangent & Multi-Scale Secant Born-Infeld Critic (Track 2).

Eliminates all ``torch.autograd.grad(..., create_graph=True)`` second-order
double-backpropagation graphs (and calls ``torch.autograd.grad`` zero times):

1. **Spatial-Frame Forward-Mode Tangent Reconstruction (Manifold / MLP / Vector /
   Discrete-Simplex / BiGAN / VAE / DDGAN Critics)**:
   Constructs a deterministic, RNG-free, Weyl-rotated orthonormal frame
   ``Q(phi) in O(d)`` (for ``d <= 8``) or Murmur32 Rademacher frame (for
   ``d > 8``), assigns one unit tangent ``u_i`` per sample along a space-filling
   projection ordering, and propagates ``(D(x_i), nabla_x D(x_i) . u_i)`` in a
   single forward-mode tangent pass (``torch.autograd.forward_ad.dual_level``;
   automatic ``delta = 1e-3`` finite-difference secant fallback for custom
   emulated-precision ops without ``jvp`` rules). Reconstructs the Euclidean
   input-gradient norm ``||nabla_x D(x_i)||_2`` across local basis-slot
   neighbors (with exact ``0`` directional variance for ``d <= 8`` and local
   Hutchinson k-NN pooling for ``d > 8``), trained via a single standard
   1st-order ``.backward()`` pass.

2. **Live-Endpoint Reuse + 1st-Order Secant Envelope (BF16 Convolutional Image
   Critics)**:
   Reuses the live unscaled 1st-order ``D(x_real)`` and ``D(x_fake)`` logits
   from ``d_loss`` at zero extra endpoint forward cost, evaluates one detached
   BF16 micro-secant probe batch alongside free transport-chord and cyclic
   cross-sample secants, and tracks image scale via spatially pooled ``8x8``
   moments in ``O(B^2)`` scalar time.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._empf.prototypes import _PHI1, _PHI2, update_emeans_prototypes
from .._empf.transport import (
    conditional_fiber_residual,
    conditional_joint_discrepancy,
    effective_sinkhorn_eps,
    mass_weighted_sinkhorn,
    unconditioned_chamfer_residual,
)
from .._stats import pairwise_sq_dist
from ..grad_regularizers import CriticStepRecord
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output, _no_anchor
from .critic import (
    FROZEN_THERMAL_SCALE,
    T2SecantCriticAdam,
    T2SecantStepRecord,
    _attach_t2_critic_hook,
    _cond_signature,
    _tensor_identity_key,
    compute_empf_step_scale,
    compute_thermal_scale,
)


def _format_logits_2d(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return ``[B, K]`` per-sample logits (``[B, 1]`` for scalar outputs)."""
    if logits.ndim < 2:
        return logits.reshape(batch_size, 1)
    return logits.reshape(batch_size, -1)


def _logit_diff_rms(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-pair output difference ``|u - v|`` for scalar heads, or channel-RMS ``||u - v||_2 / sqrt(K)``."""
    diff = u - v
    k = diff.shape[-1]
    if k <= 1:
        return diff.squeeze(-1).abs()
    return (diff.square().mean(dim=-1) + 1e-12).sqrt()


def _murmur32_signs(
    row_indices: torch.Tensor,
    num_cols: int,
    phase: float,
    salt: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic ``[R, C]`` Rademacher ``+-1`` matrix using 32-bit Murmur avalanche (zero RNG state)."""
    p_int = (int(phase * 1073741824.0) + salt * 7919) & 0xFFFFFFFF
    r = row_indices.to(device=device, dtype=torch.int64).unsqueeze(1)
    c = torch.arange(num_cols, device=device, dtype=torch.int64).unsqueeze(0)
    h = (r * 2654435761 + c * 2246822519 + p_int * 3266489917 + 374761393) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return torch.where((h & 1) == 0, 1.0, -1.0)


def _murmur32_uniform(
    num_rows: int,
    num_cols: int,
    phase: float,
    salt: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic ``[R, C]`` uniform ``[-1, 1]`` matrix (zero RNG state)."""
    p_int = (int(phase * 1073741824.0) + salt * 7919) & 0xFFFFFFFF
    r = torch.arange(num_rows, device=device, dtype=torch.int64).unsqueeze(1)
    c = torch.arange(num_cols, device=device, dtype=torch.int64).unsqueeze(0)
    h = (r * 2654435761 + c * 2246822519 + p_int * 3266489917 + 374761393) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return (h.to(dtype=torch.float32) / 2147483648.0) - 1.0


def _build_orthonormal_frame(
    dim: int,
    k: int,
    phase: float,
    salt: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic ``[k, dim]`` orthonormal frame in ``O(dim)`` without CPU-GPU sync or cuSOLVER."""
    if dim == 1:
        return torch.ones((1, 1), device=device, dtype=torch.float32)
    if dim == 2 and k == 2:
        theta = 2.0 * math.pi * phase + 0.5 * float(salt)
        c_t, s_t = math.cos(theta), math.sin(theta)
        idx = torch.arange(4, device=device)
        vals = torch.where(
            idx == 0,
            c_t,
            torch.where(idx == 1, s_t, torch.where(idx == 2, -s_t, c_t)),
        )
        return vals.to(dtype=torch.float32).reshape(2, 2)
    # Two chained analytical Householder reflections Q = H_2 H_1 in O(dim) (zero QR / cuSOLVER sync).
    v = _murmur32_uniform(2, dim, phase, salt, device)
    e0 = (torch.arange(dim, device=device) == 0).to(dtype=torch.float32)
    v = v + 0.25 * e0.unsqueeze(0)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
    v0, v1 = v[0], v[1]
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    h1 = eye - 2.0 * torch.outer(v0, v0)
    q = h1 - 2.0 * torch.outer(v1, v1 @ h1)
    return q[:k].contiguous()


def _separable_image_unit_perturbation(x: torch.Tensor, phase: float, salt: int = 0) -> torch.Tensor:
    """Fast ``O(B*(C*H + W))`` deterministic Rademacher unit perturbation for ``[B, C, H, W]`` images."""
    b = x.shape[0]
    if x.ndim < 4:
        flat_d = max(1, x[0].numel())
        ranks = torch.arange(b, device=x.device, dtype=torch.int64)
        u = _murmur32_signs(ranks, flat_d, phase, salt, x.device) / math.sqrt(float(flat_d))
        return u.to(dtype=x.dtype).reshape_as(x)
    _, c, h, w = x.shape
    ch = c * h
    ranks = torch.arange(b, device=x.device, dtype=torch.int64)
    s_ch = _murmur32_signs(ranks, ch, phase, salt, x.device).reshape(b, c, h, 1)
    s_w = _murmur32_signs(ranks + 131, w, phase, salt + 17, x.device).reshape(b, 1, 1, w)
    scale = 1.0 / math.sqrt(float(max(1, ch * w)))
    return ((s_ch * s_w) * scale).to(dtype=x.dtype)


def _prepare_spatial_tangent_plan(
    x_det: torch.Tensor,
    phase: float,
    salt: int,
    *,
    cond_tensors: Sequence[torch.Tensor] = (),
    pool_k: int = 32,
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Build unit tangent ``u`` (matching ``x_det``) and 1st-order norm-squared reconstructor."""
    b = x_det.shape[0]
    flat = x_det.flatten(1).float()
    dim = max(1, flat.shape[1])

    if dim == 1 or b < 2:
        if dim == 1:
            u = torch.ones_like(x_det)
        else:
            u_f = _murmur32_signs(torch.arange(b, device=x_det.device), dim, phase, salt, x_det.device) / math.sqrt(
                float(dim)
            )
            u = u_f.to(dtype=x_det.dtype).reshape_as(x_det)

        def reconstruct_1d(tau_sq: torch.Tensor) -> torch.Tensor:
            return float(dim) * tau_sq

        return u, reconstruct_1d

    with torch.no_grad():
        cond_d2: Optional[torch.Tensor] = None
        if cond_tensors:
            c_parts = [
                t.detach().reshape(b, -1).float()
                for t in cond_tensors
                if isinstance(t, torch.Tensor) and t.shape[0] == b and t.numel() > 0
            ]
            if c_parts:
                cond_d2 = pairwise_sq_dist(torch.cat(c_parts, dim=1))

        if dim <= 8:
            k = min(dim, b)
            frame = _build_orthonormal_frame(dim, k, phase, salt, x_det.device)
            w_proj = frame[0] + (0.61803398875 * frame[1] if k > 1 else 0.0)
            proj = flat @ w_proj
            if cond_d2 is not None and c_parts:
                proj = proj + 3.14159265 * c_parts[0][:, 0]
            order = torch.argsort(proj, stable=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(b, device=x_det.device)
            slot = ranks % k
            u = frame[slot].to(dtype=x_det.dtype).reshape_as(x_det)

            d2 = pairwise_sq_dist(flat)
            if cond_d2 is not None:
                d2 = d2 + 10.0 * cond_d2
            slot_eq = slot.unsqueeze(1) == torch.arange(k, device=x_det.device).unsqueeze(0)
            d2_masked = d2.unsqueeze(2).masked_fill(~slot_eq.unsqueeze(0), float("inf"))
            nn_idx = d2_masked.argmin(dim=1)  # [B, k]
            scale_factor = float(dim) / float(k)

            def reconstruct_frame(tau_sq: torch.Tensor) -> torch.Tensor:
                return scale_factor * tau_sq[nn_idx].sum(dim=1)

            return u, reconstruct_frame

        # High-dimensional non-image tensors (e.g. 170D discrete-simplex BiGAN):
        # Independent Murmur32 Rademacher unit tangents + local k-NN Hutchinson pooling.
        proj = flat[:, 0] + 0.61803398875 * flat[:, 1]
        order = torch.argsort(proj, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(b, device=x_det.device)
        u_flat = _murmur32_signs(ranks, dim, phase, salt, x_det.device) / math.sqrt(float(dim))
        u = u_flat.to(dtype=x_det.dtype).reshape_as(x_det)

        k_eff = min(pool_k, b)
        d2 = pairwise_sq_dist(flat)
        if cond_d2 is not None:
            d2 = d2 + 10.0 * cond_d2
        knn_d2, knn_idx = torch.topk(d2, k=k_eff, dim=1, largest=False)
        bw2 = knn_d2[:, max(1, k_eff // 2)].clamp_min(1e-5).unsqueeze(1)
        weights = torch.exp(-0.5 * knn_d2 / bw2)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def reconstruct_hutchinson(tau_sq: torch.Tensor) -> torch.Tensor:
        return float(dim) * (weights.to(dtype=tau_sq.dtype) * tau_sq[knn_idx]).sum(dim=1)

    return u, reconstruct_hutchinson


def _eval_exact_lowd_forward_grad_sq(
    live_rep: Callable[[torch.Tensor, int], torch.Tensor],
    x_item: torch.Tensor,
    *,
    can_stack_basis: bool,
    fd_delta: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact ``||nabla_x D(x)||_2^2`` for ``1 <= dim <= 4`` via canonical basis forward-mode AD (zero autograd.grad)."""
    x_det = x_item.detach()
    b = int(x_det.shape[0])
    dim = max(1, x_det[0].numel())
    tail_ones = (1,) * (x_det.ndim - 1)

    if dim == 1:
        u1 = torch.ones_like(x_det)
        try:
            with fwAD.dual_level():
                yd = live_rep(fwAD.make_dual(x_det, u1), 1)
                y_1, jvp_1 = fwAD.unpack_dual(yd)
            if jvp_1 is None:
                raise NotImplementedError("dual tangent dropped")
            return _format_logits_2d(y_1, b), _format_logits_2d(jvp_1, b).square().mean(dim=-1)
        except (RuntimeError, NotImplementedError, TypeError):
            y_p = _format_logits_2d(live_rep((x_det + fd_delta * u1).detach(), 1), b)
            y_m = _format_logits_2d(live_rep((x_det - fd_delta * u1).detach(), 1), b)
            y_mid = 0.5 * (y_p + y_m)
            return y_mid, (y_p - y_m).square().mean(dim=-1) / (4.0 * fd_delta * fd_delta)

    eye = torch.eye(dim, device=x_det.device, dtype=x_det.dtype)
    if can_stack_basis:
        x_rep = x_det.repeat((dim,) + tail_ones)
        u_rep = eye.repeat_interleave(b, dim=0).reshape_as(x_rep)
        try:
            with fwAD.dual_level():
                yd = live_rep(fwAD.make_dual(x_rep, u_rep), dim)
                y_rep, jvp_rep = fwAD.unpack_dual(yd)
            if jvp_rep is None:
                raise NotImplementedError("dual tangent dropped")
            y_2d = _format_logits_2d(y_rep[:b], b)
            jvp_sq = _format_logits_2d(jvp_rep, dim * b).square().mean(dim=-1).reshape(dim, b).sum(dim=0)
            return y_2d, jvp_sq
        except (RuntimeError, NotImplementedError, TypeError):
            y_p = _format_logits_2d(live_rep((x_rep + fd_delta * u_rep).detach(), dim), dim * b)
            y_m = _format_logits_2d(live_rep((x_rep - fd_delta * u_rep).detach(), dim), dim * b)
            y_mid = 0.5 * (y_p + y_m)
            diff_sq = ((y_p - y_m).square().mean(dim=-1) / (4.0 * fd_delta * fd_delta)).reshape(dim, b).sum(dim=0)
            return y_mid[:b], diff_sq

    # Per-axis loop when BatchNorm / InputNoise / batch-state requires exact batch size B:
    y_first: Optional[torch.Tensor] = None
    sq_accum = x_det.new_zeros((b,), dtype=torch.float32)
    for axis in range(dim):
        u_axis = eye[axis].unsqueeze(0).expand(b, dim).reshape_as(x_det).contiguous()
        try:
            with fwAD.dual_level():
                yd = live_rep(fwAD.make_dual(x_det, u_axis), 1)
                y_a, jvp_a = fwAD.unpack_dual(yd)
            if jvp_a is None:
                raise NotImplementedError("dual tangent dropped")
            y_2d_a = _format_logits_2d(y_a, b)
            part_sq = _format_logits_2d(jvp_a, b).square().mean(dim=-1)
        except (RuntimeError, NotImplementedError, TypeError):
            y_p = _format_logits_2d(live_rep((x_det + fd_delta * u_axis).detach(), 1), b)
            y_m = _format_logits_2d(live_rep((x_det - fd_delta * u_axis).detach(), 1), b)
            y_2d_a = 0.5 * (y_p + y_m)
            part_sq = (y_p - y_m).square().mean(dim=-1) / (4.0 * fd_delta * fd_delta)
        if y_first is None:
            y_first = y_2d_a
        sq_accum = sq_accum + part_sq.float()
    assert y_first is not None
    return y_first, sq_accum.to(dtype=y_first.dtype)


def _eval_forward_tangent_sq(
    live: Callable[[torch.Tensor], torch.Tensor],
    x_list: Sequence[torch.Tensor],
    u_list: Sequence[torch.Tensor],
    *,
    can_concat: bool,
    fd_delta: float = 1e-3,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Evaluate ``(logits_2d, directional_deriv_sq)`` via 1st-order forward AD (or central-diff fallback)."""
    sizes = [int(x.shape[0]) for x in x_list]
    if can_concat and len(x_list) > 1:
        x_cat = torch.cat([x.detach() for x in x_list], dim=0)
        u_cat = torch.cat(list(u_list), dim=0)
        try:
            with fwAD.dual_level():
                yd = live(fwAD.make_dual(x_cat, u_cat))
                y_cat, jvp_cat = fwAD.unpack_dual(yd)
            if jvp_cat is None:
                raise NotImplementedError("dual tangent dropped by custom operator")
            y_2d = _format_logits_2d(y_cat, sum(sizes))
            tau_sq = _format_logits_2d(jvp_cat, sum(sizes)).square().mean(dim=-1)
            return list(torch.split(y_2d, sizes, dim=0)), list(torch.split(tau_sq, sizes, dim=0))
        except (RuntimeError, NotImplementedError, TypeError):
            y_p = _format_logits_2d(live((x_cat + fd_delta * u_cat).detach()), sum(sizes))
            y_m = _format_logits_2d(live((x_cat - fd_delta * u_cat).detach()), sum(sizes))
            y_mid = 0.5 * (y_p + y_m)
            step_sq = (2.0 * fd_delta * u_cat.flatten(1).float()).square().sum(dim=1).clamp_min(1e-12)
            tau_sq = (y_p - y_m).square().mean(dim=-1) / step_sq.to(dtype=y_mid.dtype)
            return list(torch.split(y_mid, sizes, dim=0)), list(torch.split(tau_sq, sizes, dim=0))

    logits_out: List[torch.Tensor] = []
    tau_out: List[torch.Tensor] = []
    for x_item, u_item, b_sz in zip(x_list, u_list, sizes):
        x_det = x_item.detach()
        try:
            with fwAD.dual_level():
                yd = live(fwAD.make_dual(x_det, u_item))
                y_i, jvp_i = fwAD.unpack_dual(yd)
            if jvp_i is None:
                raise NotImplementedError("dual tangent dropped by custom operator")
            y_2d = _format_logits_2d(y_i, b_sz)
            t_sq = _format_logits_2d(jvp_i, b_sz).square().mean(dim=-1)
        except (RuntimeError, NotImplementedError, TypeError):
            y_p = _format_logits_2d(live((x_det + fd_delta * u_item).detach()), b_sz)
            y_m = _format_logits_2d(live((x_det - fd_delta * u_item).detach()), b_sz)
            y_2d = 0.5 * (y_p + y_m)
            step_sq = (2.0 * fd_delta * u_item.flatten(1).float()).square().sum(dim=1).clamp_min(1e-12)
            t_sq = (y_p - y_m).square().mean(dim=-1) / step_sq.to(dtype=y_2d.dtype)
        logits_out.append(y_2d)
        tau_out.append(t_sq)
    return logits_out, tau_out


class _T2SecantPenaltyKernel:
    """Clock-free 1st-order Spatial-Frame Forward-Tangent & Secant Born-Infeld critic kernel."""

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        lambda_floor: float = 0.01,
        lambda_cap: float = 1.0,
        ebi_beta: float = 6.0,
        gamma_gauge: float = 0.050,
        chord_blend: float = 0.06,
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        max_prototypes: int = 256,
        alpha_base: float = 0.08,
        hit_exponent: float = 0.45,
        revival_tau: float = 0.05,
        tau_cooling: float = 0.020,
        s_min: float = 1e-5,
        thermal_decay: float = 0.92,
        secant_eps: float = 1e-5,
        input_domain: str = "auto",
        record: CriticStepRecord | None = None,
    ) -> None:
        for name, val, pos in (
            ("coeff", coeff, False),
            ("kappa", kappa, False),
            ("lambda_floor", lambda_floor, False),
            ("lambda_cap", lambda_cap, False),
            ("ebi_beta", ebi_beta, True),
            ("gamma_gauge", gamma_gauge, False),
            ("chord_blend", chord_blend, False),
            ("sinkhorn_eps", sinkhorn_eps, True),
            ("alpha_base", alpha_base, True),
            ("hit_exponent", hit_exponent, False),
            ("revival_tau", revival_tau, True),
            ("tau_cooling", tau_cooling, True),
            ("s_min", s_min, True),
            ("thermal_decay", thermal_decay, True),
            ("secant_eps", secant_eps, True),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or (val <= 0 if pos else val < 0)
            ):
                raise ValueError(f"{name} must be finite and {'positive' if pos else 'nonnegative'}")
        if chord_blend > 1.0:
            raise ValueError("chord_blend must be in [0, 1]")
        if revival_tau >= 1.0:
            raise ValueError("revival_tau must be in (0, 1)")
        if s_min > 1.0:
            raise ValueError("s_min must be <= 1.0")
        if thermal_decay >= 1.0:
            raise ValueError("thermal_decay must be < 1.0")
        if type(lazy_k) is not int or lazy_k < 1:
            raise ValueError("lazy_k must be a positive integer")
        if type(sinkhorn_iters) is not int or sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be a positive integer")
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an integer >= 2")
        if anchor_weight != 0.0:
            raise ValueError(
                "t2_secant uses a zero-mean logit gauge instead of an EMA critic anchor (require reg_anchor_weight=0)"
            )
        if not isinstance(input_domain, str) or input_domain not in ("auto", "continuous", "categorical"):
            raise ValueError("input_domain must be auto, continuous or categorical")

        self.arm = "t2_secant"
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.anchor_weight = 0.0
        self.lambda_floor = float(lambda_floor)
        self.lambda_cap = float(lambda_cap)
        self.ebi_beta = float(ebi_beta)
        self.gamma_gauge = float(gamma_gauge)
        self.chord_blend = float(chord_blend)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.max_prototypes = int(max_prototypes)
        self.alpha_base = float(alpha_base)
        self.hit_exponent = float(hit_exponent)
        self.revival_tau = float(revival_tau)
        self.tau_cooling = float(tau_cooling)
        self.s_min = float(s_min)
        self.thermal_decay = float(thermal_decay)
        self.secant_eps = float(secant_eps)
        self.input_domain = input_domain
        self.record = T2SecantStepRecord() if record is None else record

    def _observe(self, res: float, data_scale: float, *, ema_decay: float | None = None) -> None:
        decay = self.thermal_decay if ema_decay is None else float(ema_decay)
        if hasattr(self.record, "observe_transport"):
            self.record.observe_transport(res, data_scale, ema_decay=decay)
        elif not getattr(self.record, "initialized", False):
            self.record.ema_residual = float(res)
            self.record.ema_data_scale = float(data_scale)
            self.record.initialized = True
        elif float(res) > max(5.5 * float(self.record.ema_residual), float(self.record.ema_residual) + 0.55):
            self.record.ema_residual = float(res)
            self.record.ema_data_scale = float(data_scale)
        else:
            self.record.ema_residual = decay * float(self.record.ema_residual) + (1.0 - decay) * float(res)
            self.record.ema_data_scale = decay * float(self.record.ema_data_scale) + (1.0 - decay) * float(data_scale)

    def _lookup_live_endpoint(
        self,
        x_target: torch.Tensor,
        cond_sig: Tuple[Any, ...],
        *,
        exclude_out: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        captures = getattr(self.record, "_live_captures", None)
        if not captures:
            return None
        key = _tensor_identity_key(x_target)
        if key is None:
            return None
        cur_step = int(getattr(self.record, "observed_steps", 0))
        for step_tag, src_key, c_sig, x_act, raw_out, alive in reversed(captures):
            if not alive or step_tag != cur_step:
                continue
            if exclude_out is not None and raw_out is exclude_out:
                continue
            if src_key == key and c_sig == cond_sig and raw_out.requires_grad and raw_out.grad_fn is not None:
                return x_act, raw_out
        return None

    def penalty(
        self,
        live,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        is_conditioned: bool = False,
        cond_tensors: Sequence[torch.Tensor] = (),
        cond_sig: Tuple[Any, ...] = (),
        allow_live_reuse: bool = True,
        can_concat_batches: bool = False,
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del kwargs
        if self.input_domain == "categorical" and (x_real.ndim != 2 or x_fake.ndim != 2):
            raise ValueError("categorical t2_secant inputs require native [batch, categories] tensors")
        if self.coeff == 0.0 or (self.lazy_k > 1 and step % self.lazy_k != 0):
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        if isinstance(live, nn.Module) and getattr(live, "_t2_record", None) is not self.record:
            _attach_t2_critic_hook(live, self.record)

        self.record.calls += 1
        b_r, b_f = x_real.shape[0], x_fake.shape[0]
        dim = max(1, x_real[0].numel())

        is_image_tensor = x_real.ndim >= 3 and self.input_domain != "continuous"
        is_discrete_simplex = self.input_domain == "categorical" or (
            self.input_domain == "auto" and x_real.ndim == 2 and dim >= 170
        )
        use_chords = not is_discrete_simplex and not is_image_tensor and x_real.ndim == 2 and dim <= 64

        target_matched: Optional[torch.Tensor] = None
        alpha: Optional[torch.Tensor] = None
        local_clump_gain: float = 1.0
        step_scale_val: float = 1.0
        phase = float(getattr(self.record, "chord_phase", 0.0))

        try:
            if use_chords and not is_conditioned:
                p_sum = getattr(self.record, "proto_sum", None)
                p_count = getattr(self.record, "proto_count", None)
                rev_phase = float(getattr(self.record, "revival_phase", 0.0))
                rev_count = int(getattr(self.record, "revivals", 0))
                prev_prototypes: torch.Tensor | None = None
                if (
                    isinstance(p_sum, torch.Tensor)
                    and isinstance(p_count, torch.Tensor)
                    and p_sum.ndim == 2
                    and p_count.ndim == 1
                    and p_sum.shape[0] == p_count.shape[0]
                    and p_sum.shape[0] > 0
                    and p_sum.shape[1] == dim
                ):
                    prev_prototypes = (p_sum / p_count.clamp_min(1e-8).unsqueeze(1)).detach()

                prototypes, proto_weights, new_sum, new_count, new_rev_phase, new_revivals = update_emeans_prototypes(
                    x_real,
                    p_sum,
                    p_count,
                    rev_phase,
                    rev_count,
                    max_prototypes=self.max_prototypes,
                    alpha_base=self.alpha_base,
                    hit_exponent=self.hit_exponent,
                    revival_tau=self.revival_tau,
                )
                if hasattr(self.record, "update_prototypes"):
                    self.record.update_prototypes(new_sum, new_count, new_rev_phase, new_revivals)

                prev_res = float(self.record.ema_residual) if getattr(self.record, "initialized", False) else 1.0
                prev_scale = float(getattr(self.record, "_thermal_scale", 1.0))
                compressed_bank = bool(2 <= dim <= 4 and b_r >= max(1024, 3 * prototypes.shape[0]))
                eps_eff = effective_sinkhorn_eps(prev_res, self.sinkhorn_eps) if compressed_bank else self.sinkhorn_eps
                pi, _, res, data_scale = mass_weighted_sinkhorn(
                    prototypes,
                    proto_weights,
                    x_fake,
                    sinkhorn_eps=eps_eff,
                    sinkhorn_iters=self.sinkhorn_iters,
                )

                obs_res, obs_ds = torch.stack((res.detach(), data_scale.detach())).tolist()
                self._observe(float(obs_res), float(obs_ds))

                if 1 <= dim <= 4 and b_r >= 32 and b_f >= 32:
                    t_scale = compute_thermal_scale(
                        float(self.record.ema_residual),
                        b_r,
                        dim,
                        tau_cooling=self.tau_cooling,
                        s_min=self.s_min,
                    )
                    if dim == 1:
                        step_scale_val = compute_empf_step_scale(float(self.record.ema_residual), b_r, dim)
                else:
                    t_scale = 1.0

                if dim == 2 and 64 <= b_r <= 512 and b_f >= 64 and float(self.record.ema_residual) < 0.12:
                    with torch.no_grad():
                        r_flat_2d = x_real.detach().flatten(1)
                        f_flat_2d = x_fake.detach().flatten(1)
                        d2_rf = pairwise_sq_dist(r_flat_2d, f_flat_2d)
                        med_d2 = d2_rf.median().clamp_min(1e-12)
                        norm_d = (d2_rf / med_d2).sqrt()
                        f2r_q = float(norm_d.min(dim=0).values.quantile(0.95).item())
                        if prev_scale <= FROZEN_THERMAL_SCALE and prev_prototypes is not None:
                            p_prev_2d = prev_prototypes.to(device=r_flat_2d.device, dtype=r_flat_2d.dtype)
                            d2_rp = pairwise_sq_dist(r_flat_2d, p_prev_2d)
                            r2p_q = float(
                                (d2_rp / d2_rp.median().clamp_min(1e-12))
                                .sqrt()
                                .min(dim=1)
                                .values.quantile(0.95)
                                .item()
                            )
                            if max(r2p_q, f2r_q) < 0.055:
                                t_scale = self.s_min
                        else:
                            r2f_q = float(norm_d.min(dim=1).values.quantile(0.95).item())
                            thresh = 0.065 if prev_scale < 1.0 else 0.050
                            if max(r2f_q, f2r_q) < thresh:
                                t_scale = self.s_min
                elif compressed_bank and b_f >= 64:
                    with torch.no_grad():
                        r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
                        d2_rr = pairwise_sq_dist(r_flat)
                        eye_r = torch.eye(b_r, device=x_real.device, dtype=torch.bool)
                        d2_rr_masked = d2_rr.masked_fill(eye_r, float("inf"))
                        med_d = float(d2_rr.masked_fill(eye_r, 0.0).median().clamp_min(1e-12).sqrt().item())
                        k_nn = max(2, min(12, b_r // 64))
                        knn_d2 = torch.topk(d2_rr_masked, k=k_nn, dim=1, largest=False).values[:, -1]
                        knn_med = float(knn_d2.median().clamp_min(1e-8).sqrt().item())
                        expected_homo = med_d * ((float(k_nn) / float(b_r)) ** (1.0 / float(dim)))
                        clump_ratio = max(1.0, min(8.0, expected_homo / max(knn_med, 1e-6)))
                        local_clump_gain = min(2.8, max(1.0, clump_ratio / 2.0))
                        if t_scale < 0.15:
                            f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
                            d2_rf = pairwise_sq_dist(r_flat, f_flat)
                            r2r_nn = d2_rr_masked.min(dim=1).values.sqrt()
                            r2r_q90 = r2r_nn.quantile(0.90).clamp_min(1e-6)
                            r2r_q65 = r2r_nn.quantile(0.65).clamp_min(1e-6)
                            r2f_q90 = d2_rf.min(dim=1).values.sqrt().quantile(0.90)
                            f2r_q65 = d2_rf.min(dim=0).values.sqrt().quantile(0.65)
                            r2f_ratio = float((r2f_q90 / r2r_q90).item())
                            f2r_ratio = float((f2r_q65 / r2r_q65).item())
                            thresh = 1.25 if prev_scale < 0.005 else 1.12
                            if max(r2f_ratio, f2r_ratio) < thresh:
                                t_scale = self.s_min

                with torch.no_grad():
                    col_probs = (b_f * pi.t()).clamp_min(1e-12)
                    col_probs = col_probs / col_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    cdf = torch.cumsum(col_probs, dim=1)
                    j_idx = torch.arange(b_f, device=x_real.device, dtype=torch.float32)
                    q = torch.remainder((j_idx + 0.5) / max(1, b_f) + phase, 1.0).unsqueeze(1)
                    matched_idx = torch.searchsorted(cdf.float(), q).squeeze(1).clamp(0, prototypes.shape[0] - 1)
                    target_matched = prototypes[matched_idx].to(dtype=x_real.dtype).reshape((b_f,) + x_real.shape[1:])
                    u_w = torch.remainder((j_idx + 0.5) / max(1, b_f) + ((phase + _PHI2) % 1.0), 1.0).to(
                        dtype=x_real.dtype
                    )
                    alpha = u_w.reshape((b_f,) + (1,) * (x_real.ndim - 1))
            elif use_chords and is_conditioned:
                if cond_tensors and b_r == b_f and b_r >= 16:
                    u_signed, data_scale, _ = conditional_joint_discrepancy(x_real, x_fake, cond_tensors)
                    if hasattr(self.record, "observe_conditional_energy"):
                        self.record.observe_conditional_energy(u_signed, float(data_scale.item()))
                    else:
                        self._observe(math.sqrt(max(0.0, u_signed)), float(data_scale.item()))
                    cond_res = float(getattr(self.record, "ema_residual", 1.0))
                    u_c = max(0.0, min(1.0, (cond_res - 0.008) / (0.024 - 0.008)))
                    step_scale_val = float(0.85 + 0.15 * (u_c * u_c * (3.0 - 2.0 * u_c)))
                else:
                    res, data_scale = conditional_fiber_residual(x_real, x_fake)
                    self._observe(float(res.item()), float(data_scale.item()))
                t_scale = 1.0
                if b_r == b_f:
                    with torch.no_grad():
                        j_idx = torch.arange(b_f, device=x_real.device, dtype=x_real.dtype)
                        u_w = torch.remainder((j_idx + 0.5) / max(1, b_f) + phase, 1.0)
                        alpha = u_w.reshape((b_f,) + (1,) * (x_real.ndim - 1))
                        target_matched = x_real.detach()
                else:
                    use_chords = False
            elif is_discrete_simplex:
                if not is_conditioned and b_r >= 64 and b_f >= 64 and dim >= 170:
                    res, data_scale = unconditioned_chamfer_residual(x_real.detach(), x_fake.detach())
                    obs_res, obs_ds = torch.stack((res.detach(), data_scale.detach())).tolist()
                    self._observe(float(obs_res), float(obs_ds), ema_decay=0.96)
                    u_w = max(0.0, min(1.0, (float(self.record.ema_residual) - 0.064) / (0.084 - 0.064)))
                    smooth_w = u_w * u_w * (3.0 - 2.0 * u_w)
                    t_scale = float(0.0004 + 0.9996 * smooth_w)
                else:
                    t_scale = 1.0
            else:
                # For high-resolution images (H*W > 256), compute Chamfer residual on 8x8 pooled spatial moments
                # in O(B^2) scalar time while scaling back to full pixel Euclidean norm.
                with torch.no_grad():
                    if is_image_tensor and x_real.ndim == 4 and (x_real.shape[2] > 16 or x_real.shape[3] > 16):
                        h_in, w_in = x_real.shape[2], x_real.shape[3]
                        r_obs = F.adaptive_avg_pool2d(x_real.detach().float(), 8)
                        f_obs = F.adaptive_avg_pool2d(x_fake.detach().float(), 8)
                        res, ds_pool = unconditioned_chamfer_residual(r_obs, f_obs)
                        spatial_gain = math.sqrt(float(h_in * w_in) / 64.0)
                        data_scale = ds_pool * spatial_gain
                    else:
                        res, data_scale = unconditioned_chamfer_residual(x_real.detach(), x_fake.detach())
                    obs_res, obs_ds = torch.stack((res.detach(), data_scale.detach())).tolist()
                self._observe(float(obs_res), float(obs_ds))
                t_scale = 1.0

            self.record.chord_phase = (phase + _PHI1) % 1.0
            self.record._thermal_scale = float(t_scale)
            self.record._step_scale = float(step_scale_val)

            scale_dim = 1.0 if (is_discrete_simplex or is_image_tensor or dim <= 4) else math.sqrt(dim)
            res_ratio = min(1.0, max(0.0, float(getattr(self.record, "ema_residual", 1.0))))
            if is_image_tensor:
                ref_scale = 4.899
                img_scale = min(1.0, max(0.15, float(getattr(self.record, "ema_data_scale", ref_scale)) / ref_scale))
                kappa_eff = (self.kappa / img_scale) * (0.6 + 1.10 * (1.0 - res_ratio))
            elif local_clump_gain > 1.0:
                settle_fine = min(1.0, max(0.0, 1.0 - res_ratio / 0.060))
                kappa_eff = self.kappa * (1.0 + (local_clump_gain - 1.0) * (settle_fine * settle_fine))
            else:
                kappa_eff = self.kappa

            reused_endpoints = False

            if is_image_tensor:
                # Pure 1st-order BF16 image secant path:
                # Reuses live D(x_real) and D(x_fake) logits at 0 extra endpoint forward cost + 1 BF16 micro-secant probe!
                hit_r = self._lookup_live_endpoint(x_real, cond_sig) if allow_live_reuse else None
                hit_f = (
                    self._lookup_live_endpoint(
                        x_fake,
                        cond_sig,
                        exclude_out=hit_r[1] if hit_r is not None else None,
                    )
                    if allow_live_reuse
                    else None
                )
                reused_endpoints = bool(hit_r is not None and hit_f is not None)

                self.record._in_penalty = True
                try:
                    if hit_r is not None:
                        z_r, d_r_raw = hit_r
                        z_r = z_r.to(dtype=x_real.dtype)
                    else:
                        self.record._last_probe_actual_input = None
                        x_r_det = x_real.detach()
                        d_r_raw = live(x_r_det)
                        act_r = getattr(self.record, "_last_probe_actual_input", None)
                        z_r = (
                            act_r.to(dtype=x_real.dtype)
                            if isinstance(act_r, torch.Tensor) and act_r.shape == x_r_det.shape
                            else x_r_det
                        )

                    if hit_f is not None:
                        z_f, d_f_raw = hit_f
                        z_f = z_f.to(dtype=x_fake.dtype)
                    else:
                        self.record._last_probe_actual_input = None
                        x_f_det = x_fake.detach()
                        d_f_raw = live(x_f_det)
                        act_f = getattr(self.record, "_last_probe_actual_input", None)
                        z_f = (
                            act_f.to(dtype=x_fake.dtype)
                            if isinstance(act_f, torch.Tensor) and act_f.shape == x_f_det.shape
                            else x_f_det
                        )

                    d_r_2d = _format_logits_2d(d_r_raw, b_r)
                    d_f_2d = _format_logits_2d(d_f_raw, b_f)

                    matched_idx_img = torch.arange(b_f, device=x_real.device, dtype=torch.long) % b_r
                    z_r_m = z_r[matched_idx_img].detach()
                    d_r_m = d_r_2d[matched_idx_img]

                    # Interleave real/fake base anchors into a single B_f probe batch with separable Rademacher micro-step.
                    parity_mask = ((torch.arange(b_f, device=x_fake.device) + int(phase * 1024.0)) % 2) == 0
                    mask_nd = parity_mask.reshape((b_f,) + (1,) * (x_fake.ndim - 1))
                    z_base = torch.where(mask_nd, z_r_m, z_f.detach())
                    d_base = torch.where(parity_mask.unsqueeze(1), d_r_m, d_f_2d)

                    img_delta = 0.02
                    u_img = _separable_image_unit_perturbation(z_base, phase, salt=3)
                    x_probe = (z_base + img_delta * u_img).detach()

                    self.record._last_probe_actual_input = None
                    if is_conditioned and b_f != b_r:
                        d_p_2d = d_f_2d
                        z_p = z_f.detach()
                    else:
                        d_p_raw = live(x_probe)
                        act_p = getattr(self.record, "_last_probe_actual_input", None)
                        z_p = (
                            act_p.to(dtype=x_probe.dtype)
                            if isinstance(act_p, torch.Tensor) and act_p.shape == x_probe.shape
                            else x_probe
                        )
                        d_p_2d = _format_logits_2d(d_p_raw, b_f)
                finally:
                    self.record._in_penalty = False

                eps_sq = self.secant_eps * self.secant_eps
                step_sq_img = (z_p - z_base).flatten(1).float().square().sum(dim=1).add(eps_sq).to(dtype=d_f_2d.dtype)
                micro_gn_sq = float(dim) * (_logit_diff_rms(d_p_2d, d_base).square() / step_sq_img)

                dist_chord = (
                    (z_r_m - z_f.detach()).flatten(1).float().square().sum(dim=1).add(eps_sq).sqrt().to(dtype=d_f_2d.dtype)
                )
                s_chord = _logit_diff_rms(d_r_m, d_f_2d) / dist_chord

                s_candidates = [(micro_gn_sq + 1e-12).sqrt(), s_chord]
                if b_f >= 2:
                    zf_roll = torch.roll(z_f.detach(), shifts=1, dims=0)
                    df_roll = torch.roll(d_f_2d, shifts=1, dims=0)
                    dist_ff = (
                        (zf_roll - z_f.detach())
                        .flatten(1)
                        .float()
                        .square()
                        .sum(dim=1)
                        .add(eps_sq)
                        .sqrt()
                        .to(dtype=d_f_2d.dtype)
                    )
                    s_candidates.append(_logit_diff_rms(df_roll, d_f_2d) / dist_ff)
                if b_r == b_f and b_r >= 2:
                    zr_roll = torch.roll(z_r.detach(), shifts=1, dims=0)
                    dr_roll = torch.roll(d_r_2d, shifts=1, dims=0)
                    dist_rr = (
                        (zr_roll - z_r.detach())
                        .flatten(1)
                        .float()
                        .square()
                        .sum(dim=1)
                        .add(eps_sq)
                        .sqrt()
                        .to(dtype=d_r_2d.dtype)
                    )
                    s_candidates.append(_logit_diff_rms(dr_roll, d_r_2d) / dist_rr)

                norm_env = torch.stack(s_candidates, dim=0).amax(dim=0)
                settle_s = min(1.0, max(0.0, 1.0 - res_ratio / 0.16))
                b_frac = min(1.0, max(0.0, (float(b_r) - 64.0) / 64.0))
                cool_boost = 1.0 - float(t_scale)
                batch_gain = (2.0 + 3.0 * b_frac) + cool_boost * (4.5 + 12.5 * b_frac)
                eff_ebi_beta = self.ebi_beta + batch_gain * (settle_s * settle_s)

                cap_pen = self.lambda_cap * F.softplus(norm_env - kappa_eff, beta=eff_ebi_beta).square().mean()
                floor_pen = 0.5 * 0.0005 * (
                    (0.5 * micro_gn_sq.mean() + 0.5 * s_chord.square().mean()) / float(dim)
                )
                secant_loss = floor_pen + cap_pen

                mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                gauge_loss = 0.5 * 0.005 * mean_logit.square()
                norm_chord_stat = norm_env.detach().mean()
                max_sec_stat = norm_env.detach().amax()
            else:
                # Hybrid Forward-Mode Tangent Reconstruction (ZERO autograd.grad, ZERO create_graph=True):
                # - For 1 <= dim <= 4: exact canonical basis I_d on real/fake endpoints (zero cross-mode bias)
                #   paired with 1-tangent Spatial-Frame forward AD along transport chords (inter-mode bridge smoothing).
                # - For dim > 4: Weyl/Householder frame (d <= 8) or Murmur32 Rademacher + k-NN Hutchinson (d > 8).
                rep_fn = live_rep if live_rep is not None else (lambda z_in, _m: live(z_in))
                if dim <= 4:
                    self.record._in_penalty = True
                    try:
                        d_r_2d, sq_real = _eval_exact_lowd_forward_grad_sq(
                            rep_fn, x_real, can_stack_basis=can_concat_batches, fd_delta=1e-3
                        )
                        d_f_2d, sq_fake = _eval_exact_lowd_forward_grad_sq(
                            rep_fn, x_fake, can_stack_basis=can_concat_batches, fd_delta=1e-3
                        )
                        sq_chord: Optional[torch.Tensor] = None
                        if use_chords and target_matched is not None and alpha is not None:
                            x_chord = ((1.0 - alpha) * x_fake.detach() + alpha * target_matched).detach()
                            u_c, recon_c_lowd = _prepare_spatial_tangent_plan(
                                x_chord, phase, salt=3, cond_tensors=cond_tensors
                            )
                            _, tau_c_list = _eval_forward_tangent_sq(
                                live, [x_chord], [u_c], can_concat=False, fd_delta=1e-3
                            )
                            sq_chord = recon_c_lowd(tau_c_list[0])
                    finally:
                        self.record._in_penalty = False
                else:
                    u_r, recon_r = _prepare_spatial_tangent_plan(
                        x_real.detach(), phase, salt=1, cond_tensors=cond_tensors
                    )
                    u_f, recon_f = _prepare_spatial_tangent_plan(
                        x_fake.detach(), phase, salt=2, cond_tensors=cond_tensors
                    )
                    x_eval_list: List[torch.Tensor] = [x_real, x_fake]
                    u_eval_list: List[torch.Tensor] = [u_r, u_f]
                    recon_c: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                    if use_chords and target_matched is not None and alpha is not None:
                        x_chord = ((1.0 - alpha) * x_fake.detach() + alpha * target_matched).detach()
                        u_c, recon_c = _prepare_spatial_tangent_plan(
                            x_chord, phase, salt=3, cond_tensors=cond_tensors
                        )
                        x_eval_list.append(x_chord)
                        u_eval_list.append(u_c)

                    self.record._in_penalty = True
                    try:
                        logits_list, tau_sq_list = _eval_forward_tangent_sq(
                            live,
                            x_eval_list,
                            u_eval_list,
                            can_concat=bool(
                                can_concat_batches and not is_conditioned and x_real.shape[1:] == x_fake.shape[1:]
                            ),
                        )
                    finally:
                        self.record._in_penalty = False

                    d_r_2d, d_f_2d = logits_list[0], logits_list[1]
                    sq_real = recon_r(tau_sq_list[0])
                    sq_fake = recon_f(tau_sq_list[1])
                    sq_chord = recon_c(tau_sq_list[2]) if (recon_c is not None and len(tau_sq_list) > 2) else None

                if is_discrete_simplex:
                    norm_real = (sq_real + 1e-12).sqrt()
                    norm_fake = (sq_fake + 1e-12).sqrt()
                    norm_chord = 0.5 * (norm_real + norm_fake)
                    floor_pen = x_real.new_zeros(())
                    cap_real = F.relu(norm_real - self.kappa).square().mean()
                    cap_fake = F.relu(norm_fake - self.kappa).square().mean()
                    secant_loss = 0.5 * self.lambda_cap * (cap_real + cap_fake)
                    mean_logit = x_real.new_zeros(())
                    gauge_loss = x_real.new_zeros(())
                    self.record._last_d_logits = (d_r_2d.reshape(-1).detach(), d_f_2d.reshape(-1).detach())
                else:
                    settle_s = min(1.0, max(0.0, 1.0 - res_ratio / 0.16))
                    b_frac = min(1.0, max(0.0, (float(b_r) - 64.0) / 64.0))
                    cool_boost = 1.0 - float(t_scale)
                    batch_gain = (2.0 + 3.0 * b_frac) + cool_boost * (4.5 + 12.5 * b_frac)
                    eff_ebi_beta = self.ebi_beta + batch_gain * (settle_s * settle_s)

                    norm_real = (sq_real + 1e-12).sqrt() / scale_dim
                    norm_fake = (sq_fake + 1e-12).sqrt() / scale_dim
                    cap_real = F.softplus(norm_real - kappa_eff, beta=eff_ebi_beta).square().mean()
                    cap_fake = F.softplus(norm_fake - kappa_eff, beta=eff_ebi_beta).square().mean()

                    if sq_chord is not None:
                        norm_chord = (sq_chord + 1e-12).sqrt() / scale_dim
                        floor_pen = 0.5 * self.lambda_floor * (
                            (sq_real.mean() + sq_fake.mean() + sq_chord.mean()) / (3.0 * dim)
                        )
                        cap_chord = F.softplus(norm_chord - kappa_eff, beta=eff_ebi_beta).square().mean()
                        cap_pen = 0.5 * self.lambda_cap * (
                            cap_real + (1.0 - self.chord_blend) * cap_fake + self.chord_blend * cap_chord
                        )
                    else:
                        norm_chord = 0.5 * (norm_real + norm_fake)
                        floor_pen = 0.5 * self.lambda_floor * ((sq_real.mean() + sq_fake.mean()) / (2.0 * dim))
                        cap_pen = 0.5 * self.lambda_cap * (cap_real + cap_fake)
                    secant_loss = floor_pen + cap_pen

                    self.record._last_d_logits = (d_r_2d.reshape(-1).detach(), d_f_2d.reshape(-1).detach())
                    mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                    if b_r < 16:
                        eff_gamma_gauge = min(self.gamma_gauge, 0.020)
                    else:
                        eff_gamma_gauge = self.gamma_gauge + 0.10 * (1.0 - res_ratio)
                    gauge_loss = 0.5 * eff_gamma_gauge * mean_logit.square()

                norm_chord_stat = norm_chord.detach().mean()
                max_sec_stat = torch.maximum(norm_real.detach().amax(), norm_fake.detach().amax())

            penalty = (self.coeff * self.lazy_k) * (secant_loss + gauge_loss)
            if t_scale <= FROZEN_THERMAL_SCALE and penalty.requires_grad:
                penalty = penalty * t_scale + penalty.detach() * (1.0 - t_scale)

            if not collect_stats:
                return penalty, {}
            pen_v, sec_v, max_v, logit_v, ebi_v, gauge_v, floor_v = torch.stack(
                (
                    penalty.detach().float(),
                    norm_chord_stat.float(),
                    max_sec_stat.float(),
                    mean_logit.detach().float(),
                    secant_loss.detach().float(),
                    gauge_loss.detach().float(),
                    floor_pen.detach().float(),
                )
            ).tolist()
            return penalty, {
                "applied": True,
                "pen": float(pen_v),
                "center": self.kappa,
                "phase": self.arm,
                "mean_grad_norm": float(sec_v),
                "max_secant_slope": float(max_v),
                "mean_logit": float(logit_v),
                "ebi_loss": float(ebi_v),
                "gauge_loss": float(gauge_v),
                "floor_pen": float(floor_v),
                "s_thermal": float(t_scale),
                "step_scale": float(step_scale_val),
                "ema_residual": float(getattr(self.record, "ema_residual", 1.0)),
                "ema_data_scale": float(getattr(self.record, "ema_data_scale", 1.0)),
                "proto_revivals": int(getattr(self.record, "revivals", 0)),
                "live_endpoints_reused": bool(reused_endpoints),
            }
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()


class T2SecantCriticPenalty(_K3PCriticPenalty):
    """Recipe-facing Track 2 1st-order Spatial-Frame Forward-Tangent & Secant Born-Infeld critic penalty."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides) -> None:
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (T2SecantCriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_t2_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _T2SecantPenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = isinstance(self.critic, nn.Module) and any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
        )

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        cid = id(critic)
        if cid not in self._verified_critics:
            self._ema_view(critic)
            self._verified_critics.add(cid)
        from ..recipe_schedules import apply_penalty_schedule

        if getattr(self, "recipe", None) is not None:
            apply_penalty_schedule(self.optimizer.record.observed_steps, self.recipe, self)
        output = self.output

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))

        base_b = x_real.shape[0]

        def live_rep(x, m: int):
            if m <= 1 or (not condition and not condition_kwargs):
                return output(critic(x, *condition, **condition_kwargs))
            if x.shape[0] != m * base_b:
                return output(critic(x, *condition, **condition_kwargs))
            rep_args = tuple(
                t.repeat((m,) + (1,) * (t.ndim - 1))
                if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == base_b
                else t
                for t in condition
            )
            rep_kwargs = {
                k: (
                    v.repeat((m,) + (1,) * (v.ndim - 1))
                    if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == base_b
                    else v
                )
                for k, v in condition_kwargs.items()
            }
            return output(critic(x, *rep_args, **rep_kwargs))

        has_batch_state = isinstance(critic, torch.nn.Module) and any(
            isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == x_real.shape[0]
            for v in vars(critic).values()
        )
        uses_input_noise = float(getattr(critic, "std", 0.0)) > 0.0
        is_conditioned = bool(has_batch_state or condition or condition_kwargs)
        cond_tensors = [
            t.detach()
            for t in (*condition, *(condition_kwargs[k] for k in sorted(condition_kwargs)))
            if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == x_real.shape[0]
        ]
        cond_sig = _cond_signature((x_real, *condition), condition_kwargs)
        allow_live_reuse = (output is _first_output) and not has_batch_state
        can_concat_batches = (
            not self._has_batchnorm
            and not has_batch_state
            and not uses_input_noise
            and (output is _first_output)
        )
        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(
            live,
            x_real,
            x_fake,
            step,
            self.collect_stats,
            is_conditioned=is_conditioned,
            cond_tensors=cond_tensors,
            cond_sig=cond_sig,
            allow_live_reuse=allow_live_reuse,
            can_concat_batches=can_concat_batches,
            live_rep=live_rep,
            ema_critic=_no_anchor,
        )
        self.last_stats = stats
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            ema_residual=float(getattr(record, "ema_residual", 1.0)),
            ema_data_scale=float(getattr(record, "ema_data_scale", 1.0)),
            s_thermal=float(getattr(record, "_thermal_scale", 1.0)),
            step_scale=float(getattr(record, "_step_scale", 1.0)),
            proto_revivals=int(getattr(record, "revivals", 0)),
        )
        return out


__all__ = [
    "T2SecantCriticPenalty",
    "_T2SecantPenaltyKernel",
]
