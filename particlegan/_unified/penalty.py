"""Pure 1st-Order Spatial-Frame Forward-Tangent & Multi-Probe Secant Born-Infeld Critic (Unified Synthesis).

Eliminates every ``torch.autograd.grad(..., create_graph=True)`` second-order double-backprop
graph (`0` calls to ``torch.autograd.grad`` across all tasks and architectures):

1. **Exact & Spatial-Frame 1st-Order Forward-Mode AD (Vector / MLP / Conditioned / Simplex Critics)**:
   - ``1 <= d <= 4``: Exact canonical basis ``I_d`` propagated via ``torch.autograd.forward_ad.dual_level``
     (zero directional estimator variance; quantization-crossing symmetric central-difference fallback
     ``delta = max(0.06, 0.04 * scale)`` for custom 4-bit/8-bit software-emulated GEMMs without ``jvp``).
   - ``5 <= d <= 8``: Zero-sync analytical Householder orthonormal frame ``Q(phi) in O(d)`` reconstructed across
     nearest basis-slot neighbors.
   - ``d > 8`` (e.g. 170D discrete-simplex BiGAN): Murmur32 Rademacher unit tangents with local k-NN
     Hutchinson variance reduction.

2. **Live-Endpoint Reuse + Orthogonal Multi-Probe BF16 Secant Critic (Convolutional Image Critics)**:
   - Reuses live unscaled ``D(x_real)`` and ``D(x_fake)`` logits from ``d_loss`` at zero extra endpoint forward cost.
   - Small discrete spatial grids (``d <= 1024``, e.g. ``8x8`` / ``16x16``): evaluates ``K = 2`` per-sample
     Gram-Schmidt orthogonalized Rademacher micro-secant probes on ``x_real`` and ``x_fake`` with local
     intra-batch neighbor variance reduction, curing single-probe directional variance.
   - High-resolution RGB images (``d > 1024``, e.g. ``32x32``..``256x256``): evaluates 1 interleaved BF16
     separable micro-secant probe batch alongside zero-cost transport and cyclic cross-sample secants
     (``< 95 ms/step`` at ``256x256``).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._stats import pairwise_sq_dist
from .._t1_coupling.primitive import T1CouplingLoss
from .._t2_critic.penalty import (
    _format_logits_2d,
    _logit_diff_rms,
    _murmur32_signs,
    _prepare_spatial_tangent_plan,
    _separable_image_unit_perturbation,
)
from ..grad_regularizers import CriticStepRecord
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .critic import (
    UnifiedCriticAdam,
    UnifiedStepRecord,
    _attach_unified_critic_hook,
    _cond_signature,
    _tensor_identity_key,
)

_GOLDEN_CONJ = 0.6180339887498949
_PLASTIC_CONJ = 0.7548776662466927


def _adaptive_fd_delta(x_det: torch.Tensor, *, quantized_fprop: bool = False) -> float:
    """Choose symmetric central-difference step size when custom autograd.Function drops fwAD tangents."""
    if not quantized_fprop:
        return 1e-3
    with torch.no_grad():
        scale = float(x_det.detach().float().abs().mean().clamp_min(1.0).item())
    # Cross 4-bit / 8-bit activation quantization bins cleanly without macro curvature bias.
    return max(0.06, 0.04 * scale)


def _eval_lowd_forward_grad_sq(
    live_rep: Callable[[torch.Tensor, int], torch.Tensor],
    x_item: torch.Tensor,
    *,
    can_stack_basis: bool,
    quantized_fprop: bool = False,
    batch_mult: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact ``||nabla_x D(x)||_2^2`` for ``1 <= dim <= 4`` via 1st-order forward AD (0 autograd.grad calls)."""
    x_det = x_item.detach()
    b = int(x_det.shape[0])
    dim = max(1, x_det[0].numel())
    tail_ones = (1,) * (x_det.ndim - 1)
    fd_delta = _adaptive_fd_delta(x_det, quantized_fprop=quantized_fprop)

    if dim == 1:
        u1 = torch.ones_like(x_det)
        if not quantized_fprop:
            try:
                with fwAD.dual_level():
                    yd = live_rep(fwAD.make_dual(x_det, u1), batch_mult)
                    y_1, jvp_1 = fwAD.unpack_dual(yd)
                if jvp_1 is None:
                    raise NotImplementedError("dual tangent dropped")
                return _format_logits_2d(y_1, b), _format_logits_2d(jvp_1, b).square().sum(dim=-1)
            except (RuntimeError, NotImplementedError, TypeError):
                fd_delta = _adaptive_fd_delta(x_det, quantized_fprop=True)
        y_p = _format_logits_2d(live_rep((x_det + fd_delta * u1).detach(), batch_mult), b)
        y_m = _format_logits_2d(live_rep((x_det - fd_delta * u1).detach(), batch_mult), b)
        y_mid = 0.5 * (y_p + y_m)
        return y_mid, (y_p - y_m).square().sum(dim=-1) / (4.0 * fd_delta * fd_delta)

    eye = torch.eye(dim, device=x_det.device, dtype=x_det.dtype)
    if can_stack_basis:
        x_rep = x_det.repeat((dim,) + tail_ones)
        u_rep = eye.repeat_interleave(b, dim=0).reshape_as(x_rep)
        if not quantized_fprop:
            try:
                with fwAD.dual_level():
                    yd = live_rep(fwAD.make_dual(x_rep, u_rep), dim * batch_mult)
                    y_rep, jvp_rep = fwAD.unpack_dual(yd)
                if jvp_rep is None:
                    raise NotImplementedError("dual tangent dropped")
                y_2d = _format_logits_2d(y_rep[:b], b)
                jvp_sq = _format_logits_2d(jvp_rep, dim * b).square().sum(dim=-1).reshape(dim, b).sum(dim=0)
                return y_2d, jvp_sq
            except (RuntimeError, NotImplementedError, TypeError):
                fd_delta = _adaptive_fd_delta(x_det, quantized_fprop=True)
        y_p = _format_logits_2d(live_rep((x_rep + fd_delta * u_rep).detach(), dim * batch_mult), dim * b)
        y_m = _format_logits_2d(live_rep((x_rep - fd_delta * u_rep).detach(), dim * batch_mult), dim * b)
        y_mid = 0.5 * (y_p + y_m)
        diff_sq = ((y_p - y_m).square().sum(dim=-1) / (4.0 * fd_delta * fd_delta)).reshape(dim, b).sum(dim=0)
        return y_mid[:b], diff_sq

    y_first: Optional[torch.Tensor] = None
    sq_accum = x_det.new_zeros((b,), dtype=torch.float32)
    for axis in range(dim):
        u_axis = eye[axis].unsqueeze(0).expand(b, dim).reshape_as(x_det).contiguous()
        if not quantized_fprop:
            try:
                with fwAD.dual_level():
                    yd = live_rep(fwAD.make_dual(x_det, u_axis), 1)
                    y_a, jvp_a = fwAD.unpack_dual(yd)
                if jvp_a is None:
                    raise NotImplementedError("dual tangent dropped")
                y_2d_a = _format_logits_2d(y_a, b)
                part_sq = _format_logits_2d(jvp_a, b).square().sum(dim=-1)
            except (RuntimeError, NotImplementedError, TypeError):
                fd_delta_a = _adaptive_fd_delta(x_det, quantized_fprop=True)
                y_p = _format_logits_2d(live_rep((x_det + fd_delta_a * u_axis).detach(), 1), b)
                y_m = _format_logits_2d(live_rep((x_det - fd_delta_a * u_axis).detach(), 1), b)
                y_2d_a = 0.5 * (y_p + y_m)
                part_sq = (y_p - y_m).square().sum(dim=-1) / (4.0 * fd_delta_a * fd_delta_a)
        else:
            y_p = _format_logits_2d(live_rep((x_det + fd_delta * u_axis).detach(), 1), b)
            y_m = _format_logits_2d(live_rep((x_det - fd_delta * u_axis).detach(), 1), b)
            y_2d_a = 0.5 * (y_p + y_m)
            part_sq = (y_p - y_m).square().sum(dim=-1) / (4.0 * fd_delta * fd_delta)
        if y_first is None:
            y_first = y_2d_a
        sq_accum = sq_accum + part_sq.float()
    assert y_first is not None
    return y_first, sq_accum.to(dtype=y_first.dtype)


def _eval_spatial_tangent_sq(
    live: Callable[[torch.Tensor], torch.Tensor],
    x_list: Sequence[torch.Tensor],
    u_list: Sequence[torch.Tensor],
    *,
    can_concat: bool,
    quantized_fprop: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Evaluate ``(logits_2d, directional_deriv_sq)`` via 1st-order forward AD (or central-diff fallback)."""
    sizes = [int(x.shape[0]) for x in x_list]
    if can_concat and len(x_list) > 1:
        x_cat = torch.cat([x.detach() for x in x_list], dim=0)
        u_cat = torch.cat(list(u_list), dim=0)
        fd_delta = _adaptive_fd_delta(x_cat, quantized_fprop=quantized_fprop)
        if not quantized_fprop:
            try:
                with fwAD.dual_level():
                    yd = live(fwAD.make_dual(x_cat, u_cat))
                    y_cat, jvp_cat = fwAD.unpack_dual(yd)
                if jvp_cat is None:
                    raise NotImplementedError("dual tangent dropped")
                y_2d = _format_logits_2d(y_cat, sum(sizes))
                tau_sq = _format_logits_2d(jvp_cat, sum(sizes)).square().sum(dim=-1)
                return list(torch.split(y_2d, sizes, dim=0)), list(torch.split(tau_sq, sizes, dim=0))
            except (RuntimeError, NotImplementedError, TypeError):
                fd_delta = _adaptive_fd_delta(x_cat, quantized_fprop=True)
        y_p = _format_logits_2d(live((x_cat + fd_delta * u_cat).detach()), sum(sizes))
        y_m = _format_logits_2d(live((x_cat - fd_delta * u_cat).detach()), sum(sizes))
        y_mid = 0.5 * (y_p + y_m)
        step_sq = (2.0 * fd_delta * u_cat.flatten(1).float()).square().sum(dim=1).clamp_min(1e-12)
        tau_sq = (y_p - y_m).square().sum(dim=-1) / step_sq.to(dtype=y_mid.dtype)
        return list(torch.split(y_mid, sizes, dim=0)), list(torch.split(tau_sq, sizes, dim=0))

    logits_out: List[torch.Tensor] = []
    tau_out: List[torch.Tensor] = []
    for x_item, u_item, b_sz in zip(x_list, u_list, sizes):
        x_det = x_item.detach()
        fd_delta = _adaptive_fd_delta(x_det, quantized_fprop=quantized_fprop)
        if not quantized_fprop:
            try:
                with fwAD.dual_level():
                    yd = live(fwAD.make_dual(x_det, u_item))
                    y_i, jvp_i = fwAD.unpack_dual(yd)
                if jvp_i is None:
                    raise NotImplementedError("dual tangent dropped")
                y_2d = _format_logits_2d(y_i, b_sz)
                t_sq = _format_logits_2d(jvp_i, b_sz).square().sum(dim=-1)
                logits_out.append(y_2d)
                tau_out.append(t_sq)
                continue
            except (RuntimeError, NotImplementedError, TypeError):
                fd_delta = _adaptive_fd_delta(x_det, quantized_fprop=True)
        y_p = _format_logits_2d(live((x_det + fd_delta * u_item).detach()), b_sz)
        y_m = _format_logits_2d(live((x_det - fd_delta * u_item).detach()), b_sz)
        y_2d = 0.5 * (y_p + y_m)
        step_sq = (2.0 * fd_delta * u_item.flatten(1).float()).square().sum(dim=1).clamp_min(1e-12)
        t_sq = (y_p - y_m).square().sum(dim=-1) / step_sq.to(dtype=y_2d.dtype)
        logits_out.append(y_2d)
        tau_out.append(t_sq)
    return logits_out, tau_out


def _orthogonal_image_probe_pair(x: torch.Tensor, phase: float, salt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``K = 2`` per-sample orthogonal unit perturbations ``(u_1, u_2)`` on ``[B, C, H, W]``."""
    b = x.shape[0]
    flat_d = max(1, x[0].numel())
    ranks = torch.arange(b, device=x.device, dtype=torch.int64)
    u1_f = _murmur32_signs(ranks, flat_d, phase, salt, x.device)
    u1_f = u1_f / u1_f.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if flat_d < 2:
        u_t = u1_f.to(dtype=x.dtype).reshape_as(x)
        return u_t, u_t
    u2_raw = _murmur32_signs(ranks + 257, flat_d, phase + 0.38196601125, salt + 31, x.device)
    # Gram-Schmidt orthogonalization per sample: <u1_i, u2_i> == 0
    proj = (u2_raw * u1_f).sum(dim=1, keepdim=True) * u1_f
    u2_orth = u2_raw - proj
    bad = u2_orth.norm(dim=1) < 1e-5
    if bool(bad.any()):
        alt = torch.roll(u1_f, shifts=1, dims=1)
        alt[:, 0] = -alt[:, 0]
        alt = alt - (alt * u1_f).sum(dim=1, keepdim=True) * u1_f
        u2_orth = torch.where(bad.unsqueeze(1), alt, u2_orth)
    u2_f = u2_orth / u2_orth.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return u1_f.to(dtype=x.dtype).reshape_as(x), u2_f.to(dtype=x.dtype).reshape_as(x)


def _local_neighbor_smooth(z_det: torch.Tensor, sq_est: torch.Tensor, k_nn: int = 4) -> torch.Tensor:
    """Smooth 2-probe directional norm-squared estimate across intra-mode spatial neighbors."""
    b = z_det.shape[0]
    if b < 4 or k_nn <= 1:
        return sq_est
    with torch.no_grad():
        zf = z_det.flatten(1).float()
        d2 = pairwise_sq_dist(zf)
        eye = torch.eye(b, device=z_det.device, dtype=torch.bool)
        d2_off = d2.masked_fill(eye, float("inf"))
        nn1 = d2_off.amin(dim=1).clamp_min(1e-6)
        k_eff = min(k_nn, b)
        kd2, kidx = torch.topk(d2, k=k_eff, dim=1, largest=False)
        bw2 = (2.5 * nn1).unsqueeze(1)
        w = torch.exp(-0.5 * kd2 / bw2)
        w[:, 0] = w[:, 0] + 1.5  # Preserve >= 60% self weight so mode boundaries stay crisp
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (w.to(dtype=sq_est.dtype) * sq_est[kidx]).sum(dim=1)


class _UnifiedPenaltyKernel:
    """Clock-free 1st-order Born-Infeld endpoint, chord, and multi-probe secant critic kernel."""

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        ebi_beta: float = 12.0,
        real_weight: float = 1.0,
        fake_weight: float = 1.0,
        lambda_floor: float = 0.005,
        gamma_gauge: float = 0.03,
        chord_weight: float = 0.25,
        akt_horizon: bool = True,
        akt_mode: str = "weibull",
        settle_beta: float = 16.0,
        settle_mode: str = "thermal",
        secant_eps: float = 1e-5,
        input_domain: str = "auto",
        record: CriticStepRecord | None = None,
    ) -> None:
        for name, val, pos in (
            ("coeff", coeff, False),
            ("kappa", kappa, False),
            ("ebi_beta", ebi_beta, True),
            ("real_weight", real_weight, False),
            ("fake_weight", fake_weight, False),
            ("lambda_floor", lambda_floor, False),
            ("gamma_gauge", gamma_gauge, False),
            ("chord_weight", chord_weight, False),
            ("settle_beta", settle_beta, False),
            ("secant_eps", secant_eps, True),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or (val <= 0.0 if pos else val < 0.0)
            ):
                raise ValueError(f"{name} must be finite and {'positive' if pos else 'nonnegative'}")
        if float(chord_weight) > 1.0:
            raise ValueError("chord_weight must be in [0, 1]")
        if type(lazy_k) is not int or lazy_k < 1:
            raise ValueError("lazy_k must be a positive integer")
        if type(akt_horizon) is not bool:
            raise ValueError("akt_horizon must be a boolean")
        if type(akt_mode) is not str or akt_mode not in ("sqrt", "weibull"):
            raise ValueError("akt_mode must be 'sqrt' or 'weibull'")
        if type(settle_mode) is not str or settle_mode not in ("residual", "thermal"):
            raise ValueError("settle_mode must be 'residual' or 'thermal'")
        if anchor_weight != 0.0:
            raise ValueError("unified does not use an EMA critic anchor (require reg_anchor_weight=0)")
        if not isinstance(input_domain, str) or input_domain not in ("auto", "continuous", "categorical"):
            raise ValueError("input_domain must be auto, continuous or categorical")

        self.arm = "unified"
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.anchor_weight = 0.0
        self.ebi_beta = float(ebi_beta)
        self.real_weight = float(real_weight)
        self.fake_weight = float(fake_weight)
        self.lambda_floor = float(lambda_floor)
        self.gamma_gauge = float(gamma_gauge)
        self.chord_weight = float(chord_weight)
        self.akt_horizon = akt_horizon
        self.akt_mode = akt_mode
        self.settle_beta = float(settle_beta)
        self.settle_mode = settle_mode
        self.secant_eps = float(secant_eps)
        self.input_domain = input_domain
        self.record = UnifiedStepRecord() if record is None else record

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

    @torch.no_grad()
    def _build_geodesic_chord(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *,
        is_conditioned: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        b_n = x_real.shape[0]
        phi = float(getattr(self.record, "t1c_chord_phase", 0.0))
        next_phi = (phi + _GOLDEN_CONJ) % 1.0
        if hasattr(self.record, "t1c_chord_phase"):
            self.record.t1c_chord_phase = next_phi
        idx_f = (torch.arange(b_n, device=x_real.device, dtype=torch.float32) + 0.5) / float(b_n)
        alpha_1d = ((idx_f + phi) % 1.0).to(dtype=x_real.dtype)
        alpha = alpha_1d.reshape(b_n, *([1] * (x_real.ndim - 1)))
        if is_conditioned or b_n <= 1:
            y_pair = x_real.detach()
        else:
            ell_init = bool(getattr(self.record, "t1c_ell_initialized", False))
            ell_s = float(getattr(self.record, "t1c_ell_ema", 0.0)) if ell_init else 0.0
            if ell_s <= 1e-6:
                ell_t = T1CouplingLoss._batch_ell_quantile_tensor(
                    x_real, akt_horizon=self.akt_horizon, akt_mode=self.akt_mode
                )
            else:
                ell_t = torch.tensor(ell_s, dtype=torch.float32, device=x_real.device)
            c_hat = T1CouplingLoss._born_infeld_c_hat_detached(x_fake, x_real, ell_t)
            L_bal = -c_hat.float() / 0.25
            u_s = L_bal.new_zeros(b_n)
            v_s = L_bal.new_zeros(b_n)
            for _ in range(6):
                v_s = -torch.logsumexp(L_bal + u_s.unsqueeze(1), dim=0)
                u_s = -torch.logsumexp(L_bal + v_s.unsqueeze(0), dim=1)
            probs = torch.exp(L_bal + u_s.unsqueeze(1) + v_s.unsqueeze(0))
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            cdf = probs.cumsum(dim=1)
            u_qmc = (idx_f + ((phi * _PLASTIC_CONJ) % 1.0)) % 1.0
            j_idx = (cdf < u_qmc.unsqueeze(1)).sum(dim=1).clamp(0, b_n - 1)
            y_pair = x_real.detach()[j_idx]
        return ((1.0 - alpha) * x_fake.detach() + alpha * y_pair).detach(), phi

    def penalty(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
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
        quantized_fprop: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del kwargs
        if self.input_domain == "categorical" and (x_real.ndim != 2 or x_fake.ndim != 2):
            raise ValueError("categorical unified inputs require native [batch, categories] tensors")
        if (
            self.coeff == 0.0
            or (
                self.real_weight == 0.0
                and self.fake_weight == 0.0
                and self.gamma_gauge == 0.0
                and self.chord_weight == 0.0
            )
            or (self.lazy_k > 1 and step % self.lazy_k != 0)
        ):
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        self.record.calls += 1
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        b_r, b_f = int(x_real.shape[0]), int(x_fake.shape[0])
        dim = max(1, x_real[0].numel())
        is_image_tensor = x_real.ndim >= 3 and self.input_domain != "continuous"

        eff_beta = self.ebi_beta
        if self.settle_beta > 0.0 and bool(getattr(self.record, "t1c_thermal_initialized", False)):
            if self.settle_mode == "thermal":
                s_th = float(getattr(self.record, "_thermal_scale", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - s_th))
            else:
                res = float(getattr(self.record, "ema_residual", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - res / 0.22))
            eff_beta = eff_beta + self.settle_beta * (settle_u * settle_u)

        reused_endpoints = False

        try:
            if is_image_tensor:
                phi = float(getattr(self.record, "t1c_chord_phase", 0.0))
                if hasattr(self.record, "t1c_chord_phase"):
                    self.record.t1c_chord_phase = (phi + _GOLDEN_CONJ) % 1.0

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

                    if dim <= 1024 and not (is_conditioned and b_r != b_f):
                        # Small discrete image tensors (e.g. 8x8 / 16x16):
                        # K = 2 Gram-Schmidt orthogonal micro-secant probes on both x_real and x_fake + neighbor smoothing!
                        img_delta = 0.015
                        ur1, ur2 = _orthogonal_image_probe_pair(z_r, phi, salt=3)
                        uf1, uf2 = _orthogonal_image_probe_pair(z_f, phi, salt=11)
                        xr1 = (z_r.detach() + img_delta * ur1).detach()
                        xr2 = (z_r.detach() + img_delta * ur2).detach()
                        xf1 = (z_f.detach() + img_delta * uf1).detach()
                        xf2 = (z_f.detach() + img_delta * uf2).detach()

                        if can_concat_batches and not is_conditioned and xr1.shape[1:] == xf1.shape[1:]:
                            x_p_all = torch.cat((xr1, xr2, xf1, xf2), dim=0)
                            d_p_all = _format_logits_2d(live(x_p_all), 2 * b_r + 2 * b_f)
                            dr1, dr2, df1, df2 = torch.split(d_p_all, [b_r, b_r, b_f, b_f], dim=0)
                        else:
                            dr1 = _format_logits_2d(live(xr1), b_r)
                            dr2 = _format_logits_2d(live(xr2), b_r)
                            df1 = _format_logits_2d(live(xf1), b_f)
                            df2 = _format_logits_2d(live(xf2), b_f)

                        inv_d2 = 1.0 / (img_delta * img_delta)
                        sq_r_raw = (0.5 * float(dim) * inv_d2) * (
                            (dr1 - d_r_2d).square().sum(dim=-1) + (dr2 - d_r_2d).square().sum(dim=-1)
                        )
                        sq_f_raw = (0.5 * float(dim) * inv_d2) * (
                            (df1 - d_f_2d).square().sum(dim=-1) + (df2 - d_f_2d).square().sum(dim=-1)
                        )
                        sq_r = _local_neighbor_smooth(z_r.detach(), sq_r_raw, k_nn=4)
                        sq_f = _local_neighbor_smooth(z_f.detach(), sq_f_raw, k_nn=4)

                        norm_r = (sq_r + 1e-12).sqrt()
                        norm_f = (sq_f + 1e-12).sqrt()
                        total = x_real.new_zeros(())
                        if self.real_weight > 0.0:
                            total = total + self.real_weight * F.softplus(norm_r - self.kappa, beta=eff_beta).square().mean()
                            if self.lambda_floor > 0.0:
                                total = total + self.real_weight * self.lambda_floor * sq_r.mean()
                        if self.fake_weight > 0.0:
                            total = total + self.fake_weight * F.softplus(norm_f - self.kappa, beta=eff_beta).square().mean()
                            if self.lambda_floor > 0.0:
                                total = total + self.fake_weight * self.lambda_floor * sq_f.mean()
                        if self.gamma_gauge > 0.0:
                            mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                            gauge_pen = self.gamma_gauge * mean_logit.square()
                        else:
                            gauge_pen = x_real.new_zeros(())
                        norm_stat = 0.5 * (norm_r.detach().mean() + norm_f.detach().mean())
                    else:
                        # High-resolution RGB images (32x32..256x256):
                        # Live-endpoint reuse (0 endpoint forwards) + 1 BF16 separable micro-secant probe!
                        matched_idx_img = torch.arange(b_f, device=x_real.device, dtype=torch.long) % b_r
                        z_r_m = z_r[matched_idx_img].detach()
                        d_r_m = d_r_2d[matched_idx_img]

                        parity_mask = ((torch.arange(b_f, device=x_fake.device) + int(phi * 1024.0)) % 2) == 0
                        mask_nd = parity_mask.reshape((b_f,) + (1,) * (x_fake.ndim - 1))
                        z_base = torch.where(mask_nd, z_r_m, z_f.detach())
                        d_base = torch.where(parity_mask.unsqueeze(1), d_r_m, d_f_2d)

                        img_delta = 0.02
                        u_img = _separable_image_unit_perturbation(z_base, phi, salt=3)
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

                        eps_sq = self.secant_eps * self.secant_eps
                        step_sq_img = (
                            (z_p - z_base).flatten(1).float().square().sum(dim=1).add(eps_sq).to(dtype=d_f_2d.dtype)
                        )
                        micro_gn_sq = float(dim) * (_logit_diff_rms(d_p_2d, d_base).square() / step_sq_img)

                        dist_chord = (
                            (z_r_m - z_f.detach())
                            .flatten(1)
                            .float()
                            .square()
                            .sum(dim=1)
                            .add(eps_sq)
                            .sqrt()
                            .to(dtype=d_f_2d.dtype)
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
                        w_sum = self.real_weight + self.fake_weight
                        cap_pen = w_sum * F.softplus(norm_env - self.kappa, beta=eff_beta).square().mean()
                        floor_pen = (
                            w_sum
                            * 0.0005
                            * ((0.5 * micro_gn_sq.mean() + 0.5 * s_chord.square().mean()) / float(dim))
                            if self.lambda_floor > 0.0
                            else x_real.new_zeros(())
                        )
                        mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                        gauge_pen = self.gamma_gauge * mean_logit.square() if self.gamma_gauge > 0.0 else x_real.new_zeros(())
                        total = cap_pen + floor_pen
                        norm_stat = norm_env.detach().mean()
                finally:
                    self.record._in_penalty = False
            else:
                # Pure 1st-Order Forward-Mode AD (0 autograd.grad calls):
                use_chord = (
                    self.chord_weight > 0.0
                    and not is_conditioned
                    and x_real.ndim <= 2
                    and dim >= 2
                    and x_real.shape == x_fake.shape
                    and x_real.dtype == x_fake.dtype
                    and b_r > 0
                )
                if use_chord:
                    x_chord, phi = self._build_geodesic_chord(x_real, x_fake, is_conditioned=is_conditioned)
                else:
                    x_chord = None
                    phi = float(getattr(self.record, "t1c_chord_phase", 0.0))
                    if hasattr(self.record, "t1c_chord_phase"):
                        self.record.t1c_chord_phase = (phi + _GOLDEN_CONJ) % 1.0

                rep_fn = live_rep if live_rep is not None else (lambda z_in, _m: live(z_in))

                self.record._in_penalty = True
                try:
                    if dim <= 4:
                        if can_concat_batches and x_real.shape == x_fake.shape and x_real.dtype == x_fake.dtype:
                            stack_parts = [x_real, x_fake] + ([x_chord] if x_chord is not None else [])
                            x_fused = torch.cat(stack_parts, dim=0)
                            y_fused, sq_fused = _eval_lowd_forward_grad_sq(
                                rep_fn,
                                x_fused,
                                can_stack_basis=True,
                                quantized_fprop=quantized_fprop,
                                batch_mult=len(stack_parts),
                            )
                            d_r_2d, d_f_2d = y_fused[:b_r], y_fused[b_r : 2 * b_r]
                            sq_r, sq_f = sq_fused[:b_r], sq_fused[b_r : 2 * b_r]
                            sq_c = sq_fused[2 * b_r :] if x_chord is not None else None
                        else:
                            d_r_2d, sq_r = _eval_lowd_forward_grad_sq(
                                rep_fn, x_real, can_stack_basis=can_concat_batches, quantized_fprop=quantized_fprop
                            )
                            d_f_2d, sq_f = _eval_lowd_forward_grad_sq(
                                rep_fn, x_fake, can_stack_basis=can_concat_batches, quantized_fprop=quantized_fprop
                            )
                            sq_c = None
                            if x_chord is not None:
                                _, sq_c = _eval_lowd_forward_grad_sq(
                                    rep_fn, x_chord, can_stack_basis=can_concat_batches, quantized_fprop=quantized_fprop
                                )
                    else:
                        u_r, recon_r = _prepare_spatial_tangent_plan(
                            x_real.detach(), phi, salt=1, cond_tensors=cond_tensors
                        )
                        u_f, recon_f = _prepare_spatial_tangent_plan(
                            x_fake.detach(), phi, salt=2, cond_tensors=cond_tensors
                        )
                        x_eval_list: List[torch.Tensor] = [x_real, x_fake]
                        u_eval_list: List[torch.Tensor] = [u_r, u_f]
                        recon_c: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                        if x_chord is not None:
                            u_c, recon_c = _prepare_spatial_tangent_plan(
                                x_chord, phi, salt=3, cond_tensors=cond_tensors
                            )
                            x_eval_list.append(x_chord)
                            u_eval_list.append(u_c)

                        logits_list, tau_sq_list = _eval_spatial_tangent_sq(
                            live,
                            x_eval_list,
                            u_eval_list,
                            can_concat=bool(
                                can_concat_batches and not is_conditioned and x_real.shape[1:] == x_fake.shape[1:]
                            ),
                            quantized_fprop=quantized_fprop,
                        )
                        d_r_2d, d_f_2d = logits_list[0], logits_list[1]
                        sq_r = recon_r(tau_sq_list[0])
                        sq_f = recon_f(tau_sq_list[1])
                        sq_c = recon_c(tau_sq_list[2]) if (recon_c is not None and len(tau_sq_list) > 2) else None
                finally:
                    self.record._in_penalty = False

                norm_r = (sq_r + 1e-12).sqrt()
                norm_f = (sq_f + 1e-12).sqrt()
                cap_r = F.softplus(norm_r - self.kappa, beta=eff_beta).square().mean()
                cap_f = F.softplus(norm_f - self.kappa, beta=eff_beta).square().mean()
                if sq_c is not None:
                    norm_c = (sq_c + 1e-12).sqrt()
                    cap_c = F.softplus(norm_c - self.kappa, beta=eff_beta).square().mean()
                    total = self.real_weight * cap_r + self.fake_weight * (
                        (1.0 - self.chord_weight) * cap_f + self.chord_weight * cap_c
                    )
                else:
                    total = self.real_weight * cap_r + self.fake_weight * cap_f

                if self.lambda_floor > 0.0:
                    total = total + self.lambda_floor * (
                        self.real_weight * sq_r.mean() + self.fake_weight * sq_f.mean()
                    )
                if self.gamma_gauge > 0.0:
                    d_r_mean = d_r_2d.mean()
                    d_f_mean = d_f_2d.mean()
                    gauge_pen = self.gamma_gauge * (0.5 * (d_r_mean + d_f_mean)).square()
                else:
                    gauge_pen = x_real.new_zeros(())

                norm_stat = 0.5 * (norm_r.detach().mean() + norm_f.detach().mean())

            pen_lip = 0.5 * coefficient * total
            pen_gauge = 0.5 * coefficient * gauge_pen
            s_therm = float(getattr(self.record, "_thermal_scale", 1.0))
            if s_therm < 1.0:
                if pen_lip.requires_grad:
                    pen_lip = pen_lip * s_therm + pen_lip.detach() * (1.0 - s_therm)
                if pen_gauge.requires_grad:
                    s_gauge = math.sqrt(max(0.0, s_therm))
                    pen_gauge = pen_gauge * s_gauge + pen_gauge.detach() * (1.0 - s_gauge)
            pen = pen_lip + pen_gauge

            if not collect_stats:
                return pen, {}

            ell_t = getattr(self.record, "_t1c_ell_t", None)
            if ell_t is not None and ell_t.device == pen.device:
                pen_v, mean_norm, ell_v = torch.stack(
                    (pen.detach().float(), norm_stat.float(), ell_t.detach().float())
                ).tolist()
            else:
                pen_v, mean_norm = torch.stack((pen.detach().float(), norm_stat.float())).tolist()
                ell_v = float(getattr(self.record, "t1c_ell_ema", 0.0))
            return pen, {
                "applied": True,
                "pen": float(pen_v),
                "center": self.kappa,
                "phase": self.arm,
                "mean_grad_norm": float(mean_norm),
                "t1c_ell": float(ell_v),
                "s_thermal": float(s_therm),
                "live_endpoints_reused": bool(reused_endpoints),
            }
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()


class UnifiedCriticPenalty(_K3PCriticPenalty):
    """Recipe-facing Unified 1st-order Forward-Tangent & Multi-Probe Secant Born-Infeld critic penalty."""

    def __init__(self, recipe: Any, optimizer: Any, *, output: Any = None, collect_stats: bool = False, **penalty_overrides: Any) -> None:
        from ..discriminators import BatchDistanceDiscriminator
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (UnifiedCriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_unified_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _UnifiedPenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = isinstance(self.critic, nn.Module) and any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
        )
        self._has_batch_distance = isinstance(self.critic, nn.Module) and any(
            isinstance(m, BatchDistanceDiscriminator) for m in self.critic.modules()
        )
        self._has_quantized_fprop = isinstance(self.critic, nn.Module) and any(
            getattr(m, "format", None) in ("nvfp4", "mxfp8")
            or "Emulated" in type(m).__name__
            or "Quantized" in type(m).__name__
            for m in self.critic.modules()
        )

    def __call__(self, critic: nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, *condition: Any, **condition_kwargs: Any) -> torch.Tensor:
        from ..recipe_schedules import apply_penalty_schedule

        cid = id(critic)
        if cid not in self._verified_critics:
            self._ema_view(critic)
            self._verified_critics.add(cid)
        if getattr(self, "recipe", None) is not None:
            apply_penalty_schedule(self.optimizer.record.observed_steps, self.recipe, self)
        output = self.output

        def live(x: torch.Tensor) -> torch.Tensor:
            return output(critic(x, *condition, **condition_kwargs))

        base_b = int(x_real.shape[0])

        def live_rep(x: torch.Tensor, m: int) -> torch.Tensor:
            if m <= 1 or (not condition and not condition_kwargs) or x.shape[0] != m * base_b:
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
            isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == base_b
            for v in vars(critic).values()
        )
        uses_input_noise = float(getattr(critic, "std", 0.0)) > 0.0
        is_conditioned = bool(has_batch_state or condition or condition_kwargs)
        cond_tensors = [
            t.detach()
            for t in (*condition, *(condition_kwargs[k] for k in sorted(condition_kwargs)))
            if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == base_b
        ]
        cond_sig = _cond_signature((x_real, *condition), condition_kwargs)
        allow_live_reuse = (output is _first_output) and not has_batch_state
        can_concat_batches = (
            not self._has_batchnorm
            and not self._has_batch_distance
            and not has_batch_state
            and not uses_input_noise
            and (output is _first_output)
        )
        quantized_fprop = bool(
            self._has_quantized_fprop
            or (
                isinstance(critic, nn.Module)
                and any(
                    getattr(m, "format", None) in ("nvfp4", "mxfp8")
                    or "Emulated" in type(m).__name__
                    for m in critic.modules()
                )
            )
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
            quantized_fprop=quantized_fprop,
        )
        self.last_stats = stats
        if getattr(getattr(self, "recipe", None), "critic_step_mode", "none") == "finite_cap":
            real_c, fake_c = x_real.detach().clone(), x_fake.detach().clone()

            def finite_cap() -> torch.Tensor:
                _, sq_rc = _eval_lowd_forward_grad_sq(
                    live_rep, real_c, can_stack_basis=can_concat_batches, quantized_fprop=quantized_fprop
                )
                _, sq_fc = _eval_lowd_forward_grad_sq(
                    live_rep, fake_c, can_stack_basis=can_concat_batches, quantized_fprop=quantized_fprop
                )
                return torch.maximum((sq_rc + 1e-12).sqrt().max(), (sq_fc + 1e-12).sqrt().max())

            self.optimizer.bind_cap_panel(finite_cap, self.regularizer.kappa)
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            unified_ell=float(getattr(record, "t1c_ell_ema", 0.0)),
            unified_ell_initialized=bool(getattr(record, "t1c_ell_initialized", False)),
            unified_chord_phase=float(getattr(record, "t1c_chord_phase", 0.0)),
            unified_thermal_scale=float(getattr(record, "_thermal_scale", 1.0)),
            unified_step_scale=float(getattr(record, "_step_scale", 1.0)),
            unified_num_charts=0 if getattr(record, "chart_count", None) is None else int(record.chart_count.shape[0]),
        )
        return out


__all__ = [
    "UnifiedCriticPenalty",
    "_UnifiedPenaltyKernel",
]
