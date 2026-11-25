"""Unified-v11 1st-Order Spatial-Frame Forward-Tangent & Multi-Probe Secant Born-Infeld Critic Penalty."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._t1_coupling.charts import project_transport_features
from .._t1_coupling.primitive import T1CouplingLoss
from .._t2_critic.penalty import (
    _format_logits_2d,
    _murmur32_signs,
    _separable_image_unit_perturbation,
)
from .._unified.penalty import (
    _GOLDEN_CONJ,
    _PLASTIC_CONJ,
    _UnifiedPenaltyKernel,
    _adaptive_fd_delta,
    _eval_lowd_forward_grad_sq,
)
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .critic import (
    UnifiedV11CriticAdam,
    UnifiedV11StepRecord,
    _attach_unified_v11_critic_hook,
    _cond_signature,
)
from .optimizer import UnifiedV11EquilibratedOptimizer

_BAT_TANGENT_CACHE: Dict[Tuple[int, int, int, torch.device], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_batched_tangent_helpers(
    m: int, b: int, dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (m, b, dim, device)
    cached = _BAT_TANGENT_CACHE.get(key)
    if cached is None:
        salts_t = torch.arange(1, m + 1, device=device, dtype=torch.int64).view(m, 1, 1)
        col_idx = torch.arange(dim, device=device, dtype=torch.int64).view(1, 1, dim)
        arange_b = torch.arange(b, device=device, dtype=torch.int64).view(1, b).expand(m, b)
        cached = (salts_t, col_idx, arange_b)
        if len(_BAT_TANGENT_CACHE) < 64:
            _BAT_TANGENT_CACHE[key] = cached
    return cached


@torch.no_grad()
def _prepare_batched_highd_tangent_plans(
    x_stack: torch.Tensor,
    phase: float,
    *,
    pool_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bit-exact batched ``[M, B, D]`` version of ``_prepare_spatial_tangent_plan`` for unconditioned ``dim > 8``."""
    m, b, dim = int(x_stack.shape[0]), int(x_stack.shape[1]), int(x_stack.shape[2])
    flat = x_stack.float()
    salts_t, col_idx, arange_b = _get_batched_tangent_helpers(m, b, dim, flat.device)
    proj = flat[:, :, 0] + 0.61803398875 * flat[:, :, 1]
    order = torch.argsort(proj, dim=1, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, arange_b)
    p_base = int(phase * 1073741824.0)
    p_int = (p_base + salts_t * 7919) & 0xFFFFFFFF
    h = (ranks.unsqueeze(2) * 2654435761 + col_idx * 2246822519 + p_int * 3266489917 + 374761393) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    u_all = (torch.where((h & 1) == 0, 1.0, -1.0) / math.sqrt(float(dim))).to(dtype=x_stack.dtype)

    k_eff = min(pool_k, b)
    d2 = torch.stack([torch.cdist(flat[i], flat[i], p=2.0) for i in range(m)], dim=0).pow(2)
    knn_d2, knn_idx = torch.topk(d2, k=k_eff, dim=2, largest=False)
    bw2 = knn_d2[:, :, max(1, k_eff // 2)].clamp_min(1e-5).unsqueeze(2)
    weights = torch.exp(-0.5 * knn_d2 / bw2)
    weights.div_(weights.sum(dim=2, keepdim=True).clamp_min_(1e-12))
    return u_all, knn_idx, weights


def _universal_image_unit_tangent(x: torch.Tensor, phase: float, salt: int) -> torch.Tensor:
    """Zero-RNG isotropic unit tangent ``||u_i||_2 = 1`` for any spatial tensor rank >= 3."""
    b = int(x.shape[0])
    d = max(1, int(x[0].numel()))
    if d <= 8192:
        ranks = torch.arange(b, device=x.device, dtype=torch.int64)
        u = _murmur32_signs(ranks, d, phase, salt, x.device)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return u.to(dtype=x.dtype).reshape_as(x)
    return _separable_image_unit_perturbation(x, phase, salt=salt)


def _bn_dual_bf16_hook(
    _module: nn.Module,
    _inputs: Tuple[Any, ...],
    output: Any,
) -> Any:
    """Keep BatchNorm dual primal + tangent aligned in BF16 during autocast fwAD passes."""
    if isinstance(output, torch.Tensor):
        primal, tangent = fwAD.unpack_dual(output)
        if tangent is not None and (primal.dtype != torch.bfloat16 or tangent.dtype != torch.bfloat16):
            return fwAD.make_dual(primal.to(dtype=torch.bfloat16), tangent.to(dtype=torch.bfloat16))
    return output


class _UnifiedV11PenaltyKernel(_UnifiedPenaltyKernel):
    """Unified-v11 universal 1st-order forward-AD Born-Infeld critic kernel (`0` `autograd.grad`)."""

    def __init__(self, *, record: Any = None, **kwargs: Any) -> None:
        rec = UnifiedV11StepRecord() if record is None else record
        super().__init__(record=rec, **kwargs)
        self.arm = "unified_v11"

    @torch.no_grad()
    def _build_geodesic_chord(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *,
        is_conditioned: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Exact fast-path geodesic chord: zero-sync ``ell_t``, fast sort quantile, and 1-buffer in-place Sinkhorn."""
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
            ell_t: Optional[torch.Tensor] = None
            reusable_buf: Optional[torch.Tensor] = None
            if ell_init:
                ell_cached = getattr(self.record, "_t1c_ell_t", None)
                if ell_cached is not None and ell_cached.device == x_real.device:
                    ell_t = ell_cached.detach().float()
                else:
                    ell_s = float(getattr(self.record, "t1c_ell_ema", 0.0))
                    if ell_s > 1e-6:
                        ell_t = x_real.new_full((), ell_s, dtype=torch.float32)
            if ell_t is None:
                y_flat = project_transport_features(x_real.detach(), max_spatial=8).float()
                dim_y = max(1, y_flat.shape[-1])
                akt_exp = max(0.0, (1.0 / float(dim_y)) - 0.5)
                if self.akt_horizon and akt_exp > 0.0:
                    if self.akt_mode == "weibull":
                        q_rate = max(1e-6, -math.log(max(1e-6, 1.0 - 0.25)))
                        akt_factor = ((2.0 * math.e * float(b_n)) / q_rate) ** akt_exp
                    else:
                        akt_factor = float(b_n) ** akt_exp
                else:
                    akt_factor = 1.0
                scale = y_flat.abs().amax().clamp_min(1.0)
                y_unit = y_flat / scale
                dist_unit = torch.cdist(y_unit, y_unit, p=2.0)
                dist_unit.fill_diagonal_(0.0)
                d_max = dist_unit.amax()
                thresh = 1e-5 * d_max
                dist_unit.fill_diagonal_(float("inf"))
                k_nn = min(b_n - 1, max(1, b_n // 200))
                if k_nn == 1:
                    nn_dist = dist_unit.amin(dim=-1)
                else:
                    nn_dist = torch.topk(dist_unit, k=k_nn, dim=-1, largest=False).values[:, -1]
                s_nn = nn_dist.sort().values
                pos_q = 0.25 * float(b_n - 1)
                lo_q = int(math.floor(pos_q))
                hi_q = int(math.ceil(pos_q))
                q_raw = s_nn[lo_q] if lo_q == hi_q else torch.lerp(s_nn[lo_q], s_nn[hi_q], pos_q - float(lo_q))
                dist_unit.masked_fill_(dist_unit <= thresh, float("inf"))
                pos_min = dist_unit.amin()
                rms_unit = y_unit.square().mean().sqrt()
                tie_base = torch.where(torch.isfinite(pos_min), 0.15 * pos_min, 0.05 * d_max)
                raw_unit = torch.where(q_raw <= thresh, tie_base, q_raw)
                nonzero_val = scale * akt_factor * raw_unit + 1e-6
                dirac_val = 0.05 * torch.clamp_min(scale * rms_unit, 1.0) + 1e-6
                ell_t = torch.where(d_max == 0.0, dirac_val, nonzero_val)
                reusable_buf = dist_unit

            c_hat = T1CouplingLoss._born_infeld_c_hat_detached(x_fake, x_real, ell_t)
            L_bal = -c_hat.float() * 4.0
            buf = reusable_buf if (reusable_buf is not None and reusable_buf.shape == L_bal.shape) else torch.empty_like(L_bal)
            # 6-iteration log-domain Sinkhorn with 1 reusable buffer (exact dim=0 / dim=1 reduction order):
            m0 = L_bal.amax(dim=0, keepdim=True)
            torch.sub(L_bal, m0, out=buf)
            buf.exp_()
            v_s = -(m0.squeeze(0) + buf.sum(dim=0).log_())
            torch.add(L_bal, v_s.unsqueeze(0), out=buf)
            m1 = buf.amax(dim=1, keepdim=True)
            buf.sub_(m1).exp_()
            u_s = -(m1.squeeze(1) + buf.sum(dim=1).log_())
            for _ in range(5):
                torch.add(L_bal, u_s.unsqueeze(1), out=buf)
                m0 = buf.amax(dim=0, keepdim=True)
                buf.sub_(m0).exp_()
                v_s = -(m0.squeeze(0) + buf.sum(dim=0).log_())
                torch.add(L_bal, v_s.unsqueeze(0), out=buf)
                m1 = buf.amax(dim=1, keepdim=True)
                buf.sub_(m1).exp_()
                u_s = -(m1.squeeze(1) + buf.sum(dim=1).log_())
            torch.add(L_bal, u_s.unsqueeze(1), out=buf)
            buf.add_(v_s.unsqueeze(0)).exp_()
            buf.div_(buf.sum(dim=1, keepdim=True).clamp_min_(1e-12))
            cdf = buf.cumsum_(dim=1)
            u_qmc = (idx_f + ((phi * _PLASTIC_CONJ) % 1.0)) % 1.0
            j_idx = (cdf < u_qmc.unsqueeze(1)).sum(dim=1).clamp_(0, b_n - 1)
            y_pair = x_real.detach()[j_idx]
        return ((1.0 - alpha) * x_fake.detach() + alpha * y_pair).detach(), phi

    def _penalty_highd_batched(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        can_concat_batches: bool = False,
        quantized_fprop: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Bit-exact batched high-D (``dim > 8``, unconditioned 2D) penalty eliminating 3x Murmur32/argsort/topk."""
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
        b_n = int(x_real.shape[0])
        dim = int(x_real.shape[1])

        eff_beta = self.ebi_beta
        if self.settle_beta > 0.0 and bool(getattr(self.record, "t1c_thermal_initialized", False)):
            if self.settle_mode == "thermal":
                s_th = float(getattr(self.record, "_thermal_scale", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - s_th))
            else:
                res = float(getattr(self.record, "ema_residual", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - res / 0.22))
            eff_beta = eff_beta + self.settle_beta * (settle_u * settle_u)

        try:
            use_chord = self.chord_weight > 0.0
            if use_chord:
                x_chord, phi = self._build_geodesic_chord(x_real, x_fake, is_conditioned=False)
                x_stack = torch.stack((x_real.detach(), x_fake.detach(), x_chord), dim=0)
                m_cnt = 3
            else:
                phi = float(getattr(self.record, "t1c_chord_phase", 0.0))
                if hasattr(self.record, "t1c_chord_phase"):
                    self.record.t1c_chord_phase = (phi + _GOLDEN_CONJ) % 1.0
                x_stack = torch.stack((x_real.detach(), x_fake.detach()), dim=0)
                m_cnt = 2

            self.record._in_penalty = True
            try:
                u_all, knn_idx, weights = _prepare_batched_highd_tangent_plans(x_stack, phi)
                if can_concat_batches:
                    x_cat = x_stack.reshape(m_cnt * b_n, dim)
                    u_cat = u_all.reshape(m_cnt * b_n, dim)
                    y_2d: Optional[torch.Tensor] = None
                    tau_sq_cat: Optional[torch.Tensor] = None
                    if not quantized_fprop:
                        try:
                            with fwAD.dual_level():
                                yd = live(fwAD.make_dual(x_cat, u_cat))
                                y_cat, jvp_cat = fwAD.unpack_dual(yd)
                            if jvp_cat is None:
                                raise NotImplementedError("dual tangent dropped")
                            y_2d = _format_logits_2d(y_cat, m_cnt * b_n)
                            tau_sq_cat = _format_logits_2d(jvp_cat, m_cnt * b_n).square().sum(dim=-1)
                        except (RuntimeError, NotImplementedError, TypeError):
                            y_2d = None
                    if y_2d is None or tau_sq_cat is None:
                        fd_delta = _adaptive_fd_delta(x_cat, quantized_fprop=True)
                        y_p = _format_logits_2d(live((x_cat + fd_delta * u_cat).detach()), m_cnt * b_n)
                        y_m = _format_logits_2d(live((x_cat - fd_delta * u_cat).detach()), m_cnt * b_n)
                        y_2d = 0.5 * (y_p + y_m)
                        step_sq = (2.0 * fd_delta * u_cat.float()).square().sum(dim=1).clamp_min(1e-12)
                        tau_sq_cat = (y_p - y_m).square().sum(dim=-1) / step_sq.to(dtype=y_2d.dtype)
                    d_r_2d = y_2d[:b_n]
                    d_f_2d = y_2d[b_n : 2 * b_n]
                    tau_m = tau_sq_cat.view(m_cnt, b_n)
                else:
                    logits_parts: list[torch.Tensor] = []
                    tau_parts: list[torch.Tensor] = []
                    for i in range(m_cnt):
                        x_i = x_stack[i]
                        u_i = u_all[i]
                        y_i_2d: Optional[torch.Tensor] = None
                        t_i_sq: Optional[torch.Tensor] = None
                        if not quantized_fprop:
                            try:
                                with fwAD.dual_level():
                                    yd = live(fwAD.make_dual(x_i, u_i))
                                    y_i, jvp_i = fwAD.unpack_dual(yd)
                                if jvp_i is None:
                                    raise NotImplementedError("dual tangent dropped")
                                y_i_2d = _format_logits_2d(y_i, b_n)
                                t_i_sq = _format_logits_2d(jvp_i, b_n).square().sum(dim=-1)
                            except (RuntimeError, NotImplementedError, TypeError):
                                y_i_2d = None
                        if y_i_2d is None or t_i_sq is None:
                            fd_delta = _adaptive_fd_delta(x_i, quantized_fprop=True)
                            y_p = _format_logits_2d(live((x_i + fd_delta * u_i).detach()), b_n)
                            y_m = _format_logits_2d(live((x_i - fd_delta * u_i).detach()), b_n)
                            y_i_2d = 0.5 * (y_p + y_m)
                            step_sq = (2.0 * fd_delta * u_i.float()).square().sum(dim=1).clamp_min(1e-12)
                            t_i_sq = (y_p - y_m).square().sum(dim=-1) / step_sq.to(dtype=y_i_2d.dtype)
                        logits_parts.append(y_i_2d)
                        tau_parts.append(t_i_sq)
                    d_r_2d, d_f_2d = logits_parts[0], logits_parts[1]
                    tau_m = torch.stack(tau_parts, dim=0)

                k_eff = int(knn_idx.shape[2])
                gathered = torch.gather(tau_m, 1, knn_idx.reshape(m_cnt, b_n * k_eff)).view(m_cnt, b_n, k_eff)
                sq_all = float(dim) * (weights.to(dtype=tau_m.dtype) * gathered).sum(dim=2)
                sq_r, sq_f = sq_all[0], sq_all[1]
                sq_c = sq_all[2] if m_cnt == 3 else None
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
                mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                gauge_pen = self.gamma_gauge * mean_logit.square()
            else:
                mean_logit = x_real.new_zeros(())
                gauge_pen = x_real.new_zeros(())

            norm_stat = 0.5 * (norm_r.detach().mean() + norm_f.detach().mean())
            pen_lip = 0.5 * coefficient * total
            pen_gauge = 0.5 * coefficient * gauge_pen
            s_therm = float(getattr(self.record, "_thermal_scale", 1.0))
            if s_therm < 1.0:
                if pen_lip.requires_grad:
                    pen_lip = pen_lip * s_therm + pen_lip.detach() * (1.0 - s_therm)
                if pen_gauge.requires_grad:
                    g_u = ((mean_logit.detach().abs() - 0.5) / 1.0).clamp(0.0, 1.0)
                    g_gate = g_u * g_u * (3.0 - 2.0 * g_u)
                    s_gauge = s_therm + g_gate * (math.sqrt(max(0.0, s_therm)) - s_therm)
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
                "live_endpoints_reused": False,
            }
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()

    def _eval_image_fwad_sq(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        x_probe: torch.Tensor,
        phi: float,
        *,
        salt: int,
        dim: int,
        has_batchnorm: bool,
        has_instancenorm: bool = False,
        critic_module: Optional[nn.Module],
        quantized_fprop: bool,
    ) -> torch.Tensor:
        """Evaluate ``dim * ||J_D(x) u||_2^2`` via 1st-order forward-mode AD (0 `autograd.grad`)."""
        b_p = int(x_probe.shape[0])
        u_p = _universal_image_unit_tangent(x_probe, phi, salt=salt)
        bn_modes = []
        bn_hooks = []
        bf16_active = bool(
            x_probe.is_cuda
            and (
                (isinstance(critic_module, nn.Module) and bool(getattr(critic_module, "bf16", False)))
                or (
                    torch.is_autocast_enabled("cuda")
                    and torch.get_autocast_dtype("cuda") == torch.bfloat16
                )
            )
        )
        use_bf16_dual_hook = bool(
            (has_batchnorm or has_instancenorm)
            and bf16_active
            and isinstance(critic_module, nn.Module)
        )
        if (has_batchnorm or (has_instancenorm and use_bf16_dual_hook)) and isinstance(critic_module, nn.Module):
            for mod in critic_module.modules():
                if has_batchnorm and isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    bn_modes.append((mod, mod.training))
                    mod.eval()
                    if use_bf16_dual_hook:
                        bn_hooks.append(mod.register_forward_hook(_bn_dual_bf16_hook))
                elif has_instancenorm and use_bf16_dual_hook and isinstance(mod, torch.nn.modules.instancenorm._InstanceNorm):
                    bn_hooks.append(mod.register_forward_hook(_bn_dual_bf16_hook))
        try:
            if not quantized_fprop:
                try:
                    if use_bf16_dual_hook:
                        x_in = x_probe.to(dtype=torch.bfloat16)
                        u_in = u_p.to(dtype=torch.bfloat16)
                    else:
                        x_in = x_probe
                        u_in = u_p
                    with fwAD.dual_level():
                        yd = live(fwAD.make_dual(x_in, u_in))
                        _, jvp = fwAD.unpack_dual(yd)
                    if jvp is None:
                        raise NotImplementedError("dual tangent dropped")
                    j2d = _format_logits_2d(jvp.float(), b_p)
                    return float(dim) * j2d.square().mean(dim=-1)
                except (RuntimeError, NotImplementedError, TypeError):
                    if bf16_active:
                        # Clean FP32 fwAD fallback if an exotic norm/custom layer disagrees with BF16 duals.
                        for h in bn_hooks:
                            h.remove()
                        bn_hooks.clear()
                        prev_bf16 = isinstance(critic_module, nn.Module) and bool(getattr(critic_module, "bf16", False))
                        if prev_bf16:
                            critic_module.bf16 = False
                        try:
                            with torch.autocast(device_type="cuda", enabled=False):
                                with fwAD.dual_level():
                                    yd = live(fwAD.make_dual(x_probe.float(), u_p.float()))
                                    _, jvp = fwAD.unpack_dual(yd)
                            if jvp is not None:
                                j2d = _format_logits_2d(jvp.float(), b_p)
                                return float(dim) * j2d.square().mean(dim=-1)
                        except (RuntimeError, NotImplementedError, TypeError):
                            pass
                        finally:
                            if prev_bf16:
                                critic_module.bf16 = True
            # Fallback strictly for custom quantized/non-jvp ops: scale-aware symmetric central diff.
            fd_delta = _adaptive_fd_delta(x_probe, quantized_fprop=True)
            xp_p = (x_probe.float() + fd_delta * u_p.float()).to(dtype=x_probe.dtype).detach()
            xp_m = (x_probe.float() - fd_delta * u_p.float()).to(dtype=x_probe.dtype).detach()
            yp = _format_logits_2d(live(xp_p).float(), b_p)
            ym = _format_logits_2d(live(xp_m).float(), b_p)
            step_sq = (xp_p.float() - xp_m.float()).flatten(1).square().sum(dim=1).clamp_min(1e-12)
            return float(dim) * ((yp - ym).square().mean(dim=-1) / step_sq)
        finally:
            for h in bn_hooks:
                h.remove()
            for mod, flag in bn_modes:
                mod.training = flag

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
        has_batchnorm: bool = False,
        has_instancenorm: bool = False,
        critic_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        is_image_tensor = x_real.ndim >= 3 and self.input_domain != "continuous"
        if not is_image_tensor:
            b_r_ni = int(x_real.shape[0])
            dim_ni = max(1, int(x_real[0].numel())) if b_r_ni > 0 else 0
            if (
                not is_conditioned
                and x_real.ndim == 2
                and dim_ni > 8
                and b_r_ni >= 2
                and x_real.shape == x_fake.shape
                and x_real.dtype == x_fake.dtype
            ):
                res = self._penalty_highd_batched(
                    live,
                    x_real,
                    x_fake,
                    step,
                    collect_stats,
                    can_concat_batches=can_concat_batches,
                    quantized_fprop=quantized_fprop,
                )
            else:
                res = super().penalty(
                    live,
                    x_real,
                    x_fake,
                    step,
                    collect_stats,
                    is_conditioned=is_conditioned,
                    cond_tensors=cond_tensors,
                    cond_sig=cond_sig,
                    allow_live_reuse=allow_live_reuse,
                    can_concat_batches=can_concat_batches,
                    live_rep=live_rep,
                    quantized_fprop=quantized_fprop,
                )
            if hasattr(self.record, "observe_scale") and not getattr(self.record, "_capture_ring", None):
                self.record.observe_scale(x_real, x_fake)
            return res

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
        dim = max(1, int(x_real[0].numel()))

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
                    z_r = z_r.detach().to(dtype=x_real.dtype)
                else:
                    self.record._last_probe_actual_input = None
                    x_r_det = x_real.detach()
                    d_r_raw = live(x_r_det)
                    act_r = getattr(self.record, "_last_probe_actual_input", None)
                    z_r = (
                        act_r.detach().to(dtype=x_real.dtype)
                        if isinstance(act_r, torch.Tensor) and act_r.shape == x_r_det.shape
                        else x_r_det
                    )

                if hit_f is not None:
                    z_f, d_f_raw = hit_f
                    z_f = z_f.detach().to(dtype=x_fake.dtype)
                else:
                    self.record._last_probe_actual_input = None
                    x_f_det = x_fake.detach()
                    d_f_raw = live(x_f_det)
                    act_f = getattr(self.record, "_last_probe_actual_input", None)
                    z_f = (
                        act_f.detach().to(dtype=x_fake.dtype)
                        if isinstance(act_f, torch.Tensor) and act_f.shape == x_f_det.shape
                        else x_f_det
                    )

                d_r_2d = _format_logits_2d(d_r_raw, b_r)
                d_f_2d = _format_logits_2d(d_f_raw, b_f)

                # Universal 1st-order forward-AD probe across ALL image spatial dimensions (8x8..256x256):
                # 1 interleaved half-real/half-fake batch (BN in eval -> zero cross-sample BN leakage).
                if not is_conditioned and b_r == b_f and b_r >= 2:
                    shift = int(phi * 1024.0) & 1
                    parity = ((torch.arange(b_f, device=x_real.device) + shift) % 2) == 0
                    mask_nd = parity.reshape((b_f,) + (1,) * (x_real.ndim - 1))
                    x_probe = torch.where(mask_nd, z_r, z_f)
                    sq_mix = self._eval_image_fwad_sq(
                        live,
                        x_probe,
                        phi,
                        salt=3,
                        dim=dim,
                        has_batchnorm=has_batchnorm,
                        has_instancenorm=has_instancenorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )
                    if shift == 0:
                        sq_r_raw = sq_mix[0::2]
                        sq_f_raw = sq_mix[1::2]
                    else:
                        sq_r_raw = sq_mix[1::2]
                        sq_f_raw = sq_mix[0::2]
                else:
                    sq_r_raw = self._eval_image_fwad_sq(
                        live,
                        z_r,
                        phi,
                        salt=3,
                        dim=dim,
                        has_batchnorm=has_batchnorm,
                        has_instancenorm=has_instancenorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )
                    sq_f_raw = self._eval_image_fwad_sq(
                        live,
                        z_f,
                        phi,
                        salt=11,
                        dim=dim,
                        has_batchnorm=has_batchnorm,
                        has_instancenorm=has_instancenorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )

                # Universal per-sample Born-Infeld cap on unbiased 1-probe forward-AD ||grad_x D||^2:
                norm_r = (sq_r_raw + 1e-12).sqrt()
                norm_f = (sq_f_raw + 1e-12).sqrt()
                total = x_real.new_zeros(())
                if self.real_weight > 0.0:
                    total = total + self.real_weight * F.softplus(norm_r - self.kappa, beta=eff_beta).square().mean()
                    if self.lambda_floor > 0.0:
                        total = total + self.real_weight * (0.0005 / float(dim)) * sq_r_raw.mean()
                if self.fake_weight > 0.0:
                    total = total + self.fake_weight * F.softplus(norm_f - self.kappa, beta=eff_beta).square().mean()
                    if self.lambda_floor > 0.0:
                        total = total + self.fake_weight * (0.0005 / float(dim)) * sq_f_raw.mean()
                if self.gamma_gauge > 0.0:
                    eff_gauge = min(self.gamma_gauge, 0.005)
                    mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                    gauge_pen = eff_gauge * mean_logit.square()
                else:
                    mean_logit = x_real.new_zeros(())
                    gauge_pen = x_real.new_zeros(())
                norm_stat = 0.5 * (norm_r.detach().mean() + norm_f.detach().mean())
            finally:
                self.record._in_penalty = False

            if hasattr(self.record, "observe_scale") and not getattr(self.record, "_capture_ring", None):
                self.record.observe_scale(z_r, z_f)

            pen_lip = 0.5 * coefficient * total
            pen_gauge = 0.5 * coefficient * gauge_pen
            s_therm = float(getattr(self.record, "_thermal_scale", 1.0))
            if s_therm < 1.0:
                if pen_lip.requires_grad:
                    pen_lip = pen_lip * s_therm + pen_lip.detach() * (1.0 - s_therm)
                if pen_gauge.requires_grad:
                    g_u = ((mean_logit.detach().abs() - 0.5) / 1.0).clamp(0.0, 1.0)
                    g_gate = g_u * g_u * (3.0 - 2.0 * g_u)
                    s_gauge = s_therm + g_gate * (math.sqrt(max(0.0, s_therm)) - s_therm)
                    pen_gauge = pen_gauge * s_gauge + pen_gauge.detach() * (1.0 - s_gauge)
            pen = pen_lip + pen_gauge

            if not collect_stats:
                return pen, {}

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


class UnifiedV11CriticPenalty(_K3PCriticPenalty):
    """Recipe-facing Unified-v11 1st-order Forward-Tangent & Multi-Probe Secant critic penalty."""

    def __init__(
        self,
        recipe: Any,
        optimizer: Any,
        *,
        output: Any = None,
        collect_stats: bool = False,
        **penalty_overrides: Any,
    ) -> None:
        from ..discriminators import BatchDistanceDiscriminator
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (UnifiedV11CriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (
                isinstance(optimizer, (NormalizedOptimizer, UnifiedV11EquilibratedOptimizer))
                and hasattr(optimizer, "record")
            )
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_unified_v11_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _UnifiedV11PenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = isinstance(self.critic, nn.Module) and any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
        )
        self._has_instancenorm = isinstance(self.critic, nn.Module) and any(
            isinstance(m, torch.nn.modules.instancenorm._InstanceNorm) for m in self.critic.modules()
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

    def __call__(
        self,
        critic: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *condition: Any,
        **condition_kwargs: Any,
    ) -> torch.Tensor:
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
                critic is not self.critic
                and isinstance(critic, nn.Module)
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
            has_batchnorm=self._has_batchnorm,
            has_instancenorm=self._has_instancenorm,
            critic_module=critic,
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
        bank = getattr(record, "cond_fiber_bank", None)
        out.update(
            unified_v11_ell=float(getattr(record, "t1c_ell_ema", 0.0)),
            unified_v11_ell_initialized=bool(getattr(record, "t1c_ell_initialized", False)),
            unified_v11_chord_phase=float(getattr(record, "t1c_chord_phase", 0.0)),
            unified_v11_thermal_scale=float(getattr(record, "_thermal_scale", 1.0)),
            unified_v11_step_scale=float(getattr(record, "_step_scale", 1.0)),
            unified_v11_num_charts=0 if getattr(record, "chart_count", None) is None else int(record.chart_count.shape[0]),
            unified_v11_cond_strata=0 if bank is None else len(bank.strata),
        )
        return out


__all__ = [
    "UnifiedV11CriticPenalty",
    "_UnifiedV11PenaltyKernel",
]
