"""Unified-v15 1st-Order Forward-Mode AD & Central-Secant Born-Infeld Critic Penalty (0 autograd.grad)."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .charts import project_transport_features
from .loss import UnifiedV15CouplingLoss as T1CouplingLoss
from .._t2_critic.penalty import (
    _format_logits_2d,
    _murmur32_signs,
    _separable_image_unit_perturbation,
)
from .._unified.penalty import (
    _GOLDEN_CONJ,
    _PLASTIC_CONJ,
    _UnifiedPenaltyKernel,
    _eval_lowd_forward_grad_sq,
)
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .critic import (
    UnifiedV15CriticAdam,
    UnifiedV15StepRecord,
    _attach_unified_v15_critic_hook,
    _cond_signature,
)
from .._unified_v11.optimizer import UnifiedV11EquilibratedOptimizer
from .optimizer import UnifiedV15Optimizer

_BAT_TANGENT_CACHE: Dict[Tuple[int, int, int, torch.device], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

_SECANT_MODULE_TYPES = (
    nn.MultiheadAttention,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    nn.Softmax,
    nn.LogSoftmax,
    nn.Softmax2d,
)
_SECANT_CODE_NAMES = frozenset({"scaled_dot_product_attention", "softmax", "log_softmax"})


def _has_spatial_dims(x: torch.Tensor) -> bool:
    """Return True iff ``x`` has non-singleton spatial dimensions beyond ``(B, C)``."""
    return x.ndim >= 3 and any(int(s) > 1 for s in x.shape[2:])


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
    """Bit-exact batched ``[M, B, D]`` version of ``_prepare_spatial_tangent_plan`` for unconditioned ``dim > 16``."""
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
    bw2 = knn_d2[:, :, min(k_eff - 1, max(1, k_eff // 2))].clamp_min(1e-5).unsqueeze(2)
    weights = torch.exp(-0.5 * knn_d2 / bw2)
    weights.div_(weights.sum(dim=2, keepdim=True).clamp_min_(1e-12))
    return u_all, knn_idx, weights


def _universal_image_unit_tangent(x: torch.Tensor, phase: float, salt: int) -> torch.Tensor:
    """Zero-RNG isotropic unit tangent ``||u_i||_2 = 1`` for any tensor."""
    b = int(x.shape[0])
    d = max(1, int(x[0].numel()))
    if d <= 8192:
        ranks = torch.arange(b, device=x.device, dtype=torch.int64)
        u = _murmur32_signs(ranks, d, phase, salt, x.device)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return u.to(dtype=x.dtype).reshape_as(x)
    return _separable_image_unit_perturbation(x, phase, salt=salt)


@torch.no_grad()
def _orthogonal_unit_probes(
    x: torch.Tensor,
    phase: float,
    *,
    salt: int,
    k_probes: int,
) -> List[torch.Tensor]:
    """Generate ``k_probes`` zero-RNG per-sample orthonormal unit probes ``||u_k||_2 = 1``."""
    b = int(x.shape[0])
    d = max(1, int(x[0].numel()))
    k_eff = max(1, min(int(k_probes), d))
    ranks = torch.arange(b, device=x.device, dtype=torch.int64)
    probes_f: List[torch.Tensor] = []
    for k in range(k_eff):
        ph_k = (phase + float(k) * _GOLDEN_CONJ) % 1.0
        salt_k = salt + 31 * k
        if d <= 8192:
            u = _murmur32_signs(ranks + 257 * k, d, ph_k, salt_k, x.device)
        else:
            u = _separable_image_unit_perturbation(x, ph_k, salt=salt_k).reshape(b, d).float()
        for prev in probes_f:
            u = u - (u * prev).sum(dim=1, keepdim=True) * prev
        nrm = u.norm(dim=1, keepdim=True)
        bad = nrm.squeeze(1) < 1e-5
        if bool(bad.any()) and probes_f:
            alt = torch.roll(probes_f[-1], shifts=1, dims=1)
            alt[:, 0] = -alt[:, 0]
            for prev in probes_f:
                alt = alt - (alt * prev).sum(dim=1, keepdim=True) * prev
            u = torch.where(bad.unsqueeze(1), alt, u)
            nrm = u.norm(dim=1, keepdim=True)
        probes_f.append(u / nrm.clamp_min(1e-8))
    return [u.to(dtype=x.dtype).reshape_as(x) for u in probes_f]


def _bn_dual_bf16_hook(
    _module: nn.Module,
    _inputs: Tuple[Any, ...],
    output: Any,
) -> Any:
    """Keep BatchNorm/InstanceNorm dual primal + tangent aligned in BF16 during autocast fwAD passes."""
    if isinstance(output, torch.Tensor):
        primal, tangent = fwAD.unpack_dual(output)
        if tangent is not None and (primal.dtype != torch.bfloat16 or tangent.dtype != torch.bfloat16):
            return fwAD.make_dual(primal.to(dtype=torch.bfloat16), tangent.to(dtype=torch.bfloat16))
    return output


def _critic_needs_secant(critic: Optional[nn.Module]) -> bool:
    """Detect Attention/SDPA/Softmax blocks that fail fwAD or fwAD+backward and require central secant."""
    if not isinstance(critic, nn.Module):
        return False
    for mod in critic.modules():
        if isinstance(mod, _SECANT_MODULE_TYPES):
            return True
        fwd = getattr(type(mod), "forward", None)
        code = getattr(fwd, "__code__", None)
        if code is not None and (_SECANT_CODE_NAMES & set(code.co_names)):
            return True
    return False


def _critic_has_non_dcgan_layers(critic: Optional[nn.Module]) -> bool:
    """Detect GroupNorm, LayerNorm, or Attention/Transformer blocks for floor/gauge parity."""
    if not isinstance(critic, nn.Module):
        return False
    if _critic_needs_secant(critic):
        return True
    norm_types = (nn.GroupNorm, nn.LayerNorm)
    return any(isinstance(m, norm_types) for m in critic.modules())


def _is_bf16_eval_context(x_probe: torch.Tensor, critic_module: Optional[nn.Module]) -> Tuple[bool, bool]:
    """Return ``(is_bf16, cuda_bf16_autocast)`` for dtype-aware fwAD hooks and central-secant step size."""
    cuda_bf16_autocast = bool(
        x_probe.is_cuda
        and (
            (isinstance(critic_module, nn.Module) and bool(getattr(critic_module, "bf16", False)))
            or (
                torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16
            )
        )
    )
    is_bf16 = bool(
        cuda_bf16_autocast
        or x_probe.dtype == torch.bfloat16
        or (
            isinstance(critic_module, nn.Module)
            and any(p.dtype == torch.bfloat16 for p in critic_module.parameters())
        )
    )
    return is_bf16, cuda_bf16_autocast


def _secant_step_size(
    x_probe: torch.Tensor,
    *,
    is_bf16: bool,
    quantized_fprop: bool,
) -> float:
    """Compute scale-equivariant, dtype-aware central-secant step ``h`` (1e-3*scale FP32, 1e-2*scale BF16)."""
    with torch.no_grad():
        scale = float(x_probe.detach().float().flatten(1).norm(dim=1).mean().item())
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = math.sqrt(float(max(1, int(x_probe[0].numel()))))
    h_coeff = 1e-2 if is_bf16 else 1e-3
    if quantized_fprop:
        h_coeff = max(h_coeff, 0.04)
    return float(h_coeff * scale)


def _debiased_fwad_sq(sq: torch.Tensor) -> torch.Tensor:
    """Unbiased batch-shrinkage variance reduction for probe sq before convex EBI cap (L2-4)."""
    n = int(sq.shape[0])
    if n < 2:
        return sq
    alpha = 0.88 * (float(n - 1) / float(n))
    return (1.0 - alpha) * sq + alpha * sq.mean()


class _UnifiedV15PenaltyKernel(_UnifiedPenaltyKernel):
    """Unified-v15 strictly zero-``autograd.grad`` 3-regime Born-Infeld critic penalty kernel.

    Regimes:
    - ``fwad_exact``: batched identity-basis ``fwAD`` (``[B * d, ...]`` with ``eye(d)`` tangents) for ``d <= 16``.
    - ``fwad_debiased``: native multi-probe ``fwAD`` for Conv2d / BN / GN / LN / EmulatedLinear critics (FP32 & BF16).
    - ``secant_debiased``: 1st-order central-secant Hutchinson probes for Attention/SDPA/Softmax or any op raising
      under ``fwAD`` (dtype-aware ``h``: ``1e-3 * scale`` FP32, ``1e-2 * scale`` BF16).
    """

    def __init__(self, *, record: Any = None, **kwargs: Any) -> None:
        rec = UnifiedV15StepRecord() if record is None else record
        super().__init__(record=rec, **kwargs)
        self.arm = "unified_v15"
        self._last_image_grad_mode: str = ""
        self._fwad_unsupported: set[int] = set()

    @property
    def beta(self) -> float:
        return self.ebi_beta

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

    @staticmethod
    def _enter_bn_eval_and_hooks(
        critic_module: Optional[nn.Module],
        *,
        has_batchnorm: bool,
        has_instancenorm: bool,
        attach_bf16_dual_hook: bool,
    ) -> Tuple[List[Tuple[nn.Module, bool]], List[Any]]:
        bn_modes: List[Tuple[nn.Module, bool]] = []
        bn_hooks: List[Any] = []
        if not isinstance(critic_module, nn.Module):
            return bn_modes, bn_hooks
        if has_batchnorm or (has_instancenorm and attach_bf16_dual_hook):
            for mod in critic_module.modules():
                if has_batchnorm and isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    bn_modes.append((mod, mod.training))
                    mod.eval()
                    if attach_bf16_dual_hook:
                        bn_hooks.append(mod.register_forward_hook(_bn_dual_bf16_hook))
                elif (
                    has_instancenorm
                    and attach_bf16_dual_hook
                    and isinstance(mod, torch.nn.modules.instancenorm._InstanceNorm)
                ):
                    bn_hooks.append(mod.register_forward_hook(_bn_dual_bf16_hook))
        return bn_modes, bn_hooks

    @staticmethod
    def _exit_bn_eval_and_hooks(bn_modes: List[Tuple[nn.Module, bool]], bn_hooks: List[Any]) -> None:
        for h in bn_hooks:
            h.remove()
        for mod, flag in bn_modes:
            mod.training = flag

    def _eval_fwad_exact_sq(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
        x_probe: torch.Tensor,
        *,
        can_stack_basis: bool,
        batch_mult: int = 1,
        has_batchnorm: bool,
        has_instancenorm: bool,
        critic_module: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact low-D (``d <= 16``) batched identity-basis ``fwAD`` (``[B * d, ...]`` with ``eye(d)`` tangents)."""
        x_det = x_probe.detach()
        b_p = int(x_det.shape[0])
        dim = max(1, int(x_det[0].numel()))
        tail_ones = (1,) * (x_det.ndim - 1)
        rep_fn = live_rep if live_rep is not None else (lambda z_in, _m: live(z_in))

        _, cuda_bf16_autocast = _is_bf16_eval_context(x_det, critic_module)
        use_bf16_dual_hook = bool(
            (has_batchnorm or has_instancenorm)
            and cuda_bf16_autocast
            and isinstance(critic_module, nn.Module)
        )
        bn_modes, bn_hooks = self._enter_bn_eval_and_hooks(
            critic_module,
            has_batchnorm=has_batchnorm,
            has_instancenorm=has_instancenorm,
            attach_bf16_dual_hook=use_bf16_dual_hook,
        )
        try:
            in_dtype = torch.bfloat16 if use_bf16_dual_hook else x_det.dtype
            x_in = x_det.to(dtype=in_dtype)
            eye_d = torch.eye(dim, device=x_det.device, dtype=in_dtype)
            if dim == 1:
                u1 = torch.ones_like(x_in)
                with fwAD.dual_level():
                    yd = rep_fn(fwAD.make_dual(x_in, u1), batch_mult)
                    y_1, jvp_1 = fwAD.unpack_dual(yd)
                if jvp_1 is None:
                    raise NotImplementedError("dual tangent dropped")
                return _format_logits_2d(y_1, b_p), _format_logits_2d(jvp_1.float(), b_p).square().mean(dim=-1)

            if can_stack_basis and not has_batchnorm:
                x_rep = x_in.repeat((dim,) + tail_ones)
                u_rep = eye_d.repeat_interleave(b_p, dim=0).reshape_as(x_rep)
                with fwAD.dual_level():
                    yd = rep_fn(fwAD.make_dual(x_rep, u_rep), dim * batch_mult)
                    y_rep, jvp_rep = fwAD.unpack_dual(yd)
                if jvp_rep is None:
                    raise NotImplementedError("dual tangent dropped")
                y_2d = _format_logits_2d(y_rep[:b_p], b_p)
                sq = _format_logits_2d(jvp_rep.float(), dim * b_p).square().mean(dim=-1).reshape(dim, b_p).sum(dim=0)
                return y_2d, sq

            y_first: Optional[torch.Tensor] = None
            sq_acc = x_det.new_zeros((b_p,), dtype=torch.float32)
            for axis in range(dim):
                u_axis = eye_d[axis].unsqueeze(0).expand(b_p, dim).reshape_as(x_in).contiguous()
                with fwAD.dual_level():
                    yd = live(fwAD.make_dual(x_in, u_axis))
                    y_a, jvp_a = fwAD.unpack_dual(yd)
                if jvp_a is None:
                    raise NotImplementedError("dual tangent dropped")
                if y_first is None:
                    y_first = _format_logits_2d(y_a, b_p)
                sq_acc = sq_acc + _format_logits_2d(jvp_a.float(), b_p).square().mean(dim=-1)
            assert y_first is not None
            return y_first, sq_acc
        finally:
            self._exit_bn_eval_and_hooks(bn_modes, bn_hooks)

    def _eval_fwad_debiased_sq(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
        x_probe: torch.Tensor,
        phi: float,
        *,
        salt: int,
        dim: int,
        k_probes: int,
        can_stack_probes: bool,
        has_batchnorm: bool,
        has_instancenorm: bool,
        critic_module: Optional[nn.Module],
    ) -> torch.Tensor:
        """Multi-probe orthogonal ``fwAD`` estimator ``(dim / K) * sum_k ||J_D(x) u_k||_2^2`` (0 ``autograd.grad``)."""
        x_det = x_probe.detach()
        b_p = int(x_det.shape[0])
        probes = _orthogonal_unit_probes(x_det, phi, salt=salt, k_probes=k_probes)
        k_eff = len(probes)
        _, cuda_bf16_autocast = _is_bf16_eval_context(x_det, critic_module)
        use_bf16_dual_hook = bool(
            (has_batchnorm or has_instancenorm)
            and cuda_bf16_autocast
            and isinstance(critic_module, nn.Module)
        )
        bn_modes, bn_hooks = self._enter_bn_eval_and_hooks(
            critic_module,
            has_batchnorm=has_batchnorm,
            has_instancenorm=has_instancenorm,
            attach_bf16_dual_hook=use_bf16_dual_hook,
        )
        try:
            in_dtype = torch.bfloat16 if use_bf16_dual_hook else x_det.dtype
            x_in = x_det.to(dtype=in_dtype)
            if can_stack_probes and not has_batchnorm and k_eff > 1 and live_rep is not None:
                tail_ones = (1,) * (x_in.ndim - 1)
                x_rep = x_in.repeat((k_eff,) + tail_ones)
                u_rep = torch.cat([u.to(dtype=in_dtype) for u in probes], dim=0)
                with fwAD.dual_level():
                    yd = live_rep(fwAD.make_dual(x_rep, u_rep), k_eff)
                    _, jvp_rep = fwAD.unpack_dual(yd)
                if jvp_rep is None:
                    raise NotImplementedError("dual tangent dropped")
                j2d = _format_logits_2d(jvp_rep.float(), k_eff * b_p)
                return (float(dim) / float(k_eff)) * j2d.square().mean(dim=-1).reshape(k_eff, b_p).sum(dim=0)

            sq_acc = x_det.new_zeros((b_p,), dtype=torch.float32)
            for u_k in probes:
                u_in = u_k.to(dtype=in_dtype)
                with fwAD.dual_level():
                    yd = live(fwAD.make_dual(x_in, u_in))
                    _, jvp = fwAD.unpack_dual(yd)
                if jvp is None:
                    raise NotImplementedError("dual tangent dropped")
                j2d = _format_logits_2d(jvp.float(), b_p)
                sq_acc = sq_acc + j2d.square().mean(dim=-1)
            return (float(dim) / float(k_eff)) * sq_acc
        finally:
            self._exit_bn_eval_and_hooks(bn_modes, bn_hooks)

    def _eval_secant_debiased_sq(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
        x_probe: torch.Tensor,
        phi: float,
        *,
        salt: int,
        dim: int,
        k_probes: int,
        exact_basis: bool,
        can_stack_probes: bool,
        batch_mult: int = 1,
        has_batchnorm: bool,
        critic_module: Optional[nn.Module],
        quantized_fprop: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """1st-order central-secant Hutchinson probes ``t_k = (D(x + h v_k) - D(x - h v_k)) / (2 h)`` (0 autograd.grad)."""
        x_det = x_probe.detach()
        b_p = int(x_det.shape[0])
        is_bf16, _ = _is_bf16_eval_context(x_det, critic_module)
        h = _secant_step_size(x_det, is_bf16=is_bf16, quantized_fprop=quantized_fprop)
        rep_fn = live_rep if live_rep is not None else (lambda z_in, _m: live(z_in))

        if exact_basis:
            eye_d = torch.eye(dim, device=x_det.device, dtype=torch.float32)
            probes = [eye_d[a].unsqueeze(0).expand(b_p, dim).reshape_as(x_det) for a in range(dim)]
            scale_mult = 1.0
        else:
            probes = _orthogonal_unit_probes(x_det, phi, salt=salt, k_probes=k_probes)
            scale_mult = float(dim) / float(len(probes))

        bn_modes, bn_hooks = self._enter_bn_eval_and_hooks(
            critic_module,
            has_batchnorm=has_batchnorm,
            has_instancenorm=False,
            attach_bf16_dual_hook=False,
        )
        try:
            k_eff = len(probes)
            if can_stack_probes and not has_batchnorm and k_eff > 1:
                tail_ones = (1,) * (x_det.ndim - 1)
                x_rep = x_det.repeat((k_eff,) + tail_ones)
                v_rep = torch.cat([u.float() for u in probes], dim=0)
                xp = (x_rep.float() + h * v_rep).to(dtype=x_det.dtype).detach()
                xm = (x_rep.float() - h * v_rep).to(dtype=x_det.dtype).detach()
                yp = _format_logits_2d(rep_fn(xp, k_eff * batch_mult).float(), k_eff * b_p)
                ym = _format_logits_2d(rep_fn(xm, k_eff * batch_mult).float(), k_eff * b_p)
                step_len = (xp.float() - xm.float()).flatten(1).norm(dim=1).clamp_min(1e-12)
                t_sq = ((yp - ym) / step_len.unsqueeze(-1)).square().mean(dim=-1).reshape(k_eff, b_p).sum(dim=0)
                y_mid = (0.5 * (yp[:b_p] + ym[:b_p])).to(dtype=x_det.dtype)
                return y_mid, scale_mult * t_sq

            y_first: Optional[torch.Tensor] = None
            sq_acc = x_det.new_zeros((b_p,), dtype=torch.float32)
            for v_k in probes:
                xp = (x_det.float() + h * v_k.float()).to(dtype=x_det.dtype).detach()
                xm = (x_det.float() - h * v_k.float()).to(dtype=x_det.dtype).detach()
                yp = _format_logits_2d(live(xp).float(), b_p)
                ym = _format_logits_2d(live(xm).float(), b_p)
                if y_first is None:
                    y_first = (0.5 * (yp + ym)).to(dtype=x_det.dtype)
                step_len = (xp.float() - xm.float()).flatten(1).norm(dim=1).clamp_min(1e-12)
                t_k = (yp - ym) / step_len.unsqueeze(-1)
                sq_acc = sq_acc + t_k.square().mean(dim=-1)
            assert y_first is not None
            return y_first, scale_mult * sq_acc
        finally:
            self._exit_bn_eval_and_hooks(bn_modes, bn_hooks)

    def _compute_eff_beta(self) -> float:
        eff_beta = self.ebi_beta
        if self.settle_beta > 0.0 and bool(getattr(self.record, "t1c_thermal_initialized", False)):
            if self.settle_mode == "thermal":
                s_th = float(getattr(self.record, "_thermal_scale", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - s_th))
            else:
                res = float(getattr(self.record, "ema_residual", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - res / 0.22))
            eff_beta = eff_beta + self.settle_beta * (settle_u * settle_u)
        return eff_beta

    def _finish_vector_penalty(
        self,
        x_real: torch.Tensor,
        d_r_2d: torch.Tensor,
        d_f_2d: torch.Tensor,
        sq_r: torch.Tensor,
        sq_f: torch.Tensor,
        sq_c: Optional[torch.Tensor],
        *,
        coefficient: float,
        eff_beta: float,
        collect_stats: bool,
        grad_mode: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
            mean_logit = 0.5 * (d_r_2d.float().mean() + d_f_2d.float().mean())
            gauge_pen = self.gamma_gauge * mean_logit.square()
        else:
            mean_logit = x_real.new_zeros((), dtype=torch.float32)
            gauge_pen = x_real.new_zeros((), dtype=torch.float32)

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
        self._last_image_grad_mode = grad_mode

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
            "grad_mode": grad_mode,
        }

    def _penalty_lowd_exact(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        is_conditioned: bool = False,
        can_concat_batches: bool = False,
        quantized_fprop: bool = False,
        has_batchnorm: bool = False,
        has_instancenorm: bool = False,
        critic_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Low-D (``d <= 16``) exact identity-basis ``fwad_exact`` with ``secant_debiased`` fallback."""
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
        b_r = int(x_real.shape[0])
        dim = max(1, int(x_real[0].numel()))
        eff_beta = self._compute_eff_beta()
        safe_concat = bool(can_concat_batches and not has_batchnorm)
        cid = id(critic_module) if critic_module is not None else id(live)
        needs_secant = _critic_needs_secant(critic_module) or (cid in self._fwad_unsupported)

        try:
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

            self.record._in_penalty = True
            try:
                grad_mode = "fwad_exact"
                d_r_2d: Optional[torch.Tensor] = None
                d_f_2d: Optional[torch.Tensor] = None
                sq_r: Optional[torch.Tensor] = None
                sq_f: Optional[torch.Tensor] = None
                sq_c: Optional[torch.Tensor] = None
                if not needs_secant:
                    try:
                        if safe_concat and not is_conditioned and x_real.shape == x_fake.shape and x_real.dtype == x_fake.dtype:
                            stack_parts = [x_real, x_fake] + ([x_chord] if x_chord is not None else [])
                            x_fused = torch.cat(stack_parts, dim=0)
                            y_fused, sq_fused = self._eval_fwad_exact_sq(
                                live,
                                live_rep,
                                x_fused,
                                can_stack_basis=True,
                                batch_mult=len(stack_parts),
                                has_batchnorm=has_batchnorm,
                                has_instancenorm=has_instancenorm,
                                critic_module=critic_module,
                            )
                            d_r_2d, d_f_2d = y_fused[:b_r], y_fused[b_r : 2 * b_r]
                            sq_r, sq_f = sq_fused[:b_r], sq_fused[b_r : 2 * b_r]
                            sq_c = sq_fused[2 * b_r :] if x_chord is not None else None
                        else:
                            d_r_2d, sq_r = self._eval_fwad_exact_sq(
                                live,
                                live_rep,
                                x_real,
                                can_stack_basis=safe_concat,
                                has_batchnorm=has_batchnorm,
                                has_instancenorm=has_instancenorm,
                                critic_module=critic_module,
                            )
                            d_f_2d, sq_f = self._eval_fwad_exact_sq(
                                live,
                                live_rep,
                                x_fake,
                                can_stack_basis=safe_concat,
                                has_batchnorm=has_batchnorm,
                                has_instancenorm=has_instancenorm,
                                critic_module=critic_module,
                            )
                            if x_chord is not None:
                                _, sq_c = self._eval_fwad_exact_sq(
                                    live,
                                    live_rep,
                                    x_chord,
                                    can_stack_basis=safe_concat,
                                    has_batchnorm=has_batchnorm,
                                    has_instancenorm=has_instancenorm,
                                    critic_module=critic_module,
                                )
                    except (RuntimeError, NotImplementedError, TypeError):
                        self._fwad_unsupported.add(cid)
                        needs_secant = True
                        d_r_2d = d_f_2d = sq_r = sq_f = sq_c = None

                if needs_secant or d_r_2d is None or d_f_2d is None or sq_r is None or sq_f is None:
                    grad_mode = "secant_debiased"
                    if safe_concat and not is_conditioned and x_real.shape == x_fake.shape and x_real.dtype == x_fake.dtype:
                        stack_parts = [x_real, x_fake] + ([x_chord] if x_chord is not None else [])
                        x_fused = torch.cat(stack_parts, dim=0)
                        y_fused, sq_fused = self._eval_secant_debiased_sq(
                            live,
                            live_rep,
                            x_fused,
                            phi,
                            salt=3,
                            dim=dim,
                            k_probes=dim,
                            exact_basis=True,
                            can_stack_probes=True,
                            batch_mult=len(stack_parts),
                            has_batchnorm=has_batchnorm,
                            critic_module=critic_module,
                            quantized_fprop=quantized_fprop,
                        )
                        d_r_2d, d_f_2d = y_fused[:b_r], y_fused[b_r : 2 * b_r]
                        sq_r, sq_f = sq_fused[:b_r], sq_fused[b_r : 2 * b_r]
                        sq_c = sq_fused[2 * b_r :] if x_chord is not None else None
                    else:
                        d_r_2d, sq_r = self._eval_secant_debiased_sq(
                            live,
                            live_rep,
                            x_real,
                            phi,
                            salt=3,
                            dim=dim,
                            k_probes=dim,
                            exact_basis=True,
                            can_stack_probes=safe_concat,
                            has_batchnorm=has_batchnorm,
                            critic_module=critic_module,
                            quantized_fprop=quantized_fprop,
                        )
                        d_f_2d, sq_f = self._eval_secant_debiased_sq(
                            live,
                            live_rep,
                            x_fake,
                            phi,
                            salt=11,
                            dim=dim,
                            k_probes=dim,
                            exact_basis=True,
                            can_stack_probes=safe_concat,
                            has_batchnorm=has_batchnorm,
                            critic_module=critic_module,
                            quantized_fprop=quantized_fprop,
                        )
                        sq_c = None
                        if x_chord is not None:
                            _, sq_c = self._eval_secant_debiased_sq(
                                live,
                                live_rep,
                                x_chord,
                                phi,
                                salt=19,
                                dim=dim,
                                k_probes=dim,
                                exact_basis=True,
                                can_stack_probes=safe_concat,
                                has_batchnorm=has_batchnorm,
                                critic_module=critic_module,
                                quantized_fprop=quantized_fprop,
                            )
            finally:
                self.record._in_penalty = False

            return self._finish_vector_penalty(
                x_real,
                d_r_2d,
                d_f_2d,
                sq_r,
                sq_f,
                sq_c,
                coefficient=coefficient,
                eff_beta=eff_beta,
                collect_stats=collect_stats,
                grad_mode=grad_mode,
            )
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()

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
        has_batchnorm: bool = False,
        has_instancenorm: bool = False,
        critic_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Batched high-D (``dim > 16``, unconditioned 2D) penalty with BN eval safety and SDPA secant fallback."""
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
        tail_shape = tuple(x_real.shape[1:])
        dim = max(1, int(x_real[0].numel()))
        xr_2d = x_real.detach().reshape(b_n, dim)
        xf_2d = x_fake.detach().reshape(b_n, dim)
        eff_beta = self._compute_eff_beta()
        safe_concat = bool(can_concat_batches and not has_batchnorm)
        cid = id(critic_module) if critic_module is not None else id(live)
        needs_secant = _critic_needs_secant(critic_module) or (cid in self._fwad_unsupported)
        is_bf16, cuda_bf16_autocast = _is_bf16_eval_context(x_real, critic_module)
        use_bf16_dual_hook = bool(
            (has_batchnorm or has_instancenorm)
            and cuda_bf16_autocast
            and isinstance(critic_module, nn.Module)
        )

        try:
            use_chord = self.chord_weight > 0.0
            if use_chord:
                x_chord, phi = self._build_geodesic_chord(xr_2d, xf_2d, is_conditioned=False)
                x_stack = torch.stack((xr_2d, xf_2d, x_chord), dim=0)
                m_cnt = 3
            else:
                phi = float(getattr(self.record, "t1c_chord_phase", 0.0))
                if hasattr(self.record, "t1c_chord_phase"):
                    self.record.t1c_chord_phase = (phi + _GOLDEN_CONJ) % 1.0
                x_stack = torch.stack((xr_2d, xf_2d), dim=0)
                m_cnt = 2

            self.record._in_penalty = True
            bn_modes, bn_hooks = self._enter_bn_eval_and_hooks(
                critic_module,
                has_batchnorm=has_batchnorm,
                has_instancenorm=has_instancenorm,
                attach_bf16_dual_hook=use_bf16_dual_hook and not needs_secant,
            )
            try:
                u_all, knn_idx, weights = _prepare_batched_highd_tangent_plans(x_stack, phi)
                grad_mode = "fwad_debiased"
                if safe_concat:
                    x_cat = x_stack.reshape(m_cnt * b_n, *tail_shape)
                    u_cat = u_all.reshape(m_cnt * b_n, *tail_shape)
                    y_2d: Optional[torch.Tensor] = None
                    tau_sq_cat: Optional[torch.Tensor] = None
                    if not needs_secant:
                        try:
                            in_dtype = torch.bfloat16 if use_bf16_dual_hook else x_cat.dtype
                            with fwAD.dual_level():
                                yd = live(fwAD.make_dual(x_cat.to(dtype=in_dtype), u_cat.to(dtype=in_dtype)))
                                y_cat, jvp_cat = fwAD.unpack_dual(yd)
                            if jvp_cat is None:
                                raise NotImplementedError("dual tangent dropped")
                            y_2d = _format_logits_2d(y_cat, m_cnt * b_n)
                            tau_sq_cat = _format_logits_2d(jvp_cat.float(), m_cnt * b_n).square().mean(dim=-1)
                        except (RuntimeError, NotImplementedError, TypeError):
                            self._fwad_unsupported.add(cid)
                            y_2d = None
                    if y_2d is None or tau_sq_cat is None:
                        grad_mode = "secant_debiased"
                        h = _secant_step_size(x_cat, is_bf16=is_bf16, quantized_fprop=quantized_fprop)
                        xp = (x_cat.float() + h * u_cat.float()).to(dtype=x_cat.dtype).detach()
                        xm = (x_cat.float() - h * u_cat.float()).to(dtype=x_cat.dtype).detach()
                        y_p = _format_logits_2d(live(xp).float(), m_cnt * b_n)
                        y_m = _format_logits_2d(live(xm).float(), m_cnt * b_n)
                        y_2d = (0.5 * (y_p + y_m)).to(dtype=x_cat.dtype)
                        step_sq = (xp.float() - xm.float()).flatten(1).square().sum(dim=1).clamp_min(1e-12)
                        tau_sq_cat = (y_p - y_m).square().mean(dim=-1) / step_sq
                    d_r_2d = y_2d[:b_n]
                    d_f_2d = y_2d[b_n : 2 * b_n]
                    tau_m = tau_sq_cat.view(m_cnt, b_n)
                else:
                    logits_parts: List[torch.Tensor] = []
                    tau_parts: List[torch.Tensor] = []
                    for i in range(m_cnt):
                        x_i = x_stack[i].reshape(b_n, *tail_shape)
                        u_i = u_all[i].reshape(b_n, *tail_shape)
                        y_i_2d: Optional[torch.Tensor] = None
                        t_i_sq: Optional[torch.Tensor] = None
                        if not needs_secant:
                            try:
                                in_dtype = torch.bfloat16 if use_bf16_dual_hook else x_i.dtype
                                with fwAD.dual_level():
                                    yd = live(fwAD.make_dual(x_i.to(dtype=in_dtype), u_i.to(dtype=in_dtype)))
                                    y_i, jvp_i = fwAD.unpack_dual(yd)
                                if jvp_i is None:
                                    raise NotImplementedError("dual tangent dropped")
                                y_i_2d = _format_logits_2d(y_i, b_n)
                                t_i_sq = _format_logits_2d(jvp_i.float(), b_n).square().mean(dim=-1)
                            except (RuntimeError, NotImplementedError, TypeError):
                                self._fwad_unsupported.add(cid)
                                needs_secant = True
                                y_i_2d = None
                        if y_i_2d is None or t_i_sq is None:
                            grad_mode = "secant_debiased"
                            h = _secant_step_size(x_i, is_bf16=is_bf16, quantized_fprop=quantized_fprop)
                            xp = (x_i.float() + h * u_i.float()).to(dtype=x_i.dtype).detach()
                            xm = (x_i.float() - h * u_i.float()).to(dtype=x_i.dtype).detach()
                            y_p = _format_logits_2d(live(xp).float(), b_n)
                            y_m = _format_logits_2d(live(xm).float(), b_n)
                            y_i_2d = (0.5 * (y_p + y_m)).to(dtype=x_i.dtype)
                            step_sq = (xp.float() - xm.float()).flatten(1).square().sum(dim=1).clamp_min(1e-12)
                            t_i_sq = (y_p - y_m).square().mean(dim=-1) / step_sq
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
                self._exit_bn_eval_and_hooks(bn_modes, bn_hooks)
                self.record._in_penalty = False

            return self._finish_vector_penalty(
                x_real,
                d_r_2d,
                d_f_2d,
                sq_r,
                sq_f,
                sq_c,
                coefficient=coefficient,
                eff_beta=eff_beta,
                collect_stats=collect_stats,
                grad_mode=grad_mode,
            )
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()

    def _penalty_highd_general(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        can_concat_batches: bool = False,
        quantized_fprop: bool = False,
        has_batchnorm: bool = False,
        has_instancenorm: bool = False,
        critic_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """General high-D (``dim > 16``, conditioned or uneven batch) non-image penalty (0 ``autograd.grad``)."""
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
        eff_beta = self._compute_eff_beta()
        safe_concat = bool(can_concat_batches and not has_batchnorm)
        cid = id(critic_module) if critic_module is not None else id(live)
        needs_secant = _critic_needs_secant(critic_module) or (cid in self._fwad_unsupported)
        k_probes = 4 if dim <= 1024 else 2

        try:
            phi = float(getattr(self.record, "t1c_chord_phase", 0.0))
            if hasattr(self.record, "t1c_chord_phase"):
                self.record.t1c_chord_phase = (phi + _GOLDEN_CONJ) % 1.0

            self.record._in_penalty = True
            try:
                grad_mode = "fwad_debiased"
                sq_r_raw: Optional[torch.Tensor] = None
                sq_f_raw: Optional[torch.Tensor] = None
                d_r_2d: Optional[torch.Tensor] = None
                d_f_2d: Optional[torch.Tensor] = None
                if not needs_secant:
                    try:
                        bn_modes, bn_hooks = self._enter_bn_eval_and_hooks(
                            critic_module,
                            has_batchnorm=has_batchnorm,
                            has_instancenorm=False,
                            attach_bf16_dual_hook=False,
                        )
                        try:
                            d_r_2d = _format_logits_2d(live(x_real.detach()), b_r)
                            d_f_2d = _format_logits_2d(live(x_fake.detach()), b_f)
                        finally:
                            self._exit_bn_eval_and_hooks(bn_modes, bn_hooks)
                        sq_r_raw = self._eval_fwad_debiased_sq(
                            live,
                            live_rep,
                            x_real,
                            phi,
                            salt=3,
                            dim=dim,
                            k_probes=k_probes,
                            can_stack_probes=safe_concat,
                            has_batchnorm=has_batchnorm,
                            has_instancenorm=has_instancenorm,
                            critic_module=critic_module,
                        )
                        sq_f_raw = self._eval_fwad_debiased_sq(
                            live,
                            live_rep,
                            x_fake,
                            phi,
                            salt=11,
                            dim=dim,
                            k_probes=k_probes,
                            can_stack_probes=safe_concat,
                            has_batchnorm=has_batchnorm,
                            has_instancenorm=has_instancenorm,
                            critic_module=critic_module,
                        )
                    except (RuntimeError, NotImplementedError, TypeError):
                        self._fwad_unsupported.add(cid)
                        needs_secant = True
                        sq_r_raw = sq_f_raw = None

                if needs_secant or sq_r_raw is None or sq_f_raw is None or d_r_2d is None or d_f_2d is None:
                    grad_mode = "secant_debiased"
                    d_r_2d, sq_r_raw = self._eval_secant_debiased_sq(
                        live,
                        live_rep,
                        x_real,
                        phi,
                        salt=3,
                        dim=dim,
                        k_probes=k_probes,
                        exact_basis=False,
                        can_stack_probes=safe_concat,
                        has_batchnorm=has_batchnorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )
                    d_f_2d, sq_f_raw = self._eval_secant_debiased_sq(
                        live,
                        live_rep,
                        x_fake,
                        phi,
                        salt=11,
                        dim=dim,
                        k_probes=k_probes,
                        exact_basis=False,
                        can_stack_probes=safe_concat,
                        has_batchnorm=has_batchnorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )
                sq_r_cap = _debiased_fwad_sq(sq_r_raw)
                sq_f_cap = _debiased_fwad_sq(sq_f_raw)
            finally:
                self.record._in_penalty = False

            return self._finish_vector_penalty(
                x_real,
                d_r_2d,
                d_f_2d,
                sq_r_cap,
                sq_f_cap,
                None,
                coefficient=coefficient,
                eff_beta=eff_beta,
                collect_stats=collect_stats,
                grad_mode=grad_mode,
            )
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()

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
        del cond_tensors
        safe_concat = bool(can_concat_batches and not has_batchnorm)
        has_spatial = _has_spatial_dims(x_real)
        is_image_tensor = has_spatial and self.input_domain != "continuous"
        if not is_image_tensor:
            b_r_ni = int(x_real.shape[0])
            dim_ni = max(1, int(x_real[0].numel())) if b_r_ni > 0 else 0
            if dim_ni <= 16:
                res = self._penalty_lowd_exact(
                    live,
                    live_rep,
                    x_real,
                    x_fake,
                    step,
                    collect_stats,
                    is_conditioned=is_conditioned,
                    can_concat_batches=safe_concat,
                    quantized_fprop=quantized_fprop,
                    has_batchnorm=has_batchnorm,
                    has_instancenorm=has_instancenorm,
                    critic_module=critic_module,
                )
            elif (
                not is_conditioned
                and not has_spatial
                and not _has_spatial_dims(x_fake)
                and x_real.ndim >= 2
                and x_fake.ndim >= 2
                and b_r_ni >= 2
                and int(x_fake.shape[0]) == b_r_ni
                and int(x_fake[0].numel()) == dim_ni
                and x_real.dtype == x_fake.dtype
            ):
                res = self._penalty_highd_batched(
                    live,
                    x_real,
                    x_fake,
                    step,
                    collect_stats,
                    can_concat_batches=safe_concat,
                    quantized_fprop=quantized_fprop,
                    has_batchnorm=has_batchnorm,
                    has_instancenorm=has_instancenorm,
                    critic_module=critic_module,
                )
            else:
                res = self._penalty_highd_general(
                    live,
                    live_rep,
                    x_real,
                    x_fake,
                    step,
                    collect_stats,
                    can_concat_batches=safe_concat,
                    quantized_fprop=quantized_fprop,
                    has_batchnorm=has_batchnorm,
                    has_instancenorm=has_instancenorm,
                    critic_module=critic_module,
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
        eff_beta = self._compute_eff_beta()
        cid = id(critic_module) if critic_module is not None else id(live)
        needs_secant = _critic_needs_secant(critic_module) or (cid in self._fwad_unsupported)
        k_probes = 8 if dim <= 1024 else 2

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
                bn_ep_modes: List[Tuple[nn.Module, bool]] = []
                if has_batchnorm and (hit_r is None or hit_f is None) and isinstance(critic_module, nn.Module):
                    for mod in critic_module.modules():
                        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                            bn_ep_modes.append((mod, mod.training))
                            mod.eval()
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
                finally:
                    for mod, flag in bn_ep_modes:
                        mod.training = flag

                d_r_2d = _format_logits_2d(d_r_raw, b_r)
                d_f_2d = _format_logits_2d(d_f_raw, b_f)

                # Three strictly zero-autograd.grad regimes:
                # 1) fwad_exact: batched identity-basis fwAD for low-D spatial tensors (d <= 16).
                # 2) fwad_debiased: native multi-probe fwAD for Conv2d / BN / GN / LN / EmulatedLinear critics.
                # 3) secant_debiased: 1st-order central-secant Hutchinson probes for Attention/SDPA/Softmax
                #    or any op raising NotImplementedError/RuntimeError under fwAD.
                # Mixed-batch BN guard: NEVER concatenate or interleave real + fake when has_batchnorm is True.
                can_cat_rf = bool(
                    safe_concat
                    and not is_conditioned
                    and b_r == b_f
                    and z_r.shape == z_f.shape
                    and z_r.dtype == z_f.dtype
                )
                sq_r_raw: Optional[torch.Tensor] = None
                sq_f_raw: Optional[torch.Tensor] = None
                grad_mode = "fwad_exact" if dim <= 16 else "fwad_debiased"

                if not needs_secant:
                    try:
                        if dim <= 16:
                            grad_mode = "fwad_exact"
                            if can_cat_rf:
                                x_cat = torch.cat((z_r, z_f), dim=0)
                                _, sq_cat = self._eval_fwad_exact_sq(
                                    live,
                                    live_rep,
                                    x_cat,
                                    can_stack_basis=True,
                                    batch_mult=2,
                                    has_batchnorm=has_batchnorm,
                                    has_instancenorm=has_instancenorm,
                                    critic_module=critic_module,
                                )
                                sq_r_raw, sq_f_raw = sq_cat[:b_r], sq_cat[b_r:]
                            else:
                                _, sq_r_raw = self._eval_fwad_exact_sq(
                                    live,
                                    live_rep,
                                    z_r,
                                    can_stack_basis=safe_concat,
                                    has_batchnorm=has_batchnorm,
                                    has_instancenorm=has_instancenorm,
                                    critic_module=critic_module,
                                )
                                _, sq_f_raw = self._eval_fwad_exact_sq(
                                    live,
                                    live_rep,
                                    z_f,
                                    can_stack_basis=safe_concat,
                                    has_batchnorm=has_batchnorm,
                                    has_instancenorm=has_instancenorm,
                                    critic_module=critic_module,
                                )
                        else:
                            grad_mode = "fwad_debiased"
                            if can_cat_rf:
                                x_cat = torch.cat((z_r, z_f), dim=0)
                                sq_cat = self._eval_fwad_debiased_sq(
                                    live,
                                    live_rep,
                                    x_cat,
                                    phi,
                                    salt=3,
                                    dim=dim,
                                    k_probes=k_probes,
                                    can_stack_probes=True,
                                    has_batchnorm=has_batchnorm,
                                    has_instancenorm=has_instancenorm,
                                    critic_module=critic_module,
                                )
                                sq_r_raw, sq_f_raw = sq_cat[:b_r], sq_cat[b_r:]
                            else:
                                sq_r_raw = self._eval_fwad_debiased_sq(
                                    live,
                                    live_rep,
                                    z_r,
                                    phi,
                                    salt=3,
                                    dim=dim,
                                    k_probes=k_probes,
                                    can_stack_probes=safe_concat,
                                    has_batchnorm=has_batchnorm,
                                    has_instancenorm=has_instancenorm,
                                    critic_module=critic_module,
                                )
                                sq_f_raw = self._eval_fwad_debiased_sq(
                                    live,
                                    live_rep,
                                    z_f,
                                    phi,
                                    salt=11,
                                    dim=dim,
                                    k_probes=k_probes,
                                    can_stack_probes=safe_concat,
                                    has_batchnorm=has_batchnorm,
                                    has_instancenorm=has_instancenorm,
                                    critic_module=critic_module,
                                )
                    except (RuntimeError, NotImplementedError, TypeError):
                        self._fwad_unsupported.add(cid)
                        needs_secant = True
                        sq_r_raw = sq_f_raw = None

                if needs_secant or sq_r_raw is None or sq_f_raw is None:
                    grad_mode = "secant_debiased"
                    exact_b = dim <= 16
                    if can_cat_rf:
                        x_cat = torch.cat((z_r, z_f), dim=0)
                        _, sq_cat = self._eval_secant_debiased_sq(
                            live,
                            live_rep,
                            x_cat,
                            phi,
                            salt=3,
                            dim=dim,
                            k_probes=dim if exact_b else k_probes,
                            exact_basis=exact_b,
                            can_stack_probes=True,
                            batch_mult=2,
                            has_batchnorm=has_batchnorm,
                            critic_module=critic_module,
                            quantized_fprop=quantized_fprop,
                        )
                        sq_r_raw, sq_f_raw = sq_cat[:b_r], sq_cat[b_r:]
                    else:
                        _, sq_r_raw = self._eval_secant_debiased_sq(
                            live,
                            live_rep,
                            z_r,
                            phi,
                            salt=3,
                            dim=dim,
                            k_probes=dim if exact_b else k_probes,
                            exact_basis=exact_b,
                            can_stack_probes=safe_concat,
                            has_batchnorm=has_batchnorm,
                            critic_module=critic_module,
                            quantized_fprop=quantized_fprop,
                        )
                        _, sq_f_raw = self._eval_secant_debiased_sq(
                            live,
                            live_rep,
                            z_f,
                            phi,
                            salt=11,
                            dim=dim,
                            k_probes=dim if exact_b else k_probes,
                            exact_basis=exact_b,
                            can_stack_probes=safe_concat,
                            has_batchnorm=has_batchnorm,
                            critic_module=critic_module,
                            quantized_fprop=quantized_fprop,
                        )

                self._last_image_grad_mode = grad_mode
                if dim <= 16:
                    sq_r_cap = sq_r_raw
                    sq_f_cap = sq_f_raw
                else:
                    sq_r_cap = _debiased_fwad_sq(sq_r_raw)
                    sq_f_cap = _debiased_fwad_sq(sq_f_raw)

                # Born-Infeld cap + D-09 dimension-independent normal-axis dissipation floor (never vanishes on r256):
                norm_r = (sq_r_cap + 1e-12).sqrt()
                norm_f = (sq_f_cap + 1e-12).sqrt()
                use_dcgan_parity_floor = (dim > 1024) and (not _critic_has_non_dcgan_layers(critic_module))
                img_floor = (
                    (self.lambda_floor if use_dcgan_parity_floor else min(self.lambda_floor, 1.0e-4))
                    if self.lambda_floor > 0.0
                    else 0.0
                )
                total = x_real.new_zeros((), dtype=torch.float32)
                if self.real_weight > 0.0:
                    total = total + self.real_weight * F.softplus(norm_r - self.kappa, beta=eff_beta).square().mean()
                    if img_floor > 0.0:
                        total = total + self.real_weight * img_floor * sq_r_raw.mean()
                if self.fake_weight > 0.0:
                    total = total + self.fake_weight * F.softplus(norm_f - self.kappa, beta=eff_beta).square().mean()
                    if img_floor > 0.0:
                        total = total + self.fake_weight * img_floor * sq_f_raw.mean()
                if self.gamma_gauge > 0.0:
                    eff_gauge = self.gamma_gauge if use_dcgan_parity_floor else min(self.gamma_gauge, 0.005)
                    mean_logit = 0.5 * (d_r_2d.float().mean() + d_f_2d.float().mean())
                    gauge_pen = eff_gauge * mean_logit.square()
                else:
                    mean_logit = x_real.new_zeros((), dtype=torch.float32)
                    gauge_pen = x_real.new_zeros((), dtype=torch.float32)
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
                "grad_mode": self._last_image_grad_mode,
            }
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()


class UnifiedV15CriticPenalty(_K3PCriticPenalty):
    """Recipe-facing Unified-v15 1st-order Forward-Mode AD & Central-Secant critic penalty (0 autograd.grad)."""

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
            isinstance(optimizer, (UnifiedV15CriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (
                isinstance(optimizer, (NormalizedOptimizer, UnifiedV11EquilibratedOptimizer, UnifiedV15Optimizer))
                and hasattr(optimizer, "record")
            )
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_unified_v15_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _UnifiedV15PenaltyKernel(record=optimizer.record, **options)
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

    @property
    def coefficient(self) -> float:
        return float(self.regularizer.coeff)

    def __call__(
        self,
        critic: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *condition: Any,
        **condition_kwargs: Any,
    ) -> torch.Tensor:
        from ..discriminators import BatchDistanceDiscriminator
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
        if critic is self.critic:
            has_batchnorm = self._has_batchnorm
            has_instancenorm = self._has_instancenorm
            has_batch_distance = self._has_batch_distance
            quantized_fprop = self._has_quantized_fprop
        else:
            has_batchnorm = isinstance(critic, nn.Module) and any(
                isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in critic.modules()
            )
            has_instancenorm = isinstance(critic, nn.Module) and any(
                isinstance(m, torch.nn.modules.instancenorm._InstanceNorm) for m in critic.modules()
            )
            has_batch_distance = isinstance(critic, nn.Module) and any(
                isinstance(m, BatchDistanceDiscriminator) for m in critic.modules()
            )
            quantized_fprop = bool(
                self._has_quantized_fprop
                or (
                    isinstance(critic, nn.Module)
                    and any(
                        getattr(m, "format", None) in ("nvfp4", "mxfp8")
                        or "Emulated" in type(m).__name__
                        or "Quantized" in type(m).__name__
                        for m in critic.modules()
                    )
                )
            )
        allow_live_reuse = (output is _first_output) and not has_batch_state
        can_concat_batches = (
            not has_batchnorm
            and not has_batch_distance
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
            quantized_fprop=quantized_fprop,
            has_batchnorm=has_batchnorm,
            has_instancenorm=has_instancenorm,
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
            unified_v15_ell=float(getattr(record, "t1c_ell_ema", 0.0)),
            unified_v15_ell_initialized=bool(getattr(record, "t1c_ell_initialized", False)),
            unified_v15_chord_phase=float(getattr(record, "t1c_chord_phase", 0.0)),
            unified_v15_thermal_scale=float(getattr(record, "_thermal_scale", 1.0)),
            unified_v15_step_scale=float(getattr(record, "_step_scale", 1.0)),
            unified_v15_num_charts=0 if getattr(record, "chart_count", None) is None else int(record.chart_count.shape[0]),
            unified_v15_cond_strata=0 if bank is None else len(bank.strata),
        )
        return out


__all__ = [
    "UnifiedV15CriticPenalty",
    "_UnifiedV15PenaltyKernel",
]
