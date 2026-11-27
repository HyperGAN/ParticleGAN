"""Axiom L1 Critic Potential Geometry (Law 2) and Normal Dissipation (Law 4.5).

Zero-``torch.autograd.grad`` scale-equivariant critic regularization using
forward-mode AD (``fwAD``) as primary with capability-triggered precision-aware
central secant fallback, continuous element-budget probing, exact-basis vs.
linear zero-Jensen penalties, and live-endpoint zero-mean logit gauge.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

from .math_ops import PHI1, smoothstep01, weyl_rademacher
from .state import (
    AxiomStepRecord,
    _attach_axiom_critic_hook,
    _attach_axiom_generator_hook,
    bind_axiom_record,
    check_scalar,
    eval_probe_forward,
    extract_axiom_inputs_parts as _extract_axiom_inputs_parts,
    first_tensor_output,
    format_logits_2d,
    record_axiom_capture as _record_axiom_capture,
)

_ELEMENT_BUDGET: int = 32_768
_SECANT_DELTA_REL: float = 0.02


# ---------------------------------------------------------------------------
# Law 2.1 & 2.2: Intrinsic Effective Dimension, Scale & Element-Budget Probes
# ---------------------------------------------------------------------------


@torch.no_grad()
def effective_dimension(x: torch.Tensor) -> torch.Tensor:
    """Unbiased U-statistic effective dimension ``d_eff in [1, d]`` from centered Gram matrix."""
    batch_size = int(x.shape[0])
    dim_x = max(1, int(x[0].numel()))
    if batch_size < 4 or dim_x <= 1:
        return x.new_full((), float(dim_x), dtype=torch.float64)
    xc = x.detach().reshape(batch_size, dim_x).float()
    xc = xc - xc.mean(dim=0, keepdim=True)
    sx = xc.abs().amax()
    xu = xc / torch.where(sx > 0.0, sx, sx.new_ones(()))
    gram = (xu @ xu.T).double()
    diag = gram.diagonal()
    s1, s11, s2 = diag.sum(), diag.square().sum(), gram.square().sum()
    a = float((batch_size - 1) * (batch_size - 2))
    c = float(batch_size * (batch_size - 1))
    num = a * s1 * s1 - c * s11 + 2.0 * s2
    den = a * s2 - c * s11 + s1 * s1
    valid = (sx > 0.0) & (num > 0.0) & (den > 0.0)
    d_est = (num / den.clamp_min(1e-30)).clamp(1.0, float(dim_x))
    return torch.where(valid, d_est, s1.new_full((), float(dim_x)))


def probe_schedule(batch_size: int, dim: int) -> Tuple[int, int]:
    """Continuous element-budget ``(K(d), b_reg(B, d))`` with zero modality/dimension cliffs."""
    d_safe = max(1, int(dim))
    b_safe = max(1, int(batch_size))
    k_probes = min(d_safe, max(1, 16 // d_safe))
    raw_b = _ELEMENT_BUDGET // (d_safe * k_probes)
    b_lo = min(b_safe, max(2, b_safe // 4))
    b_reg = min(b_safe, max(b_lo, raw_b))
    return k_probes, b_reg


@torch.no_grad()
def _coord_scale(
    x_real: torch.Tensor, x_fake: torch.Tensor, record: AxiomStepRecord
) -> torch.Tensor:
    """Degree-1 homogeneous coordinate scale ``s_scale`` via uniform ``d_eff`` and shock-tracked EMA."""
    dev = x_real.device
    d_eff = effective_dimension(x_real)
    xr = x_real.detach().reshape(int(x_real.shape[0]), -1).double()
    xf = x_fake.detach().reshape(int(x_fake.shape[0]), -1).double()
    xm = torch.maximum(xr.abs().amax(), xf.abs().amax())
    sx = torch.where(xm > 0.0, xm, xm.new_ones(()))
    xru, xfu = xr / sx, xf / sx
    mr, mf = xru.mean(dim=0, keepdim=True), xfu.mean(dim=0, keepdim=True)
    var_r = (xru - mr).square().sum(dim=-1).mean()
    var_f = (xfu - mf).square().sum(dim=-1).mean()
    mean_sq = (mr - mf).square().sum()
    spread_sq = torch.maximum(torch.maximum(var_r, 0.25 * var_f), mean_sq)
    tot_sq = torch.where(spread_sq > 0.0, spread_sq, xru.square().sum(dim=-1).mean())
    s_cur = ((tot_sq.sqrt() * sx) / d_eff.sqrt()).clamp(1e-30, 1e30)
    ds = record.dev_ema("ema_data_scale", dev)
    mi = record.dev_ema("margin_initialized", dev)
    s_ema = (ds / d_eff.sqrt()).clamp(1e-30, 1e30)
    sc = torch.where(mi > 0.5, s_ema, torch.where(xm > 0.0, s_cur, s_ema.new_ones(())))
    return sc.float()


def _cyclic_slice(val: Any, batch_size: int, start: int, count: int) -> Any:
    if (
        not isinstance(val, torch.Tensor)
        or val.ndim < 1
        or val.shape[0] != batch_size
        or count >= batch_size
    ):
        return val
    end = start + count
    if end <= batch_size:
        return val[start:end]
    return torch.cat((val[start:], val[: end - batch_size]), dim=0)


def _slice_call_args(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_size: int,
    start: int,
    count: int,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    sliced_a = tuple(_cyclic_slice(item, batch_size, start, count) for item in args)
    sliced_kw = {
        key: _cyclic_slice(val, batch_size, start, count) for key, val in kwargs.items()
    }
    return sliced_a, sliced_kw


# ---------------------------------------------------------------------------
# Deterministic Zero-RNG Probe Direction Generators (fwAD & Sparse Secant)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _build_fwad_directions(
    x: torch.Tensor, k_probes: int, phase: float, salt: int, dtype: torch.dtype
) -> List[torch.Tensor]:
    """Build ``K`` deterministic O(1)-entry ``fwAD`` tangents (exact basis for ``K == d``)."""
    batch_size = int(x.shape[0])
    dim = max(1, int(x[0].numel()))
    dev = x.device
    if k_probes == dim:
        eye = torch.eye(dim, device=dev, dtype=dtype)
        return [
            eye[k].unsqueeze(0).expand(batch_size, dim).reshape_as(x)
            for k in range(dim)
        ]
    v0 = weyl_rademacher(
        (batch_size, dim), phase=phase, salt=salt, device=dev, dtype=torch.float32
    )
    if k_probes == 1:
        return [v0.to(dtype=dtype).reshape_as(x)]
    u0 = v0 / math.sqrt(float(dim))
    v1 = weyl_rademacher(
        (batch_size, dim), phase=phase, salt=salt + 17, device=dev, dtype=torch.float32
    )
    proj = (v1 * u0).sum(dim=1, keepdim=True)
    v1_orth = v1 - proj * u0
    nrm = v1_orth.norm(dim=1, keepdim=True)
    u1 = torch.where(
        nrm > 1e-6, v1_orth / nrm.clamp_min(1e-6), v1 / math.sqrt(float(dim))
    )
    sqrt_d = math.sqrt(float(dim))
    return [
        (u0 * sqrt_d).to(dtype=dtype).reshape_as(x),
        (u1 * sqrt_d).to(dtype=dtype).reshape_as(x),
    ]


def _secant_sparsity(dim: int, delta_rel: float, dtype: torch.dtype) -> int:
    """Precision-aware coordinate sparsity preventing ULP underflow in ``x +/- delta * u``."""
    eps_dt = (
        float(torch.finfo(dtype).eps)
        if dtype.is_floating_point
        else float(torch.finfo(torch.float32).eps)
    )
    ratio = float(delta_rel) / max(4.0 * eps_dt, 1e-30)
    return min(dim, max(1, int(math.floor(ratio * ratio))))


def _coprime_stride(dim: int) -> int:
    stride = max(1, int(round(dim * PHI1))) % dim
    if stride == 0:
        stride = 1
    while math.gcd(stride, dim) != 1:
        stride = (stride + 1) % dim or 1
    return stride


@torch.no_grad()
def _sparse_unit_rademacher(
    batch_size: int,
    dim: int,
    sparsity: int,
    phase: float,
    salt: int,
    device: torch.device,
) -> torch.Tensor:
    """Unit-norm ``s``-sparse Rademacher probe ``u`` with ``||u||_2 = 1`` and ``E[u u^T] = I/d``."""
    if sparsity >= dim:
        raw = weyl_rademacher(
            (batch_size, dim),
            phase=phase,
            salt=salt,
            device=device,
            dtype=torch.float32,
        )
        return raw / math.sqrt(float(dim))
    stride = _coprime_stride(dim)
    base = int(math.floor((phase % 1.0) * dim)) + salt * 104_729
    row_off = (
        torch.arange(batch_size, device=device, dtype=torch.int64) * 7_919 + base
    ) % dim
    cols = (
        row_off[:, None]
        + torch.arange(sparsity, device=device, dtype=torch.int64)[None, :] * stride
    ) % dim
    signs = weyl_rademacher(
        (batch_size, sparsity),
        phase=phase,
        salt=salt + 31,
        device=device,
        dtype=torch.float32,
    )
    out = torch.zeros((batch_size, dim), device=device, dtype=torch.float32)
    out.scatter_(1, cols, signs / math.sqrt(float(sparsity)))
    return out


@torch.no_grad()
def _build_secant_directions(
    x: torch.Tensor, k_probes: int, phase: float, salt: int
) -> List[torch.Tensor]:
    """Build ``K`` unit-norm precision-aware central-secant probe directions."""
    batch_size = int(x.shape[0])
    dim = max(1, int(x[0].numel()))
    dev = x.device
    if k_probes == dim:
        eye = torch.eye(dim, device=dev, dtype=torch.float32)
        return [
            eye[k].unsqueeze(0).expand(batch_size, dim).reshape_as(x)
            for k in range(dim)
        ]
    sparsity = _secant_sparsity(dim, _SECANT_DELTA_REL, x.dtype)
    u0 = _sparse_unit_rademacher(batch_size, dim, sparsity, phase, salt, dev)
    if k_probes == 1:
        return [u0.reshape_as(x)]
    u1 = _sparse_unit_rademacher(batch_size, dim, sparsity, phase, salt + 19, dev)
    if sparsity >= dim:
        v1_orth = u1 - (u1 * u0).sum(dim=1, keepdim=True) * u0
        nrm = v1_orth.norm(dim=1, keepdim=True)
        u1 = torch.where(nrm > 1e-6, v1_orth / nrm.clamp_min(1e-6), u1)
    return [u0.reshape_as(x), u1.reshape_as(x)]


# ---------------------------------------------------------------------------
# BN / BF16 Dual Evaluation Context
# ---------------------------------------------------------------------------


def _bn_dual_bf16_hook(
    _module: nn.Module, _inputs: Tuple[Any, ...], output: Any
) -> Any:
    if not isinstance(output, torch.Tensor):
        return output
    primal, tangent = fwAD.unpack_dual(output)
    if tangent is None:
        return output
    if primal.dtype != torch.bfloat16 or tangent.dtype != torch.bfloat16:
        return fwAD.make_dual(primal.to(torch.bfloat16), tangent.to(torch.bfloat16))
    return output


def _is_bf16_critic(x: torch.Tensor, mod: Optional[nn.Module]) -> bool:
    if not x.is_cuda or not isinstance(mod, nn.Module):
        return False
    autocast_bf16 = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    return bool(getattr(mod, "bf16", False) or autocast_bf16)


@contextmanager
def _penalty_eval_context(
    mod: Optional[nn.Module], record: AxiomStepRecord, *, bf16_dual: bool = False
) -> Iterator[None]:
    """Preserve BN running stats/modes and register BF16 dual hooks during penalty forwards."""
    bn_states: List[Tuple[nn.Module, bool]] = []
    hooks: List[Any] = []
    if isinstance(mod, nn.Module):
        for sub in mod.modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                bn_states.append((sub, sub.training))
                sub.eval()
                if bf16_dual:
                    hooks.append(sub.register_forward_hook(_bn_dual_bf16_hook))
            elif bf16_dual and isinstance(sub, nn.modules.instancenorm._InstanceNorm):
                hooks.append(sub.register_forward_hook(_bn_dual_bf16_hook))
    prev_pen = record._in_penalty
    record._in_penalty = True
    try:
        yield
    finally:
        record._in_penalty = prev_pen
        for handle in hooks:
            handle.remove()
        for sub, was_training in bn_states:
            sub.training = was_training


# ---------------------------------------------------------------------------
# Law 2 + 4.5 Penalty Kernel & Public Wrapper
# ---------------------------------------------------------------------------


class AxiomPenaltyKernel:
    """Zero-``autograd.grad`` scale-equivariant ``fwAD`` + secant Law 2/4.5 critic penalty."""

    def __init__(
        self,
        *,
        record: Optional[AxiomStepRecord] = None,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lambda_floor: float = 0.025,
        beta_ema: float = 1.5,
        gamma_gauge: float = 0.03,
        max_lambda: float = 4.0,
    ) -> None:
        self.record = AxiomStepRecord() if record is None else record
        self.coeff = check_scalar(coeff, "coeff", lo=0.0)
        self.kappa = check_scalar(kappa, "kappa", lo=0.0, strict_lo=True)
        self.lambda_floor = check_scalar(
            lambda_floor, "lambda_floor", lo=0.0, strict_lo=True
        )
        self.beta_ema = check_scalar(beta_ema, "beta_ema", lo=0.0)
        self.gamma_gauge = check_scalar(gamma_gauge, "gamma_gauge", lo=0.0)
        self.max_lambda = check_scalar(max_lambda, "max_lambda", lo=self.lambda_floor)
        self._c_max = math.log(max(1.0, self.max_lambda / self.lambda_floor))
        self._fwad_unsupported: set[int] = set()

    def _fwad_batch(
        self,
        live: Callable[..., torch.Tensor],
        x: torch.Tensor,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        k_probes: int,
        phi: float,
        salt: int,
        bf16: bool,
        sc_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(x.shape[0])
        dim = max(1, int(x[0].numel()))
        in_dt = torch.bfloat16 if bf16 else x.dtype
        xd = x.detach().to(dtype=in_dt)
        dirs = _build_fwad_directions(x, k_probes, phi, salt, in_dt)
        scale_w = 1.0 if k_probes == dim else (1.0 / float(k_probes))
        y_primal: Optional[torch.Tensor] = None
        sq_acc: Optional[torch.Tensor] = None
        with fwAD.dual_level():
            for vec in dirs:
                y_raw, j_raw = fwAD.unpack_dual(
                    live(fwAD.make_dual(xd, vec), *args, **kwargs)
                )
                if j_raw is None:
                    raise NotImplementedError("forward-mode AD dropped dual tangent")
                y2 = format_logits_2d(y_raw, batch_size)
                j2 = format_logits_2d(j_raw, batch_size).float() * sc_t
                term = j2.square().mean(dim=-1)
                sq_acc = term if sq_acc is None else (sq_acc + term)
                if y_primal is None:
                    y_primal = y2
        assert y_primal is not None and sq_acc is not None
        return y_primal, sq_acc * scale_w

    def _secant_batch(
        self,
        live: Callable[..., torch.Tensor],
        x: torch.Tensor,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        k_probes: int,
        phi: float,
        salt: int,
        sc_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(x.shape[0])
        dim = max(1, int(x[0].numel()))
        xd = x.detach()
        delta_t = (sc_t * _SECANT_DELTA_REL).clamp_min(1e-30)
        delta_x = delta_t.to(dtype=xd.dtype)
        inv_2d = (0.5 * sc_t) / delta_t
        dirs = _build_secant_directions(xd, k_probes, phi, salt)
        scale_w = 1.0 if k_probes == dim else (float(dim) / float(k_probes))
        y_mid: Optional[torch.Tensor] = None
        sq_acc: Optional[torch.Tensor] = None
        for u_vec in dirs:
            step = (u_vec.to(dtype=xd.dtype)) * delta_x
            yp = format_logits_2d(live(xd + step, *args, **kwargs), batch_size)
            ym = format_logits_2d(live(xd - step, *args, **kwargs), batch_size)
            tau_scaled = (yp.float() - ym.float()) * inv_2d
            term = tau_scaled.square().mean(dim=-1)
            sq_acc = term if sq_acc is None else (sq_acc + term)
            if y_mid is None:
                y_mid = 0.5 * (yp + ym)
        assert y_mid is not None and sq_acc is not None
        return y_mid, sq_acc * scale_w

    def _eval_pair(
        self,
        batch_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        b_reg: int,
        phi: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        br, bf = int(x_real.shape[0]), int(x_fake.shape[0])
        sr = int(phi * br) % br
        sf = int(((phi + 0.5) % 1.0) * bf) % bf
        full_batch = b_reg >= min(br, bf)
        if not args and not kwargs and br == bf and x_real.shape == x_fake.shape:
            br_s = br if full_batch else max(1, b_reg // 2)
            bf_s = bf if full_batch else max(1, b_reg - br_s)
            try:
                x_cat = torch.cat(
                    (
                        _cyclic_slice(x_real.detach(), br, sr, br_s),
                        _cyclic_slice(x_fake.detach(), bf, sf, bf_s),
                    ),
                    dim=0,
                )
                y_cat, sq_cat = batch_fn(x_cat, (), {}, 1)
                return y_cat[:br_s], y_cat[br_s:], sq_cat[:br_s], sq_cat[br_s:]
            except RuntimeError:
                pass
        br_s = br if full_batch else max(1, min(br, b_reg // 2))
        bf_s = bf if full_batch else max(1, min(bf, b_reg - br_s))
        try:
            ar, kwr = _slice_call_args(args, kwargs, br, sr, br_s)
            af, kwf = _slice_call_args(args, kwargs, bf, sf, bf_s)
            yr, sqr = batch_fn(_cyclic_slice(x_real.detach(), br, sr, br_s), ar, kwr, 1)
            yf, sqf = batch_fn(_cyclic_slice(x_fake.detach(), bf, sf, bf_s), af, kwf, 2)
            return yr, yf, sqr, sqf
        except RuntimeError:
            if br_s == br and bf_s == bf:
                raise
            yr, sqr = batch_fn(x_real.detach(), args, kwargs, 1)
            yf, sqf = batch_fn(x_fake.detach(), args, kwargs, 2)
            return yr, yf, sqr, sqf

    def _assemble_penalty(
        self,
        dr: torch.Tensor,
        df: torch.Tensor,
        sqr: torch.Tensor,
        sqf: torch.Tensor,
        *,
        exact_basis: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        s_hat = 0.5 * (sqr.mean() + sqf.mean())
        ema_sq = self.record.dev_ema("ema_grad_sq", s_hat.device)
        excess = torch.relu(ema_sq - self.kappa * self.kappa)
        lam = (
            self.lambda_floor
            * torch.exp((self.beta_ema * excess).clamp_max(self._c_max))
        ).to(s_hat.dtype)
        self.record.observe_grad_sq_tensor(s_hat.detach())
        slope_term = lam * s_hat
        if exact_basis:
            cap_r = torch.relu((sqr + 1e-12).sqrt() - self.kappa).square().mean()
            cap_f = torch.relu((sqf + 1e-12).sqrt() - self.kappa).square().mean()
            slope_term = slope_term + 0.5 * (cap_r + cap_f)
        pen_slope = 0.5 * self.coeff * slope_term
        mean_logit = (
            0.5 * (dr.float().mean() + df.float().mean())
            if self.gamma_gauge > 0.0
            else s_hat.new_zeros(())
        )
        pen_gauge = (
            0.5 * self.coeff * self.gamma_gauge * mean_logit.square()
            if self.gamma_gauge > 0.0
            else s_hat.new_zeros(())
        )
        s_th = float(self.record.thermal_scale)
        if s_th < 1.0:
            if pen_slope.requires_grad:
                pen_slope = pen_slope * s_th + pen_slope.detach() * (1.0 - s_th)
            if pen_gauge.requires_grad:
                gu = smoothstep01((mean_logit.detach().abs() - 0.5) / 1.0)
                s_gauge = s_th + gu * (math.sqrt(max(0.0, s_th)) - s_th)
                pen_gauge = pen_gauge * s_gauge + pen_gauge.detach() * (1.0 - s_gauge)
        return pen_slope + pen_gauge, s_hat, lam, s_th

    def penalty(
        self,
        live: Callable[..., torch.Tensor],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *args: Any,
        collect_stats: bool = False,
        critic_module: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evaluate zero-``autograd.grad`` Law 2/4.5 critic penalty and optional diagnostics."""
        if self.coeff == 0.0:
            self.record.clear_transient_captures()
            stats = (
                {"applied": False, "pen": 0.0, "center": self.kappa}
                if collect_stats
                else {}
            )
            return x_real.new_zeros(()), stats
        self.record.calls += 1
        phi = float(self.record.chord_phase)
        self.record.chord_phase = (phi + PHI1) % 1.0
        mod = (
            critic_module
            if isinstance(critic_module, nn.Module)
            else (live if isinstance(live, nn.Module) else None)
        )
        cid = id(mod) if mod is not None else id(live)
        dim_x = max(1, int(x_real[0].numel()))
        k_probes, b_reg = probe_schedule(
            min(int(x_real.shape[0]), int(x_fake.shape[0])), dim_x
        )
        sc_t = _coord_scale(x_real, x_fake, self.record)
        dr_live = self.record.find_live_endpoint(x_real, args, kwargs)
        df_live = self.record.find_live_endpoint(x_fake, args, kwargs)
        reused = dr_live is not None and df_live is not None
        use_secant = cid in self._fwad_unsupported
        try:
            if not use_secant:
                bf16 = _is_bf16_critic(x_real, mod)
                try:
                    with _penalty_eval_context(mod, self.record, bf16_dual=bf16):
                        dr_s, df_s, sqr, sqf = self._eval_pair(
                            lambda x, a, kw, s: self._fwad_batch(
                                live, x, a, kw, k_probes, phi, s, bf16, sc_t
                            ),
                            x_real,
                            x_fake,
                            args,
                            kwargs,
                            b_reg,
                            phi,
                        )
                except (RuntimeError, NotImplementedError, TypeError):
                    self._fwad_unsupported.add(cid)
                    use_secant = True
            if use_secant:
                with _penalty_eval_context(mod, self.record, bf16_dual=False):
                    dr_s, df_s, sqr, sqf = self._eval_pair(
                        lambda x, a, kw, s: self._secant_batch(
                            live, x, a, kw, k_probes, phi, s, sc_t
                        ),
                        x_real,
                        x_fake,
                        args,
                        kwargs,
                        b_reg,
                        phi,
                    )
            dr_use, df_use = (dr_live, df_live) if reused else (dr_s, df_s)
            pen, s_hat, lam, s_th = self._assemble_penalty(
                dr_use, df_use, sqr, sqf, exact_basis=(k_probes == dim_x)
            )
            if not collect_stats:
                return pen, {}
            pv, sv, lv = torch.stack(
                (pen.detach().float(), s_hat.detach().float(), lam.detach().float())
            ).tolist()
            return pen, {
                "applied": True,
                "pen": float(pv),
                "s_hat": float(sv),
                "mean_grad_norm": math.sqrt(max(0.0, float(sv))),
                "lambda_eff": float(lv),
                "center": self.kappa,
                "regime": "secant" if use_secant else "fwad",
                "k_probes": k_probes,
                "b_reg": b_reg,
                "s_thermal": s_th,
                "live_endpoints_reused": bool(reused),
            }
        finally:
            self.record.clear_transient_captures()


class AxiomCriticPenalty:
    """Public callable wrapper matching GANTrainer, WordFixture, DDGAN, and BehaviorComponents."""

    ema_critic = None

    def __init__(
        self,
        optimizer_or_recipe: Any,
        optimizer: Any = None,
        *,
        coeff: Optional[float] = None,
        kappa: Optional[float] = None,
        output: Any = None,
        collect_stats: bool = False,
        **options: Any,
    ) -> None:
        self.recipe, self.optimizer = (
            (None, optimizer_or_recipe)
            if optimizer is None
            else (optimizer_or_recipe, optimizer)
        )
        if not hasattr(self.optimizer, "param_groups"):
            raise TypeError("AxiomCriticPenalty requires a critic optimizer")
        rec = getattr(self.optimizer, "record", None) or AxiomStepRecord()
        self.optimizer.record = rec
        self.optimizer._axiom_record = rec
        self.critic = getattr(self.optimizer, "critic", None)
        _attach_axiom_critic_hook(self.critic, rec)
        k_opts: Dict[str, Any] = {
            "coeff": float(getattr(self.recipe, "reg_coeff", 1.0))
            if coeff is None
            else float(coeff),
            "kappa": float(getattr(self.recipe, "reg_kappa", 1.0))
            if kappa is None
            else float(kappa),
        }
        for key in (
            "coeff",
            "kappa",
            "lambda_floor",
            "beta_ema",
            "gamma_gauge",
            "max_lambda",
        ):
            if options.get(key) is not None:
                k_opts[key] = options[key]
        self.regularizer = AxiomPenaltyKernel(record=rec, **k_opts)
        self.initial_coeff = self.regularizer.coeff
        self.output = first_tensor_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}

    def __call__(
        self,
        critic: Callable[..., torch.Tensor],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        mod = critic if isinstance(critic, nn.Module) else self.critic
        pen, self.last_stats = self.regularizer.penalty(
            lambda x, *a, **kw: self.output(critic(x, *a, **kw)),
            x_real,
            x_fake,
            *args,
            collect_stats=self.collect_stats,
            critic_module=mod,
            **kwargs,
        )
        return pen

    def diagnostics(self) -> Dict[str, Any]:
        """Return scalar critic regularizer diagnostics."""
        rec = self.regularizer.record
        return {
            "axiom_calls": int(rec.calls),
            "axiom_ema_grad_sq": float(rec.ema_grad_sq),
            "axiom_chord_phase": float(rec.chord_phase),
            "axiom_thermal_scale": float(rec.thermal_scale),
            "axiom_step_scale": float(rec.step_scale),
        }


__all__ = [
    "AxiomCriticPenalty",
    "AxiomPenaltyKernel",
    "AxiomStepRecord",
    "_attach_axiom_critic_hook",
    "_attach_axiom_generator_hook",
    "_extract_axiom_inputs_parts",
    "_record_axiom_capture",
    "bind_axiom_record",
    "effective_dimension",
    "eval_probe_forward",
    "probe_schedule",
]
