"""Axiom L1 Representational Geometry: Anisotropic LTC Prototype Atlas & GPU Hamilton Ledger."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import torch

from .math_ops import (
    PHI1,
    hamilton_quotas,
    power_principal_axis,
    safe_cdist,
    smoothstep01,
)

SCHEMA_VERSION: int = 2
MAX_FULL_COV_DIM: int = 64
MAX_LEDGER_PARTICLES: int = 2048
_VAR_FLOOR: float = 1e-24
_BETA_N: float = 0.04
_PROPOSAL_ROUNDS: int = 8

_LEDGER_KEYS = frozenset(
    {
        "schema",
        "initialized",
        "num_particles",
        "dim",
        "row_comp",
        "row_target",
        "row_dist",
    }
)
_ATLAS_TENSOR_KEYS = (
    "proto_c",
    "proto_cov",
    "proto_w",
    "w_fake_ema",
    "proto_n",
    "psi",
    "v_push",
)
_ATLAS_KEYS = frozenset(
    {
        "schema",
        "max_prototypes",
        "beta_n",
        "initialized",
        "dim",
        "rho_atom_ema",
        "g_conn_ema",
        *_ATLAS_TENSOR_KEYS,
        "ledger",
    }
)


@dataclass(frozen=True)
class CoalescedAtlas:
    """Directional-LTC coalesced atlas with per-prototype and unique-component views."""

    proto_comp_id: torch.Tensor
    proto_c_bar: torch.Tensor
    proto_cov_bar: torch.Tensor
    proto_var_bar: torch.Tensor
    w_mat: torch.Tensor
    multicell: torch.Tensor
    isolated: torch.Tensor
    comp_c: torch.Tensor
    comp_cov: torch.Tensor
    comp_cov_tgt: torch.Tensor
    comp_var: torch.Tensor
    comp_w: torch.Tensor
    comp_n_eff: torch.Tensor
    comp_isolated: torch.Tensor
    all_iso: torch.Tensor


@dataclass(frozen=True)
class LedgerAssignment:
    """On-device output of a unified GPU Hamilton commitment ledger step."""

    batch_comp: torch.Tensor
    batch_target: torch.Tensor
    nn_comp_dist: torch.Tensor
    quotas: torch.Tensor
    occupancy: torch.Tensor
    tail_residual: torch.Tensor
    row_comp: torch.Tensor


# ---------------------------------------------------------------------------
# Unified Covariance Geometry (D <= 64 Full SPD vs D > 64 Diagonal)
# ---------------------------------------------------------------------------


def cov_trace_mean(cov: torch.Tensor) -> torch.Tensor:
    """Mean per-coordinate variance ``tr(Sigma) / D`` for ``[K, D, D]`` or ``[K, D]``."""
    diag = cov.diagonal(dim1=-2, dim2=-1) if cov.ndim == 3 else cov
    return diag.mean(dim=-1).clamp_min(_VAR_FLOOR)


def cov_from_scalar(var: torch.Tensor, dim: int) -> torch.Tensor:
    """Build isotropic covariance ``v I_D`` (``[K, D, D]`` if ``D <= 64`` else ``[K, D]``)."""
    vf = var.float().clamp_min(_VAR_FLOOR)
    if dim <= MAX_FULL_COV_DIM:
        eye = torch.eye(dim, device=vf.device, dtype=torch.float32)
        return vf.view(-1, 1, 1) * eye.unsqueeze(0)
    return vf.unsqueeze(-1).expand(int(vf.shape[0]), dim).clone()


def cov_directional_var(cov: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Directional variance ``u^T Sigma u`` for full ``[K, D, D]`` or diagonal ``[K, D]``."""
    if cov.ndim == 3:
        sub = "kmd,kde,kme->km" if dirs.ndim == 3 else "kd,kde,ke->k"
        val = torch.einsum(sub, dirs, cov, dirs)
    else:
        val = (dirs.square() * (cov.unsqueeze(1) if dirs.ndim == 3 else cov)).sum(
            dim=-1
        )
    return val.clamp_min(_VAR_FLOOR)


def cov_add_outer(
    cov: torch.Tensor, delta: torch.Tensor, scale: float | torch.Tensor = 1.0
) -> torch.Tensor:
    """Compute ``cov + scale * outer(delta)`` transparently for full SPD or diagonal storage."""
    if cov.ndim == 3:
        term = delta.unsqueeze(-1) * delta.unsqueeze(-2)
        fac = (
            scale.view(*scale.shape, 1, 1)
            if isinstance(scale, torch.Tensor)
            else float(scale)
        )
        out = cov + term * fac
        return 0.5 * (out + out.transpose(-1, -2))
    fac = scale.unsqueeze(-1) if isinstance(scale, torch.Tensor) else float(scale)
    return (cov + delta.square() * fac).clamp_min(_VAR_FLOOR)


def _scale_sanitize_cov(
    cov: torch.Tensor, factor: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if factor is not None:
        cov = cov * factor.view(-1, 1, 1) if cov.ndim == 3 else cov * factor.view(-1, 1)
    if cov.ndim == 3:
        sym = 0.5 * (cov + cov.transpose(-1, -2))
        floor_add = (_VAR_FLOOR - sym.diagonal(dim1=-2, dim2=-1)).clamp_min(0.0)
        return sym + torch.diag_embed(floor_add)
    return cov.clamp_min(_VAR_FLOOR)


def _scatter_cov(
    diff: torch.Tensor, assign: torch.Tensor, k: int, is_full: bool
) -> torch.Tensor:
    dim = int(diff.shape[1])
    if is_full:
        return diff.new_zeros((k, dim, dim)).index_add_(
            0, assign, diff.unsqueeze(2) * diff.unsqueeze(1)
        )
    return diff.new_zeros((k, dim)).index_add_(0, assign, diff.square())


def cov_principal_split(
    cov_k: torch.Tensor, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact 100%-LTC, strictly-PSD principal-axis split returning ``(lam1, delta, cov_child)``."""
    c64 = cov_k.detach().double()
    if c64.ndim == 2:
        sym = 0.5 * (c64 + c64.transpose(-1, -2))
        lam1, u1 = power_principal_axis(sym, iters=6, seed=seed)
        su = sym @ u1
        ray = (u1 * su).sum()
        delta = torch.where(
            ray > _VAR_FLOOR,
            0.40 * su * torch.rsqrt(ray.clamp_min(_VAR_FLOOR)),
            torch.zeros_like(su),
        )
        child = 0.5 * (sym - delta.unsqueeze(1) * delta.unsqueeze(0))
        child = child + child.transpose(-1, -2)
        child = child + torch.diag_embed(
            (_VAR_FLOOR - child.diagonal(dim1=-2, dim2=-1)).clamp_min(0.0)
        )
        return (
            lam1.clamp_min(0.0).to(cov_k.dtype),
            delta.to(cov_k.dtype),
            child.to(cov_k.dtype),
        )

    diag = c64.clamp_min(0.0)
    lam1, idx = diag.max(dim=-1)
    active = lam1 > _VAR_FLOOR
    delta = torch.zeros_like(diag)
    delta[idx] = torch.where(
        active, 0.40 * torch.sqrt(lam1.clamp_min(_VAR_FLOOR)), lam1.new_zeros(())
    )
    child = diag.clone()
    child[idx] = torch.where(active, 0.84 * lam1, diag[idx])
    return (
        lam1.to(cov_k.dtype),
        delta.to(cov_k.dtype),
        child.clamp_min(_VAR_FLOOR).to(cov_k.dtype),
    )


def cov_shrink_mp(cov: torch.Tensor, n_eff: torch.Tensor) -> torch.Tensor:
    """Marchenko-Pastur SPD shrinkage ``(1 - lam) Sigma + lam * v_bar * I_D``."""
    dim = int(cov.shape[-1])
    lam = smoothstep01(
        (float(dim) / n_eff.float().clamp_min(1.0) - 0.25) / 0.75
    ).clamp_min(1e-4)
    v_bar = cov_trace_mean(cov)
    if cov.ndim == 3:
        eye = torch.eye(dim, device=cov.device, dtype=torch.float32).unsqueeze(0)
        return (1.0 - lam.view(-1, 1, 1)) * cov + (lam * v_bar).view(-1, 1, 1) * eye
    return ((1.0 - lam.unsqueeze(-1)) * cov + (lam * v_bar).unsqueeze(-1)).clamp_min(
        _VAR_FLOOR
    )


# ---------------------------------------------------------------------------
# Continuous Annular-Gap Ratio rho_atom & Connectivity Gate g_conn
# ---------------------------------------------------------------------------


@torch.no_grad()
def batch_annular_gap_ratio(y_real: torch.Tensor) -> torch.Tensor:
    """Scale-invariant warmup annular-gap ratio from sorted batch pairwise distances."""
    yf = y_real.detach().reshape(int(y_real.shape[0]), -1).float()
    b = int(yf.shape[0])
    if b < 4:
        return yf.new_ones(())
    if b > 256:
        yf, b = yf[(torch.arange(256, device=yf.device) * b) // 256], 256
    iu, ju = torch.triu_indices(b, b, offset=1, device=yf.device)
    ds = safe_cdist(yf)[iu, ju].sort().values
    d_max = ds[-1]
    ds_safe = ds.clamp_min(1e-7 * d_max.clamp_min(_VAR_FLOOR))
    t_min = 1
    t_max = max(t_min + 1, int(ds.numel()) // 2)
    gaps = torch.log(ds_safe[t_min + 1 : t_max + 1]) - torch.log(ds_safe[t_min:t_max])
    max_gap, rel_idx = gaps.max(dim=0)
    gap_idx = rel_idx + t_min
    ratio_edge = torch.exp(-max_gap.clamp_min(0.0))
    ratio_med = ds[gap_idx // 2] / ds_safe[gap_idx + 1].clamp_min(_VAR_FLOOR)
    sharp = smoothstep01((max_gap - 0.50) / 0.70)
    rho = sharp * ratio_med + (1.0 - sharp) * ratio_edge
    return torch.where(d_max <= _VAR_FLOOR, yf.new_zeros(()), rho.clamp(0.0, 1.0))


@torch.no_grad()
def manifold_connectivity_stats(
    y_real: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimax-path connectivity statistics ``(g_graph, g_macro, q90_u, y_scale)`` on-device."""
    yf = y_real.detach().reshape(int(y_real.shape[0]), -1).float()
    b = int(yf.shape[0])
    amax = yf.abs().amax() if yf.numel() > 0 else yf.new_ones(())
    y_sc = torch.where(amax < 1e-2, amax.clamp_min(1e-12), amax.clamp_min(1.0))
    if b < 4:
        z0 = yf.new_zeros(())
        return z0, z0, yf.new_full((), 0.1), y_sc
    yu = (yf / y_sc)[:: max(1, b // 256)] if b > 512 else (yf / y_sc)
    dyy = safe_cdist(yu)
    off = ~torch.eye(int(yu.shape[0]), dtype=torch.bool, device=yu.device)
    q90 = torch.quantile(
        dyy.masked_fill(~off, float("inf")).amin(dim=-1), 0.90
    ).clamp_min(1e-8)
    span = torch.quantile(dyy[off], 0.90) / q90
    mm = dyy.clone()
    for _ in range(8):
        mm = torch.minimum(
            mm, torch.maximum(mm.unsqueeze(2), mm.unsqueeze(0)).amin(dim=1)
        )
    g_graph = smoothstep01((10.0 - mm.amax() / q90) / 3.0) * smoothstep01(
        (span - 1.15) / 0.25
    )
    return (
        g_graph,
        g_graph * smoothstep01((span - 3.80) / 1.00),
        q90.clamp_min(1e-4),
        y_sc,
    )


@torch.no_grad()
def compute_rho_atom(
    comp_c: torch.Tensor,
    comp_cov: torch.Tensor,
    comp_n_eff: torch.Tensor,
    y_real: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Continuous annular-gap ratio ``median_C(sigma_intra_C / Delta_inter_C)`` + warmup gap."""
    k_c, dim = int(comp_c.shape[0]), int(comp_c.shape[1])
    rho_batch = (
        batch_annular_gap_ratio(y_real) if y_real is not None else comp_c.new_ones(())
    )
    if k_c < 2:
        w_a = (
            smoothstep01((comp_n_eff.median() - 2.0) / 6.0)
            if k_c == 1
            else comp_c.new_zeros(())
        )
        return torch.maximum(rho_batch, w_a).clamp(0.0, 1.0)
    d_inter, nn_idx = safe_cdist(comp_c, diag_inf=True).min(dim=1)
    u_dir = (comp_c - comp_c[nn_idx]) / d_inter.clamp_min(1e-12).unsqueeze(1)
    sig_intra = torch.sqrt(float(dim) * cov_directional_var(comp_cov, u_dir))
    ratio = torch.where(
        d_inter > 1e-12, sig_intra / d_inter.clamp_min(1e-12), comp_c.new_ones(k_c)
    )
    rho_comp = ratio.median().clamp(0.0, 1.0)
    if y_real is None:
        return rho_comp
    w_a = smoothstep01((comp_n_eff.median() - 2.0) / 6.0)
    return ((1.0 - w_a) * rho_batch + w_a * rho_comp).clamp(0.0, 1.0)


@torch.no_grad()
def compute_connectivity_gate(
    y_real: torch.Tensor, *, rho_atom: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Continuous connectivity gate ``g_conn in [0, 1]`` (0 discrete, 1 connected manifold)."""
    g_graph, _, _, _ = manifold_connectivity_stats(y_real)
    if rho_atom is None:
        return g_graph.clamp(0.0, 1.0)
    return (g_graph * smoothstep01((rho_atom - 0.15) / 0.20)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Directional LTC Coalescence (Law 1.1 & 1.3)
# ---------------------------------------------------------------------------


def _transitive_closure(adj: torch.Tensor, k: int) -> torch.Tensor:
    out = adj.clone()
    out.fill_diagonal_(1.0)
    n_sq = min(8, max(4, (k - 1).bit_length()))
    for _ in range(n_sq):
        out = ((out @ out) > 0).float()
    return out


def _ltc_cluster_stats(
    c_f: torch.Tensor,
    cov_c0: torch.Tensor,
    w_c: torch.Tensor,
    adj: torch.Tensor,
    eye_k: torch.Tensor,
    cand: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = int(c_f.shape[0])
    cw = torch.where(cand[:, None] & cand[None, :], adj, eye_k) * w_c.unsqueeze(0)
    cw = cw / cw.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cc = cw @ c_f
    m2_j = cov_add_outer(cov_c0, c_f - cc)
    cov_c = _scale_sanitize_cov((cw @ m2_j.reshape(k, -1)).reshape(m2_j.shape))
    return cw, cc, cov_c


@torch.no_grad()
def coalesce_directional_ltc(
    proto_c: torch.Tensor,
    proto_cov: torch.Tensor,
    proto_w: torch.Tensor,
    proto_n: Optional[torch.Tensor] = None,
) -> CoalescedAtlas:
    """Directional LTC sibling coalescence & external isolation preserving exact ``Sigma_C``."""
    c_f = proto_c.detach().float()
    k, dim, dev = int(c_f.shape[0]), int(c_f.shape[1]), c_f.device
    w_c = proto_w.detach().float().clamp_min(1e-12)
    w_c = w_c / w_c.sum().clamp_min(1e-12)
    n_vec = (
        torch.ones(k, device=dev, dtype=torch.float32)
        if proto_n is None
        else proto_n.detach().float().clamp_min(0.0)
    )
    cov_c0 = _scale_sanitize_cov(proto_cov.detach().float())
    eye_k = torch.eye(k, device=dev, dtype=torch.float32)
    if k < 2:
        v0 = cov_trace_mean(cov_c0)
        zb = torch.zeros(k, device=dev, dtype=torch.bool)
        return CoalescedAtlas(
            torch.arange(k, device=dev, dtype=torch.int64),
            c_f,
            cov_c0,
            v0,
            eye_k,
            zb,
            zb,
            c_f,
            cov_c0,
            cov_shrink_mp(cov_c0, n_vec),
            v0,
            w_c,
            n_vec,
            zb,
            torch.zeros((), device=dev, dtype=torch.bool),
        )

    d_cc = safe_cdist(c_f)
    sqrt_d = math.sqrt(float(dim))
    u_ij = (c_f.unsqueeze(1) - c_f.unsqueeze(0)) / d_cc.clamp_min(1e-12).unsqueeze(-1)
    cov_dir = cov_shrink_mp(cov_c0, n_vec)
    sig_ij = cov_directional_var(cov_dir, u_ij).sqrt()
    sig_tr = cov_trace_mean(cov_c0).sqrt()
    s_max = torch.maximum(sig_tr[:, None], sig_tr[None, :])
    s_min = torch.minimum(sig_tr[:, None], sig_tr[None, :]).clamp_min(1e-12)
    width_ok = s_max <= (2.2 * s_min)
    adj = _transitive_closure(
        ((d_cc <= (2.6 * sqrt_d * torch.maximum(sig_ij, sig_ij.T))) & width_ok).float(),
        k,
    )

    cs = adj.sum(dim=1)
    cand = (cs >= 1.0) & (cs < float(k))
    _, _, cov_c1 = _ltc_cluster_stats(c_f, cov_c0, w_c, adj, eye_k, cand)
    sig_c1_ij = cov_directional_var(cov_c1, u_ij).sqrt()
    multi = (cs >= 1.5) & cand
    absorb = (d_cc <= (2.6 * sqrt_d * sig_c1_ij.T)) & multi[None, :] & width_ok
    adj = _transitive_closure(((adj > 0.5) | absorb | absorb.T).float(), k)

    cs = adj.sum(dim=1)
    cand = (cs >= 1.0) & (cs < float(k))
    cw, cc, cov_c = _ltc_cluster_stats(c_f, cov_c0, w_c, adj, eye_k, cand)

    ratio_c = d_cc / (4.2 * sqrt_d * cov_directional_var(cov_c, u_ij).sqrt()).clamp_min(
        1e-12
    )
    int_ok = (adj @ (~((ratio_c * adj).amax(dim=1) <= 1.0)).float()) < 0.5
    ext_ok = (
        adj
        @ (~(ratio_c.masked_fill(adj > 0.5, float("inf")).amin(dim=1) >= 1.0)).float()
    ) < 0.5
    iso_proto = cand & int_ok & ext_ok
    coal_ok = iso_proto & ((w_c * iso_proto.float()).sum() >= 0.70)

    c_bar = torch.where(coal_ok.unsqueeze(1), cc, c_f)
    cmask = coal_ok.view(k, 1, 1) if cov_c0.ndim == 3 else coal_ok.unsqueeze(1)
    cov_bar = torch.where(cmask, cov_c, cov_c0)
    w_mat = torch.where(coal_ok.unsqueeze(1), cw, eye_k)
    eff_adj = torch.where(coal_ok[:, None] & coal_ok[None, :], adj, eye_k)

    rep = (eff_adj > 0.5).float().argmax(dim=1)
    is_rep = rep == torch.arange(k, device=dev)
    proto_comp_id = (torch.cumsum(is_rep.long(), dim=0) - 1)[rep]
    rep_idx = torch.nonzero(is_rep, as_tuple=False).squeeze(-1)
    comp_cov = cov_bar[rep_idx]
    comp_w = eff_adj[rep_idx] @ w_c
    comp_w = comp_w / comp_w.sum().clamp_min(1e-12)
    comp_n_eff = eff_adj[rep_idx] @ n_vec
    comp_iso = iso_proto[rep_idx]
    all_iso = (
        comp_iso.all()
        if int(rep_idx.numel()) >= 2
        else torch.zeros((), device=dev, dtype=torch.bool)
    )
    return CoalescedAtlas(
        proto_comp_id,
        c_bar,
        cov_bar,
        cov_trace_mean(cov_bar),
        w_mat,
        (cs > 1.5) & coal_ok,
        iso_proto,
        c_bar[rep_idx],
        comp_cov,
        cov_shrink_mp(comp_cov, comp_n_eff),
        cov_trace_mean(comp_cov),
        comp_w,
        comp_n_eff,
        comp_iso,
        all_iso,
    )


@torch.no_grad()
def coalesce_pythagorean_ltv(
    proto_c: torch.Tensor, proto_var: torch.Tensor, proto_w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isotropic LTC coalescence compatibility wrapper delegating to directional LTC."""
    coal = coalesce_directional_ltc(
        proto_c, cov_from_scalar(proto_var, int(proto_c.shape[1])), proto_w
    )
    return coal.proto_c_bar, coal.proto_var_bar, coal.w_mat, coal.multicell


# ---------------------------------------------------------------------------
# Strict Schema-2 Validation Helpers
# ---------------------------------------------------------------------------


def _check_float_scalar(
    val: Any,
    name: str,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    strict: bool = False,
) -> float:
    if type(val) not in (int, float) or not math.isfinite(float(val)):
        raise ValueError(f"invalid scalar {name!r}: {val!r}")
    out = float(val)
    if lo is not None and (out <= lo if strict else out < lo):
        raise ValueError(f"{name}={out} below {lo}")
    if hi is not None and (out >= hi if strict else out > hi):
        raise ValueError(f"{name}={out} above {hi}")
    return out


def _check_float_tensor(
    t: Any, name: str, *, shape: Optional[Tuple[int, ...]] = None, nonneg: bool = False
) -> torch.Tensor:
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        raise ValueError(f"invalid float tensor {name!r}")
    if shape is not None and tuple(t.shape) != shape:
        raise ValueError(f"{name!r} shape {tuple(t.shape)} != {shape}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"non-finite values in {name!r}")
    if nonneg and bool((t < 0.0).any()):
        raise ValueError(f"negative values in {name!r}")
    return t


def _validate_ledger_header(state: Any) -> Tuple[bool, int, int]:
    if not isinstance(state, dict) or set(state.keys()) != _LEDGER_KEYS:
        raise ValueError("invalid HamiltonCommitmentLedger checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != SCHEMA_VERSION:
        raise ValueError(f"ledger requires schema={SCHEMA_VERSION}")
    init, p_cnt, dim = state["initialized"], state["num_particles"], state["dim"]
    if type(init) is not bool or type(p_cnt) is not int or type(dim) is not int:
        raise ValueError("invalid ledger header field types")
    return init, p_cnt, dim


def validate_ledger_state(state: Any) -> Dict[str, Any]:
    """Fail-closed schema-2 validator for :class:`HamiltonCommitmentLedger` checkpoints."""
    init, p_cnt, dim = _validate_ledger_header(state)
    if not init:
        if p_cnt != 0 or dim != 0:
            raise ValueError("uninitialized ledger must have zero dims")
        if any(state[k] is not None for k in ("row_comp", "row_target", "row_dist")):
            raise ValueError("uninitialized ledger must have None tensors")
        return state
    if not (1 <= p_cnt <= MAX_LEDGER_PARTICLES) or dim < 1:
        raise ValueError("ledger dims out of valid range")
    rc = state["row_comp"]
    if not isinstance(rc, torch.Tensor) or rc.dtype != torch.int64:
        raise ValueError("ledger row_comp must be int64 tensor")
    if tuple(rc.shape) != (p_cnt,) or bool((rc < -1).any()):
        raise ValueError("invalid ledger row_comp shape or entries")
    _check_float_tensor(state["row_target"], "row_target", shape=(p_cnt, dim))
    _check_float_tensor(state["row_dist"], "row_dist", shape=(p_cnt,), nonneg=True)
    return state


def _validate_atlas_header(state: Any) -> Tuple[int, bool, int]:
    if not isinstance(state, dict) or set(state.keys()) != _ATLAS_KEYS:
        raise ValueError("invalid PrototypeAtlas checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != SCHEMA_VERSION:
        raise ValueError(f"PrototypeAtlas requires schema={SCHEMA_VERSION}")
    max_p, init, dim = state["max_prototypes"], state["initialized"], state["dim"]
    if type(max_p) is not int or max_p < 2:
        raise ValueError("invalid PrototypeAtlas max_prototypes")
    if type(init) is not bool or type(dim) is not int:
        raise ValueError("invalid PrototypeAtlas initialized/dim types")
    _check_float_scalar(state["beta_n"], "beta_n", lo=0.0, hi=1.0, strict=True)
    _check_float_scalar(state["rho_atom_ema"], "rho_atom_ema", lo=0.0)
    _check_float_scalar(state["g_conn_ema"], "g_conn_ema", lo=0.0, hi=1.0)
    return max_p, init, dim


def _validate_atlas_cov(cov: Any, k: int, dim: int) -> None:
    expected = (k, dim, dim) if dim <= MAX_FULL_COV_DIM else (k, dim)
    t = _check_float_tensor(
        cov, "proto_cov", shape=expected, nonneg=(dim > MAX_FULL_COV_DIM)
    )
    if dim <= MAX_FULL_COV_DIM:
        asym = float((t - t.transpose(-1, -2)).abs().amax().item())
        scale = float(t.abs().amax().clamp_min(1.0).item())
        if asym > 1e-4 * scale or bool((t.diagonal(dim1=-2, dim2=-1) < 0.0).any()):
            raise ValueError("proto_cov must be symmetric with non-negative diagonal")


def _validate_atlas_tensors(state: Dict[str, Any], k: int, dim: int) -> None:
    _validate_atlas_cov(state["proto_cov"], k, dim)
    for name in ("proto_w", "w_fake_ema"):
        wt = _check_float_tensor(state[name], name, shape=(k,), nonneg=True)
        if abs(float(wt.double().sum().item()) - 1.0) > 1e-4:
            raise ValueError(f"PrototypeAtlas.{name} must sum to 1.0")
    for name in ("proto_n", "v_push"):
        _check_float_tensor(state[name], name, shape=(k,), nonneg=True)
    _check_float_tensor(state["psi"], "psi", shape=(k,))


def validate_atlas_state(state: Any) -> Dict[str, Any]:
    """Fail-closed schema-2 validator for :class:`PrototypeAtlas` checkpoints."""
    max_p, init, dim = _validate_atlas_header(state)
    ledger = state["ledger"]
    if ledger is not None:
        validate_ledger_state(ledger)
    if not init:
        if dim != 0 or any(state[k] is not None for k in _ATLAS_TENSOR_KEYS):
            raise ValueError(
                "uninitialized PrototypeAtlas must have dim=0 and None tensors"
            )
        return state
    pc = _check_float_tensor(state["proto_c"], "proto_c")
    if dim < 1 or pc.ndim != 2 or int(pc.shape[1]) != dim:
        raise ValueError("invalid PrototypeAtlas proto_c shape")
    k = int(pc.shape[0])
    if not (1 <= k <= max_p):
        raise ValueError("PrototypeAtlas prototype count exceeds max_prototypes")
    _validate_atlas_tensors(state, k, dim)
    if ledger is not None and ledger["initialized"] and int(ledger["dim"]) != dim:
        raise ValueError("PrototypeAtlas ledger dim mismatch")
    return state


# ---------------------------------------------------------------------------
# Unified GPU-Vectorized Hamilton Commitment Ledger (Schema = 2)
# ---------------------------------------------------------------------------


def _group_rank_ascending(
    groups: torch.Tensor, keys: torch.Tensor, num_groups: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(keys, stable=True)
    order = order[torch.argsort(groups[order], stable=True)]
    g_sorted = groups[order]
    counts = torch.bincount(g_sorted, minlength=num_groups)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = (
        torch.arange(int(groups.numel()), device=groups.device, dtype=torch.int64)
        - starts[g_sorted]
    )
    return order, rank


class HamiltonCommitmentLedger:
    """Zero-CPU-bounce, zero-sync GPU Hamilton quota commitment ledger (``P <= 2048``)."""

    SCHEMA = SCHEMA_VERSION

    def __init__(self) -> None:
        self.initialized: bool = False
        self.num_particles: int = 0
        self.dim: int = 0
        self.row_comp: Optional[torch.Tensor] = None
        self.row_target: Optional[torch.Tensor] = None
        self.row_dist: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.initialized, self.num_particles, self.dim = False, 0, 0
        self.row_comp = self.row_target = self.row_dist = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "initialized": bool(self.initialized),
            "num_particles": int(self.num_particles),
            "dim": int(self.dim),
            "row_comp": None
            if self.row_comp is None
            else self.row_comp.detach().cpu().clone(),
            "row_target": None
            if self.row_target is None
            else self.row_target.detach().cpu().clone(),
            "row_dist": None
            if self.row_dist is None
            else self.row_dist.detach().cpu().clone(),
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        validate_only: bool = False,
    ) -> None:
        val = validate_ledger_state(state)
        if validate_only:
            return
        self.initialized = bool(val["initialized"])
        self.num_particles, self.dim = int(val["num_particles"]), int(val["dim"])
        if not self.initialized:
            self.row_comp = self.row_target = self.row_dist = None
            return
        dst = device if device is not None else val["row_target"].device
        self.row_comp = (
            val["row_comp"].detach().to(device=dst, dtype=torch.int64).clone()
        )
        self.row_target = (
            val["row_target"].detach().to(device=dst, dtype=torch.float32).clone()
        )
        self.row_dist = (
            val["row_dist"].detach().to(device=dst, dtype=torch.float32).clone()
        )

    def _ensure_storage(
        self, p_cnt: int, dim: int, dev: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.initialized
            and self.num_particles == p_cnt
            and self.dim == dim
            and self.row_comp is not None
            and self.row_target is not None
            and self.row_dist is not None
        ):
            if self.row_comp.device != dev:
                self.row_comp = self.row_comp.to(dev)
                self.row_target = self.row_target.to(dev)
                self.row_dist = self.row_dist.to(dev)
            return self.row_comp, self.row_target, self.row_dist
        self.initialized, self.num_particles, self.dim = True, p_cnt, dim
        self.row_comp = torch.full((p_cnt,), -1, dtype=torch.int64, device=dev)
        self.row_target = torch.zeros((p_cnt, dim), dtype=torch.float32, device=dev)
        self.row_dist = torch.full((p_cnt,), 1e30, dtype=torch.float32, device=dev)
        return self.row_comp, self.row_target, self.row_dist

    def _reconcile_and_evict(
        self,
        rc: torch.Tensor,
        rt: torch.Tensor,
        rd: torch.Tensor,
        comp_c: torch.Tensor,
        d_pc: torch.Tensor,
        in_batch: torch.Tensor,
        quotas: torch.Tensor,
        weyl_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_c = int(comp_c.shape[0])
        active = rc >= 0
        new_c = safe_cdist(rt, comp_c).argmin(dim=1)
        matched = comp_c[new_c]
        rc = torch.where(active, new_c, rc)
        rd = torch.where(active, torch.maximum(rd, (rt - matched).norm(dim=1)), rd)
        rt = torch.where(active.unsqueeze(1), matched, rt)
        cur_d = d_pc.gather(1, rc.clamp_min(0).unsqueeze(1)).squeeze(1)
        rd = torch.where(in_batch & (rc >= 0), cur_d, rd)

        comp_key = torch.where(rc >= 0, rc, rc.new_full((), k_c))
        order, rank = _group_rank_ascending(
            comp_key, rd.clamp_max(1e6) + 1e-6 * weyl_p, k_c + 1
        )
        c_ord = comp_key[order]
        evict = torch.zeros_like(active)
        evict.scatter_(
            0, order, (c_ord < k_c) & (rank >= quotas[c_ord.clamp_max(k_c - 1)])
        )
        return (
            torch.where(evict, rc.new_full((), -1), rc),
            rt,
            torch.where(evict, rd.new_full((), 1e30), rd),
        )

    def _assign_deficit(
        self,
        rc: torch.Tensor,
        rt: torch.Tensor,
        rd: torch.Tensor,
        d_pc: torch.Tensor,
        comp_c: torch.Tensor,
        in_batch: torch.Tensor,
        quotas: torch.Tensor,
        weyl_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_c, p_cnt, dev = int(comp_c.shape[0]), int(rc.numel()), rc.device
        occ = torch.bincount(
            torch.where(rc >= 0, rc, rc.new_full((), k_c)), minlength=k_c + 1
        )[:k_c]
        deficit = (quotas - occ).clamp_min(0)
        unassigned = in_batch & (rc < 0)
        inf_t = d_pc.new_tensor(float("inf"))

        for _ in range(_PROPOSAL_ROUNDS):
            pref_d, pref_c = d_pc.masked_fill((deficit <= 0).unsqueeze(0), inf_t).min(
                dim=1
            )
            g_prop = torch.where(
                unassigned & torch.isfinite(pref_d), pref_c, rc.new_full((), k_c)
            )
            order, rank = _group_rank_ascending(
                g_prop, pref_d.clamp_max(1e6) + 1e-6 * weyl_p, k_c + 1
            )
            g_ord = g_prop[order]
            accepted = torch.zeros_like(unassigned)
            accepted.scatter_(
                0, order, (g_ord < k_c) & (rank < deficit[g_ord.clamp_max(k_c - 1)])
            )
            rc = torch.where(accepted, pref_c, rc)
            rt = torch.where(accepted.unsqueeze(1), comp_c[pref_c], rt)
            rd = torch.where(accepted, pref_d, rd)
            unassigned = unassigned & (~accepted)
            taken = torch.bincount(
                torch.where(accepted, pref_c, rc.new_full((), k_c)), minlength=k_c + 1
            )[:k_c]
            deficit = (deficit - taken).clamp_min(0)

        # Fixed-shape GPU searchsorted completion guarantees exact quotas on clustered/tied starts.
        u_min = d_pc.masked_fill((deficit <= 0).unsqueeze(0), inf_t).min(dim=1).values
        comp_ord = torch.argsort(
            d_pc.masked_fill(~unassigned.unsqueeze(1), inf_t)
            .amin(dim=0)
            .clamp_max(1e6),
            stable=True,
        )
        slot_ends = torch.cumsum(deficit[comp_ord], dim=0)
        ranks = torch.arange(p_cnt, device=dev, dtype=torch.int64)
        slot_comp = comp_ord[
            torch.searchsorted(slot_ends, ranks, right=True).clamp_max(k_c - 1)
        ]
        u_key = torch.where(
            unassigned,
            u_min.clamp_max(1e6) + 1e-6 * weyl_p,
            weyl_p.new_full((p_cnt,), 1e12),
        )
        u_ord = torch.argsort(u_key, stable=True)
        fill_id = torch.full((p_cnt,), -1, dtype=torch.int64, device=dev)
        fill_id.scatter_(
            0,
            u_ord,
            torch.where(unassigned[u_ord] & (ranks < slot_ends[-1]), slot_comp, -1),
        )
        do_fill = fill_id >= 0
        fc = fill_id.clamp_min(0)
        return (
            torch.where(do_fill, fc, rc),
            torch.where(do_fill.unsqueeze(1), comp_c[fc], rt),
            torch.where(do_fill, d_pc.gather(1, fc.unsqueeze(1)).squeeze(1), rd),
        )

    @torch.no_grad()
    def update(
        self,
        x_fake: torch.Tensor,
        rows: torch.Tensor,
        comp_c: torch.Tensor,
        comp_w: torch.Tensor,
        num_particles: int,
        *,
        allow_update: bool = True,
    ) -> LedgerAssignment:
        """Compute GPU Hamilton quotas, evict furthest over-quota, and assign deficit particles."""
        b, k_c, p_cnt = int(x_fake.shape[0]), int(comp_c.shape[0]), int(num_particles)
        if rows.ndim != 1 or int(rows.shape[0]) != b:
            raise ValueError("rows must be 1D matching x_fake batch size")
        if k_c < 1 or not (1 <= p_cnt <= MAX_LEDGER_PARTICLES):
            raise ValueError("ledger requires K_C >= 1 and 1 <= num_particles <= 2048")

        dev = x_fake.device
        cf = comp_c.detach().to(device=dev, dtype=torch.float32).reshape(k_c, -1)
        xf = (
            x_fake.detach()
            .to(device=dev, dtype=torch.float32)
            .reshape(b, int(cf.shape[1]))
        )
        rc, rt, rd = self._ensure_storage(p_cnt, int(cf.shape[1]), dev)
        rl = rows.detach().to(device=dev, dtype=torch.int64).clamp(0, p_cnt - 1)
        d_bc = safe_cdist(xf, cf)
        quotas = hamilton_quotas(
            p_cnt,
            comp_w.detach().to(dev),
            active_scores=-d_bc.amin(dim=0),
            min_one_when_feasible=True,
        )
        if allow_update and b > 0:
            in_batch = torch.zeros(p_cnt, dtype=torch.bool, device=dev)
            in_batch[rl] = True
            first_pos = torch.full((p_cnt,), b, dtype=torch.int64, device=dev)
            first_pos.scatter_reduce_(
                0,
                rl,
                torch.arange(b, device=dev, dtype=torch.int64),
                reduce="amin",
                include_self=True,
            )
            d_pc = d_bc[first_pos.clamp_max(b - 1)]
            weyl_p = torch.remainder(
                (torch.arange(p_cnt, device=dev, dtype=torch.float32) + 0.5) * PHI1, 1.0
            )
            rc, rt, rd = self._reconcile_and_evict(
                rc, rt, rd, cf, d_pc, in_batch, quotas, weyl_p
            )
            rc, rt, rd = self._assign_deficit(
                rc, rt, rd, d_pc, cf, in_batch, quotas, weyl_p
            )
            self.row_comp, self.row_target, self.row_dist = rc, rt, rd

        nn_cc = (
            safe_cdist(cf, diag_inf=True).amin(dim=1).clamp_min(1e-12)
            if k_c >= 2
            else torch.ones((k_c,), device=dev, dtype=torch.float32)
        )
        tk = torch.where(rc[rl] >= 0, rc[rl].clamp_min(0), d_bc.argmin(dim=1))
        occ = torch.bincount(
            torch.where(rc >= 0, rc, rc.new_full((), k_c)), minlength=k_c + 1
        )[:k_c]
        n_uns = (
            float(p_cnt)
            - ((rc >= 0) & (rd <= 0.09 * nn_cc[rc.clamp_min(0)])).float().sum()
        )
        r_cov = torch.where(
            (n_uns > 0.5) | (occ < quotas).any(),
            0.18 + 0.12 * (n_uns.clamp_min(1.0) / float(max(1, p_cnt))),
            xf.new_zeros(()),
        )
        r_dist = 1.6 * ((xf - cf[tk]).norm(dim=1) / nn_cc[tk]).clamp_max(10.0).mean()
        return LedgerAssignment(
            tk, cf[tk], nn_cc, quotas, occ, torch.maximum(r_dist, r_cov), rc
        )


# ---------------------------------------------------------------------------
# Clock-Free Anisotropic Prototype Atlas (Schema = 2)
# ---------------------------------------------------------------------------


class PrototypeAtlas:
    """Clock-free Law 1A/1B geometric store with anisotropic LTC covariance & GPU Hamilton ledger."""

    SCHEMA = SCHEMA_VERSION

    def __init__(
        self,
        max_prototypes: int = 256,
        *,
        beta_n: float = _BETA_N,
        device: Optional[torch.device] = None,
    ) -> None:
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an int >= 2")
        self.beta_n: float = _check_float_scalar(
            beta_n, "beta_n", lo=0.0, hi=1.0, strict=True
        )
        self.max_prototypes: int = int(max_prototypes)
        self.device: Optional[torch.device] = device
        self.ledger: HamiltonCommitmentLedger = HamiltonCommitmentLedger()
        self.reset()

    @property
    def proto_var(self) -> Optional[torch.Tensor]:
        return None if self.proto_cov is None else cov_trace_mean(self.proto_cov)

    @property
    def n_eff(self) -> Optional[torch.Tensor]:
        return self.proto_n

    @property
    def num_prototypes(self) -> int:
        return 0 if self.proto_c is None else int(self.proto_c.shape[0])

    @property
    def rho_atom_ema(self) -> float:
        self._flush_scalars()
        return self._rho_atom_f

    @property
    def g_conn_ema(self) -> float:
        self._flush_scalars()
        return self._g_conn_f

    def _flush_scalars(self) -> None:
        if self._rho_atom_t is not None and self._g_conn_t is not None:
            vals = torch.stack(
                (self._rho_atom_t.double(), self._g_conn_t.double())
            ).tolist()
            self._rho_atom_f, self._g_conn_f = float(vals[0]), float(vals[1])
            self._rho_atom_t = self._g_conn_t = None

    def reset(self) -> None:
        self.initialized, self.dim = False, 0
        self.proto_c: Optional[torch.Tensor] = None
        self.proto_cov: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.w_fake_ema: Optional[torch.Tensor] = None
        self.proto_n: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.v_push: Optional[torch.Tensor] = None
        self._rho_atom_t: Optional[torch.Tensor] = None
        self._g_conn_t: Optional[torch.Tensor] = None
        self._rho_atom_f, self._g_conn_f = 1.0, 0.0
        self.ledger.reset()

    def _move_to_device(self, dev: torch.device) -> None:
        self.device = dev
        for name in _ATLAS_TENSOR_KEYS:
            t = getattr(self, name)
            if isinstance(t, torch.Tensor) and t.device != dev:
                setattr(self, name, t.to(dev))
        if self._rho_atom_t is not None and self._rho_atom_t.device != dev:
            self._rho_atom_t = self._rho_atom_t.to(dev)
        if self._g_conn_t is not None and self._g_conn_t.device != dev:
            self._g_conn_t = self._g_conn_t.to(dev)

    def state_dict(self) -> Dict[str, Any]:
        self._flush_scalars()
        return {
            "schema": SCHEMA_VERSION,
            "max_prototypes": int(self.max_prototypes),
            "beta_n": float(self.beta_n),
            "initialized": bool(self.initialized),
            "dim": int(self.dim),
            "rho_atom_ema": float(self._rho_atom_f),
            "g_conn_ema": float(self._g_conn_f),
            **{
                k: None
                if getattr(self, k) is None
                else getattr(self, k).detach().cpu().clone()
                for k in _ATLAS_TENSOR_KEYS
            },
            "ledger": self.ledger.state_dict() if self.ledger.initialized else None,
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        validate_only: bool = False,
    ) -> None:
        val = validate_atlas_state(state)
        if val["max_prototypes"] != self.max_prototypes:
            raise ValueError("PrototypeAtlas max_prototypes mismatch")
        if validate_only:
            return
        dst = device if device is not None else self.device
        self.beta_n, self.initialized, self.dim = (
            float(val["beta_n"]),
            bool(val["initialized"]),
            int(val["dim"]),
        )
        self._rho_atom_f, self._g_conn_f = (
            float(val["rho_atom_ema"]),
            float(val["g_conn_ema"]),
        )
        self._rho_atom_t = self._g_conn_t = None
        for k in _ATLAS_TENSOR_KEYS:
            t = val[k]
            out = (
                None
                if t is None
                else (
                    t.detach().to(device=dst, dtype=torch.float32)
                    if dst is not None
                    else t.detach().float()
                ).clone()
            )
            setattr(self, k, out)
        if val["ledger"] is None:
            self.ledger.reset()
        else:
            self.ledger.load_state_dict(val["ledger"], device=dst)

    @torch.no_grad()
    def merge_pair(self, a: int | torch.Tensor, b_idx: int | torch.Tensor) -> None:
        """Exact Law-of-Total-Covariance merge ``(a, b_idx) -> a`` preserving mass and ``psi``."""
        wa, wb = self.proto_w[a].double(), self.proto_w[b_idx].double()  # type: ignore[index]
        wtot = (wa + wb).clamp_min(1e-12)
        ca, cb = self.proto_c[a].double(), self.proto_c[b_idx].double()  # type: ignore[index]
        c_ab = (wa * ca + wb * cb) / wtot
        cov_a = cov_add_outer(
            self.proto_cov[a].double().unsqueeze(0),
            (ca - c_ab).unsqueeze(0),  # type: ignore[index]
        )[0]
        cov_b = cov_add_outer(
            self.proto_cov[b_idx].double().unsqueeze(0),
            (cb - c_ab).unsqueeze(0),  # type: ignore[index]
        )[0]
        self.proto_c[a] = c_ab.float()  # type: ignore[index]
        self.proto_cov[a] = _scale_sanitize_cov(
            ((wa * cov_a + wb * cov_b) / wtot).unsqueeze(0).float()
        )[0]  # type: ignore[index]
        self.v_push[a] = (
            (  # type: ignore[index]
                (wa * self.v_push[a].double() + wb * self.v_push[b_idx].double()) / wtot  # type: ignore[index]
            )
            .clamp_min(_VAR_FLOOR)
            .float()
        )
        self.psi[a] = (
            (wa * self.psi[a].double() + wb * self.psi[b_idx].double()) / wtot
        ).float()  # type: ignore[index]
        self.psi[b_idx] = 0.0  # type: ignore[index]
        self.proto_w[a], self.proto_w[b_idx] = wtot.float(), 0.0  # type: ignore[index]
        self.w_fake_ema[a], self.w_fake_ema[b_idx] = (  # type: ignore[index]
            self.w_fake_ema[a] + self.w_fake_ema[b_idx],  # type: ignore[index]
            0.0,
        )
        self.proto_n[a], self.proto_n[b_idx] = (
            self.proto_n[a] + self.proto_n[b_idx],
            0.0,
        )  # type: ignore[index]

    @torch.no_grad()
    def split_slot(self, parent: int | torch.Tensor, dst: int | torch.Tensor) -> None:
        """Exact 100%-LTC, strictly-PSD principal-axis split ``parent -> (parent, dst)``."""
        seed = int(parent.item()) if isinstance(parent, torch.Tensor) else int(parent)
        c0 = self.proto_c[parent].double()  # type: ignore[index]
        cov_p = self.proto_cov[parent]  # type: ignore[index]
        _, delta, cov_child = cov_principal_split(cov_p, seed=seed)
        tr_ratio = (
            cov_trace_mean(cov_child.unsqueeze(0))[0]
            / cov_trace_mean(cov_p.unsqueeze(0))[0]
        )
        self.proto_c[parent] = (c0 - delta.double()).float()  # type: ignore[index]
        self.proto_c[dst] = (c0 + delta.double()).float()  # type: ignore[index]
        self.proto_cov[parent], self.proto_cov[dst] = cov_child, cov_child.clone()  # type: ignore[index]
        self.v_push[parent] = self.v_push[dst] = (
            self.v_push[parent] * tr_ratio
        ).clamp_min(_VAR_FLOOR)  # type: ignore[index]
        self.proto_n[parent] = self.proto_n[dst] = 0.5 * self.proto_n[parent]  # type: ignore[index]
        self.proto_w[parent] = self.proto_w[dst] = 0.5 * (
            self.proto_w[parent] + self.proto_w[dst]
        )  # type: ignore[index]
        self.w_fake_ema[parent] = self.w_fake_ema[dst] = 0.5 * (
            self.w_fake_ema[parent] + self.w_fake_ema[dst]
        )  # type: ignore[index]
        self.psi[parent] = self.psi[dst] = 0.5 * (self.psi[parent] + self.psi[dst])  # type: ignore[index]

    def _select_init_centers(self, x: torch.Tensor) -> torch.Tensor:
        b, dim, dev = int(x.shape[0]), int(x.shape[1]), x.device
        raw_sp = 2.0 * x.var(dim=0, unbiased=False).sum()
        if not bool((raw_sp > _VAR_FLOOR).item()):
            return x[:1].clone()
        tol = 1e-5 * raw_sp.sqrt()
        is_first = (safe_cdist(x) <= tol).float().argmax(dim=1) == torch.arange(
            b, device=dev
        )
        uniq = x[is_first]
        n_u = int(uniq.shape[0])
        if n_u < b and n_u <= self.max_prototypes:
            return uniq.clone()
        src = uniq if n_u < b else x
        n_src = int(src.shape[0])
        k0 = (
            self.max_prototypes
            if n_u < b
            else min(
                self.max_prototypes, max(4, (3 * n_src) // 4) if n_src >= 8 else n_src
            )
        )
        proj = (
            torch.remainder(
                torch.arange(1, dim + 1, device=dev, dtype=torch.float32) * PHI1, 1.0
            )
            - 0.5
        )
        order = torch.argsort(src @ (proj / proj.norm().clamp_min(1e-8)), stable=True)
        pick = ((torch.arange(k0, device=dev) * n_src + n_src // 2) // k0).clamp(
            0, n_src - 1
        )
        return src[order[pick]].clone()

    def _init_from_batch(self, x: torch.Tensor) -> None:
        dim, dev = int(x.shape[1]), x.device
        centers = self._select_init_centers(x)
        k0 = int(centers.shape[0])
        assign = safe_cdist(x, centers).argmin(dim=1)
        b_cnt = torch.bincount(assign, minlength=k0).float()
        raw_v = x.var(dim=0, unbiased=False).mean().clamp_min(_VAR_FLOOR) / float(
            max(1, k0)
        )
        if k0 >= 2:
            d_nn = safe_cdist(centers, diag_inf=True).amin(dim=1)
            pos = d_nn > 1e-12
            d_sort = torch.where(pos, d_nn, d_nn.new_tensor(float("inf"))).sort().values
            med_pos = d_sort[(pos.sum() // 2).clamp_max(k0 - 1)]
            v_geom = torch.where(
                pos.sum() > 0,
                (0.04 * med_pos.square() / float(dim)).clamp_min(_VAR_FLOOR),
                raw_v,
            )
        else:
            v_geom = raw_v
        multi = b_cnt >= 2.0
        b_mean = x.new_zeros((k0, dim)).index_add_(0, assign, x) / b_cnt.clamp_min(
            1.0
        ).unsqueeze(1)
        diff = x - b_mean[assign]
        v_emp = (
            diff.new_zeros(k0).index_add_(0, assign, diff.square().mean(dim=1))
            / (b_cnt - 1.0).clamp_min(1.0)
        ).clamp_min(_VAR_FLOOR)
        v_fill = (
            torch.quantile(v_emp[multi], 0.25) if bool(multi.any().item()) else v_geom
        )
        v_init = v_fill.expand(k0).clone()
        self.dim, self.initialized = dim, True
        self.proto_c, self.proto_cov = centers, cov_from_scalar(v_init, dim)
        self.proto_w = b_cnt.clamp_min(1e-12) / b_cnt.clamp_min(1e-12).sum()
        self.w_fake_ema = self.proto_w.clone()
        self.proto_n = b_cnt.clamp_min(1.0)
        self.psi = torch.zeros(k0, device=dev, dtype=torch.float32)
        self.v_push = v_init.clone()
        self.ledger.reset()
        rho0 = batch_annular_gap_ratio(x)
        self._rho_atom_t, self._g_conn_t = (
            rho0,
            compute_connectivity_gate(x, rho_atom=rho0),
        )

    def _pre_spawn_uncovered(self, x: torch.Tensor) -> None:
        k, dim, dev = int(self.proto_c.shape[0]), self.dim, x.device  # type: ignore[union-attr]
        if k >= self.max_prototypes:
            return
        d_min, assign = safe_cdist(x, self.proto_c).min(dim=1)  # type: ignore[arg-type]
        d2_min = d_min.square()
        raw_sp = 2.0 * x.var(dim=0, unbiased=False).sum()
        pair_sp = raw_sp.clamp_min(_VAR_FLOOR)
        v_prev = cov_trace_mean(self.proto_cov)  # type: ignore[arg-type]
        pre_med_v = v_prev.median().clamp_min(_VAR_FLOOR)
        shock_den = torch.maximum(
            pair_sp, 6.0 * float(dim) * v_prev[assign].mean().clamp_min(_VAR_FLOOR)
        )
        shock = ((torch.sqrt(d2_min.mean() / shock_den) - 0.45) / 0.35).clamp(0.0, 1.0)
        if not bool(((raw_sp > _VAR_FLOOR) & (shock <= 0.5)).item()):
            return
        u_x = (x - self.proto_c[assign]) / d_min.clamp_min(1e-12).unsqueeze(1)  # type: ignore[index]
        v_dir = cov_directional_var(self.proto_cov[assign], u_x)  # type: ignore[index]
        denom_ur = torch.maximum(
            36.0 * float(dim) * torch.maximum(v_dir, pre_med_v), 0.008 * pair_sp
        )
        uncov = d2_min > denom_ur
        if not bool(uncov.any().item()):
            return
        u_pts = x[uncov]
        sep_tol = torch.sqrt(
            torch.maximum(36.0 * float(dim) * pre_med_v, 0.008 * pair_sp)
        )
        is_first = (safe_cdist(u_pts) <= sep_tol).float().argmax(dim=1) == torch.arange(
            int(u_pts.shape[0]), device=dev
        )
        new_c = u_pts[is_first][: min(self.max_prototypes - k, 64)]
        n_new = int(new_c.shape[0])
        if n_new < 1:
            return
        w_seed = min(0.5 / float(n_new), 1.0 / float(k + n_new))
        keep_sc = 1.0 - float(n_new) * w_seed
        w_add = new_c.new_full((n_new,), w_seed)
        self.proto_c = torch.cat((self.proto_c, new_c), dim=0)  # type: ignore[arg-type]
        self.proto_cov = torch.cat(
            (self.proto_cov, cov_from_scalar(pre_med_v.expand(n_new), dim)),
            dim=0,  # type: ignore[arg-type]
        )
        self.v_push = torch.cat((self.v_push, pre_med_v.expand(n_new)), dim=0)  # type: ignore[arg-type]
        self.proto_w = torch.cat((self.proto_w * keep_sc, w_add), dim=0)  # type: ignore[operator]
        self.w_fake_ema = torch.cat((self.w_fake_ema * keep_sc, w_add), dim=0)  # type: ignore[operator]
        self.proto_n = torch.cat(
            (self.proto_n, new_c.new_full((n_new,), 1e-6)),
            dim=0,  # type: ignore[arg-type]
        )
        psi_cat = torch.cat((self.psi, new_c.new_zeros(n_new)), dim=0)  # type: ignore[arg-type]
        self.psi = psi_cat - psi_cat.mean()

    def _update_streaming_stats(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        c_prev, cov_prev = self.proto_c.clone(), self.proto_cov.clone()  # type: ignore[union-attr]
        b, dim, k = int(x.shape[0]), int(x.shape[1]), int(c_prev.shape[0])
        is_full = cov_prev.ndim == 3
        v_prev = cov_trace_mean(cov_prev)

        d_min, assign = safe_cdist(x, c_prev).min(dim=1)
        d2_min = d_min.square()
        raw_sp = 2.0 * x.var(dim=0, unbiased=False).sum()
        has_sp, pair_sp = raw_sp > _VAR_FLOOR, raw_sp.clamp_min(_VAR_FLOOR)
        shock_den = torch.maximum(
            pair_sp, 6.0 * float(dim) * v_prev[assign].mean().clamp_min(_VAR_FLOOR)
        )
        shock = (
            (
                torch.where(
                    has_sp, torch.sqrt(d2_min.mean() / shock_den), raw_sp.new_zeros(())
                )
                - 0.45
            )
            / 0.35
        ).clamp(0.0, 1.0)

        b_cnt = torch.bincount(assign, minlength=k).float()
        hit = b_cnt > 0.0
        n_dec = torch.where(
            shock > 0.25, self.proto_n * (1.0 - 0.8 * shock), self.proto_n
        )  # type: ignore[operator]
        self.proto_n = ((1.0 - self.beta_n) * n_dec + b_cnt).clamp_min(1e-6)

        aw = 0.02 + 0.48 * shock * shock
        far_unhit = (
            (shock > 0.25)
            & has_sp
            & (~hit)
            & (
                (c_prev - x.mean(dim=0, keepdim=True)).square().sum(dim=1)
                > 100.0 * pair_sp
            )
        )
        aw_k = torch.where(far_unhit, aw.new_full((), 0.50), aw)
        pw = (1.0 - aw_k) * self.proto_w + aw_k * (b_cnt / float(b))  # type: ignore[operator]
        self.proto_w = pw / pw.sum().clamp_min(1e-12)

        b_mean = x.new_zeros((k, dim)).index_add_(0, assign, x) / b_cnt.clamp_min(
            1.0
        ).unsqueeze(1)
        ac = (
            (
                0.035
                + 0.25 * torch.rsqrt(self.proto_n.clamp_min(1.0))
                + 0.38 * shock * shock
            )
            * (b_cnt / (b_cnt + 2.0))
        ).clamp_max(0.50)
        self.proto_c = torch.where(
            hit.unsqueeze(1),
            (1.0 - ac.unsqueeze(1)) * c_prev + ac.unsqueeze(1) * b_mean,
            c_prev,
        )

        # Unbiased streaming covariance for all b_k >= 1 (B < K safe).
        ref = torch.where((b_cnt >= 2.0).unsqueeze(1), b_mean, c_prev)
        denom = torch.where(b_cnt >= 2.0, b_cnt - 1.0, b_cnt.clamp_min(1.0))
        cov_est = _scale_sanitize_cov(
            _scatter_cov(x - ref[assign], assign, k, is_full), 1.0 / denom
        )
        v_est = cov_trace_mean(cov_est)

        v_tot = (raw_sp / (2.0 * float(dim))).clamp_min(_VAR_FLOOR)
        v_col = has_sp & (b >= 8) & (v_prev > 4.0 * v_tot)
        v_col_g = torch.exp(
            0.15 * torch.log(v_prev)
            + 0.85 * torch.log((v_tot / float(max(1, k))).clamp_min(_VAR_FLOOR))
        ).clamp_min(_VAR_FLOOR)
        cov_base = _scale_sanitize_cov(
            cov_prev, torch.where(v_col, v_col_g / v_prev, torch.ones_like(v_prev))
        )

        exact_col = ((b_cnt >= 2.0) & (v_est < 1e-8 * v_prev)) | (
            (b_cnt >= 1.0) & (v_est <= 1e-12 * v_prev)
        )
        down_jump = exact_col | ((b_cnt >= 6.0) & (v_est < 0.04 * v_prev))
        up_jump = (b_cnt >= 2.0) & (v_est > 25.0 * v_prev)
        v_jump = hit & ((shock > 0.5) | down_jump | up_jump)
        log_r = (torch.log(v_est) - torch.log(v_prev)).abs()
        av_s = (
            0.65
            + 0.25 * torch.maximum(shock * shock, smoothstep01((log_r - 3.2) / 3.2))
        ).clamp_max(0.90)
        s_scale = torch.exp(av_s * (torch.log(v_est) - torch.log(v_prev)))
        cov_shock = _scale_sanitize_cov(cov_prev, s_scale)
        a_vec = 0.12 * b_cnt / (b_cnt + 1.0)
        a_m = a_vec.view(-1, 1, 1) if is_full else a_vec.unsqueeze(1)
        j_m = v_jump.view(-1, 1, 1) if is_full else v_jump.unsqueeze(1)
        h_m = hit.view(-1, 1, 1) if is_full else hit.unsqueeze(1)
        self.proto_cov = _scale_sanitize_cov(
            torch.where(
                h_m,
                torch.where(
                    j_m,
                    (1.0 - a_m) * cov_shock + a_m * cov_est,
                    (1.0 - a_m) * cov_base + a_m * cov_est,
                ),
                cov_base,
            )
        )
        self.v_push = torch.where(
            v_jump,
            (v_prev * s_scale).clamp_min(_VAR_FLOOR),
            torch.where(v_col, v_col_g, self.v_push),  # type: ignore[arg-type]
        )
        return assign, d2_min, v_prev, has_sp, pair_sp, aw

    def _spawn_prototype(
        self,
        y_far: torch.Tensor,
        med_v: torch.Tensor,
        new_w: torch.Tensor,
        dead_flag: bool,
    ) -> None:
        k = int(self.proto_c.shape[0])  # type: ignore[union-attr]
        new_cov = cov_from_scalar(med_v.reshape(1), self.dim)
        if dead_flag:
            slot = self.proto_w.argmin()  # type: ignore[union-attr]
        elif k < self.max_prototypes:
            self.proto_c = torch.cat((self.proto_c, y_far.unsqueeze(0)), dim=0)  # type: ignore[arg-type]
            self.proto_cov = torch.cat((self.proto_cov, new_cov), dim=0)  # type: ignore[arg-type]
            self.v_push = torch.cat((self.v_push, med_v.reshape(1)), dim=0)  # type: ignore[arg-type]
            self.proto_w = torch.cat((self.proto_w, new_w.reshape(1)), dim=0)  # type: ignore[arg-type]
            self.proto_n = torch.cat((self.proto_n, self.proto_n.new_ones((1,))), dim=0)  # type: ignore[union-attr]
            self.w_fake_ema = torch.cat((self.w_fake_ema, new_w.reshape(1)), dim=0)  # type: ignore[arg-type]
            self.psi = torch.cat((self.psi, self.psi.new_zeros((1,))), dim=0)  # type: ignore[union-attr]
            return
        else:
            flat = safe_cdist(self.proto_c, diag_inf=True).square().argmin()  # type: ignore[arg-type]
            self.merge_pair(flat // k, flat % k)
            slot = flat % k
        self.proto_c[slot], self.proto_cov[slot] = y_far, new_cov[0]  # type: ignore[index]
        self.v_push[slot], self.proto_w[slot], self.proto_n[slot] = med_v, new_w, 1.0  # type: ignore[index]
        self.w_fake_ema[slot], self.psi[slot] = new_w, 0.0  # type: ignore[index]

    def _rebalance_prototypes(
        self,
        dead: torch.Tensor,
        dead_flag: bool,
        pi_t: torch.Tensor,
        med_v: torch.Tensor,
        gamma: float,
        mean_w: float,
    ) -> None:
        k, dev = int(self.proto_c.shape[0]), self.proto_c.device  # type: ignore[union-attr]
        idx_k = torch.arange(k, device=dev)
        if not dead_flag:
            im = idx_k == pi_t
            d_min, flat = (
                safe_cdist(self.proto_c, diag_inf=True)  # type: ignore[arg-type]
                .square()
                .masked_fill(im[:, None] | im[None, :], float("inf"))
                .reshape(-1)
                .min(dim=0)
            )
            if bool((d_min <= 8.0 * float(self.dim) * med_v).item()):
                self.merge_pair(flat // k, flat % k)
                b_t = flat % k
                ge = (self.proto_w.clamp_min(1e-12) ** gamma) * cov_trace_mean(
                    self.proto_cov
                )  # type: ignore[union-attr, arg-type]
                pb_t = ge.masked_fill(idx_k == b_t, 0.0).argmax()
                if bool(((pb_t != b_t) & (self.proto_w[pb_t] > 0.20 * mean_w)).item()):  # type: ignore[index]
                    self.split_slot(pb_t, b_t)
            return
        for di in torch.nonzero(dead, as_tuple=False).squeeze(-1)[:4]:
            ge = (self.proto_w.clamp_min(1e-12) ** gamma) * cov_trace_mean(
                self.proto_cov
            )  # type: ignore[union-attr, arg-type]
            pb_t = ge.masked_fill(idx_k == di, 0.0).argmax()
            if not bool(((pb_t != di) & (self.proto_w[pb_t] > 0.20 * mean_w)).item()):  # type: ignore[index]
                break
            self.split_slot(pb_t, di)

    def _maintain_structure(
        self,
        x: torch.Tensor,
        assign: torch.Tensor,
        d2_min: torch.Tensor,
        v_prev: torch.Tensor,
        has_sp: torch.Tensor,
        pair_sp: torch.Tensor,
        aw: torch.Tensor,
    ) -> None:
        k, b, dim = int(self.proto_c.shape[0]), int(x.shape[0]), self.dim  # type: ignore[union-attr]
        mean_w = 1.0 / float(k)
        v_cur = cov_trace_mean(self.proto_cov)  # type: ignore[arg-type]
        pre_med_v = v_prev.median().clamp_min(_VAR_FLOOR)
        post_med_v = v_cur.median().clamp_min(_VAR_FLOOR)
        dead = self.proto_w < (0.04 / float(k))  # type: ignore[operator]
        gamma = float(min(dim, 4)) / float(min(dim, 4) + 2)
        denom_ur = torch.maximum(
            36.0 * float(dim) * torch.maximum(v_prev[assign], pre_med_v),
            0.008 * pair_sp,
        )
        max_ur, worst_j = (d2_min / denom_ur).max(dim=0)
        ge = (self.proto_w.clamp_min(1e-12) ** gamma) * v_cur  # type: ignore[union-attr]
        max_e, pi_t = ge.max(dim=0)
        zf = torch.zeros((), device=x.device, dtype=torch.bool)
        do_uncov = (max_ur > 1.0) & has_sp if k >= 1 else zf
        need_g = (
            (max_e > 10.0 * ge.median().clamp_min(_VAR_FLOOR))
            & (self.proto_w[pi_t] > 0.25 * mean_w)  # type: ignore[index]
            if k >= 16
            else zf
        )
        any_d = dead.any()
        do_sp = (~do_uncov) & has_sp & (any_d | need_g) if k >= 2 else zf
        u_flag, s_flag, dead_flag = torch.stack((do_uncov, do_sp, any_d)).tolist()
        if u_flag:
            new_w = torch.maximum(aw / float(b), aw.new_full((), 0.05 * mean_w))
            self._spawn_prototype(x[worst_j], post_med_v, new_w, dead_flag)
        elif s_flag:
            self._rebalance_prototypes(dead, dead_flag, pi_t, pre_med_v, gamma, mean_w)
        if u_flag or s_flag:
            self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)  # type: ignore[union-attr]
            self.w_fake_ema = self.w_fake_ema / self.w_fake_ema.sum().clamp_min(1e-12)  # type: ignore[union-attr]
            self.psi = self.psi - self.psi.mean()  # type: ignore[union-attr]

    @torch.no_grad()
    def update(self, y_real: torch.Tensor) -> None:
        """Stream one batch of real samples into the anisotropic LTC prototype atlas."""
        if y_real.ndim < 2 or int(y_real.shape[0]) < 1:
            raise ValueError("y_real must have shape (B, ...) with B >= 1")
        x = y_real.detach().reshape(int(y_real.shape[0]), -1).float()
        if int(x.shape[1]) < 1:
            raise ValueError("y_real must have non-zero feature dimension")
        self._move_to_device(x.device)
        if not self.initialized or self.proto_c is None or self.dim != int(x.shape[1]):
            self._init_from_batch(x)
            return
        self._pre_spawn_uncovered(x)
        assign, d2_min, v_prev, has_sp, pair_sp, aw = self._update_streaming_stats(x)
        self._maintain_structure(x, assign, d2_min, v_prev, has_sp, pair_sp, aw)

    @torch.no_grad()
    def coalesce(self, y_real: Optional[torch.Tensor] = None) -> CoalescedAtlas:
        """Compute directional LTC coalescence, MP shrinkage target, and update ``rho_atom``/``g_conn``."""
        if (
            not self.initialized
            or self.proto_c is None
            or self.proto_cov is None
            or self.proto_w is None
        ):
            raise RuntimeError("PrototypeAtlas must be updated before coalescing")
        coal = coalesce_directional_ltc(
            self.proto_c, self.proto_cov, self.proto_w, self.proto_n
        )
        rho_now = compute_rho_atom(
            coal.comp_c, coal.comp_cov, coal.comp_n_eff, y_real=y_real
        )
        prev_r = (
            self._rho_atom_t
            if self._rho_atom_t is not None
            else rho_now.new_full((), self._rho_atom_f)
        )
        self._rho_atom_t = 0.85 * prev_r + 0.15 * rho_now
        if y_real is not None:
            g_now = compute_connectivity_gate(y_real, rho_atom=self._rho_atom_t)
            prev_g = (
                self._g_conn_t
                if self._g_conn_t is not None
                else g_now.new_full((), self._g_conn_f)
            )
            self._g_conn_t = 0.85 * prev_g + 0.15 * g_now
        return coal

    @torch.no_grad()
    def commit_particles(
        self,
        x_fake: torch.Tensor,
        rows: torch.Tensor,
        num_particles: int,
        *,
        coalesced: Optional[CoalescedAtlas] = None,
        allow_update: bool = True,
    ) -> Optional[LedgerAssignment]:
        """Run unified GPU Hamilton ledger whenever ``all_iso``, ``K_C >= 2``, and ``P <= 2048``."""
        if not self.initialized or not (
            1 <= int(num_particles) <= MAX_LEDGER_PARTICLES
        ):
            return None
        coal = self.coalesce() if coalesced is None else coalesced
        if int(coal.comp_c.shape[0]) < 2 or not bool(coal.all_iso.item()):
            return None
        return self.ledger.update(
            x_fake,
            rows,
            coal.comp_c,
            coal.comp_w,
            int(num_particles),
            allow_update=allow_update,
        )


__all__ = [
    "CoalescedAtlas",
    "HamiltonCommitmentLedger",
    "LedgerAssignment",
    "MAX_FULL_COV_DIM",
    "MAX_LEDGER_PARTICLES",
    "PrototypeAtlas",
    "SCHEMA_VERSION",
    "batch_annular_gap_ratio",
    "coalesce_directional_ltc",
    "coalesce_pythagorean_ltv",
    "compute_connectivity_gate",
    "compute_rho_atom",
    "cov_add_outer",
    "cov_directional_var",
    "cov_from_scalar",
    "cov_principal_split",
    "cov_shrink_mp",
    "cov_trace_mean",
    "manifold_connectivity_stats",
    "validate_atlas_state",
    "validate_ledger_state",
]
