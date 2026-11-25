"""Unified-v15 4-Law Anisotropic Covariance Pullback Prior Dynamics.

Upgrades ``_unified_v11/prior.py`` to close D-07 and D-12 while solving anisotropic rotated
ellipses (``fl_law1_aniso_unequal_modes``) and noisy Zipf discrete codebooks
(``fl_law1_noisy_atomic_codebook``) with zero task-specific branching:
1. **Per-Prototype Covariance Tracking & Exact Law of Total Covariance (LTC, D-07):**
   Tracks symmetric PSD ``proto_cov[K, D, D]`` (with ``proto_var == trace(proto_cov) / D``)
   with exact LTC updates on Gersho merges, exact principal-axis LTC splits (retaining
   100% total covariance and exact ``0.84 * v`` isotropic 2D variance instead of ``0.65 * v``),
   and exact ``w_fake_ema`` unit-mass conservation after uncovered-mode teleports.
2. **Directional Anisotropic LTC Coalescence & Whitened Pullback Isometry:**
   Replaces scalar-trace sibling/isolation gates (eliminating the ``kappa* = 8.665``
   anisotropic disconnection cliff and 62%–68% Frobenius covariance collapse) with directional
   covariance gates and exact LTC covariance coalescence, paired with clean-center OT on all
   externally isolated modes (``1 <= |C| <= 4``) and eigenframe-whitened pullback covariance
   matching (zero ``eigh`` / SVD / ``pinv`` / ``autograd.grad`` and zero extra generator passes).
3. **Starvation-Free Zero-Mean Laguerre Dual & Occam Pruning (D-12):**
   Prevents global ``tv_floor`` deadbands from permanently starving minority modes/Zipf atoms,
   enforces zero-mean Laguerre gauge after clamping, and prunes the two proven-redundant
   Round-4 epicycles (4-tangent 4D pullback probe and point-mass forward-AD weighting).
"""
from __future__ import annotations

import math
import struct
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .._bispf.thermal import FROZEN_THERMAL_SCALE, _read_thermal
from .._stats import knn_excluding_self, pairwise_sq_dist
from .._t4_arch.pullback import _deterministic_box_muller
from ..bispf import BISPFGates, SBNPPriorRegularizer
from ..moment_tether import pooled_reference_moments
from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .atomic import _weyl_tie, compute_atomic_quotas
from .loss import _principal_axis_power_zerosync

_PHI1 = 0.6180339887498949

_PRIOR_V15_KEYS = frozenset({
    "schema",
    "quantizer_initialized",
    "proto_c",
    "proto_var",
    "proto_cov",
    "push_cov",
    "proto_w",
    "proto_hits",
    "v_push",
    "psi",
    "w_fake_ema",
    "phase",
    "iso_initialized",
    "iso_ema",
    "res_ema",
    "iso_err_ema",
    "rows_initialized",
    "row_m",
    "row_v",
    "row_n",
    "ledger_initialized",
    "row_pos",
    "row_tgt_pos",
    "row_valid",
})


def _quantile_1d_fast(x: torch.Tensor, q: float) -> torch.Tensor:
    """Bit-exact 1D linear-interpolation quantile matching ``torch.quantile``."""
    n = int(x.numel())
    if n <= 1:
        return x.reshape(()) if n == 1 else x.new_zeros(())
    s = x.reshape(-1).sort().values
    qf = struct.unpack("f", struct.pack("f", float(q)))[0] if x.dtype == torch.float32 else float(q)
    pos = qf * float(n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return s[lo] if lo == hi else torch.lerp(s[lo], s[hi], pos - float(lo))


@torch.no_grad()
def _sym_eig2x2_exact(cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact closed-form 2x2 symmetric PSD eigendecomposition (0 eigh/svd/pinv, 0 syncs).

    Returns ``(lam1, lam2, cos_th, sin_th)`` with scale-relative positive floor and
    principal unit eigenvector ``q1 = (cos_th, sin_th)``, minor ``q2 = (-sin_th, cos_th)``.
    """
    a = cov[..., 0, 0]
    b = 0.5 * (cov[..., 0, 1] + cov[..., 1, 0])
    d = cov[..., 1, 1]
    tr = a + d
    diff = a - d
    disc = torch.sqrt((diff * diff + 4.0 * b * b).clamp_min(0.0))
    eps_eig = (1e-10 * tr.clamp_min(0.0)).clamp_min(1e-24)
    lam1 = torch.maximum(0.5 * (tr + disc), eps_eig)
    lam2 = torch.maximum(0.5 * (tr - disc), eps_eig)
    theta = 0.5 * torch.atan2(2.0 * b, diff)
    return lam1, lam2, torch.cos(theta), torch.sin(theta)


@torch.no_grad()
def _principal_eigenpair(cov: torch.Tensor, fallback_dir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top eigenpair ``(lam1, v1)`` of symmetric PSD ``[D, D]`` with zero ``eigh``."""
    dim = int(cov.shape[0])
    if dim == 2:
        lam1, lam2, c_th, s_th = _sym_eig2x2_exact(cov)
        iso_like = (lam1 - lam2) <= (1e-6 * lam1)
        v_eig = torch.stack((c_th, s_th), dim=0)
        v1 = torch.where(iso_like, fallback_dir, v_eig)
        v1 = v1 / v1.norm().clamp_min(1e-8)
        return lam1, v1
    v1 = _principal_axis_power_zerosync(cov.float(), iters=6, canonicalize_sign=False)
    v1_norm = v1.norm()
    v1 = torch.where(v1_norm > 1e-8, v1 / v1_norm.clamp_min(1e-8), fallback_dir)
    lam1 = (v1 * (cov.float() @ v1)).sum().clamp_min(1e-24)
    return lam1, v1


@torch.no_grad()
def _compute_pythagorean_smoothed_cells_v15(
    proto_c: torch.Tensor,
    proto_cov: torch.Tensor,
    proto_w: torch.Tensor,
    *,
    eye_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact anisotropic Law-of-Total-Covariance coalescence with directional isolation gates.

    Returns ``(c_bar, cov_bar, sigma_bar, w_mat, iso_coal, multicell_coal)``:
    - ``cov_bar[i] = sum_j w_mat[i, j] * (proto_cov[j] + (c_j - c_bar[i])(c_j - c_bar[i])^T)``
      achieving exact 0% covariance Frobenius error on coalesced anisotropic modes.
    - Directional sibling variance ``u_ij^T Sigma u_ij`` eliminates the isotropic ``kappa* = 8.665``
      disconnection cliff while preserving exact parity on isotropic Gaussian lattices.
    - ``iso_coal`` identifies ALL externally isolated modes (``1 <= |C| <= 4``);
      ``multicell_coal`` identifies multi-prototype coalesced modes (``2 <= |C| <= 4``).
    """
    k, dim = proto_c.shape
    if eye_k is None or eye_k.shape != (k, k) or eye_k.device != proto_c.device:
        eye_k = torch.eye(k, device=proto_c.device, dtype=torch.float32)
    proto_var = torch.diagonal(proto_cov, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-24)
    if k < 2:
        zeros_b = torch.zeros(k, device=proto_c.device, dtype=torch.bool)
        return proto_c, proto_cov, proto_var.sqrt(), eye_k, zeros_b, zeros_b

    sqrt_d = math.sqrt(float(dim))
    diff_cc = proto_c.unsqueeze(1) - proto_c.unsqueeze(0)  # [K, K, D]
    d_cc = diff_cc.square().sum(dim=-1).clamp_min(0.0).sqrt()
    u_cc = diff_cc / d_cc.clamp_min(1e-15).unsqueeze(-1)

    # Directional std along inter-centroid vector u_ij (identically sqrt(v_i) when isotropic):
    v_dir_i = torch.einsum("ijd,ide,ije->ij", u_cc, proto_cov, u_cc).clamp_min(1e-24)
    sig_dir_i = v_dir_i.sqrt()
    sig_dir_j = sig_dir_i.transpose(0, 1)
    sig_dir_max = torch.maximum(sig_dir_i, sig_dir_j)
    sig_dir_min = torch.minimum(sig_dir_i, sig_dir_j).clamp_min(1e-12)

    sig_tr = proto_var.sqrt()
    sig_tr_max = torch.maximum(sig_tr.unsqueeze(1), sig_tr.unsqueeze(0))
    sig_tr_min = torch.minimum(sig_tr.unsqueeze(1), sig_tr.unsqueeze(0)).clamp_min(1e-12)

    in_sibling = (
        (d_cc <= (2.6 * sqrt_d * sig_dir_max))
        & (sig_dir_max <= (2.2 * sig_dir_min))
        & (sig_tr_max <= (2.2 * sig_tr_min))
    )
    in_sibling.fill_diagonal_(True)

    adj = in_sibling.float()
    adj = ((adj @ adj) > 0).float()
    adj = ((adj @ adj) > 0).float()
    adj = ((adj @ adj) > 0).float()
    comp_size = adj.sum(dim=1)
    max_comp_sz = float(min(12, max(4, k // 4)))
    cand_coalesce = (comp_size >= 1.0) & (comp_size <= max_comp_sz) & (comp_size < float(k))

    cand_mask = torch.where(cand_coalesce.unsqueeze(1) & cand_coalesce.unsqueeze(0), adj, eye_k)
    cand_w = cand_mask * proto_w.clamp_min(1e-12).unsqueeze(0)
    cand_w.div_(cand_w.sum(dim=1, keepdim=True).clamp_min(1e-12))
    cand_c = cand_w @ proto_c
    cand_diff = proto_c.unsqueeze(0) - cand_c.unsqueeze(1)  # [K, K, D]
    within_cov = torch.einsum("ij,jde->ide", cand_w, proto_cov)
    between_cov = torch.einsum("ij,ijd,ije->ide", cand_w, cand_diff, cand_diff)
    cand_cov = within_cov + between_cov
    cand_var = torch.diagonal(cand_cov, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-24)

    # Directional external isolation check (reduces bit-exact to ext_d >= 4.2*sqrt(D)*cand_sig when isotropic):
    sig_ext_dir = torch.einsum("ijd,ide,ije->ij", u_cc, cand_cov, u_cc).clamp_min(1e-24).sqrt()
    ext_ratio = (d_cc / (4.2 * sqrt_d * sig_ext_dir).clamp_min(1e-12)).masked_fill(adj > 0.5, float("inf")).min(dim=1).values
    node_isolated = ext_ratio >= 1.0
    comp_isolated = (adj @ (~node_isolated).float()) < 0.5

    iso_coal = cand_coalesce & comp_isolated
    multicell_coal = (comp_size > 1.5) & iso_coal

    w_mat = torch.where(iso_coal.unsqueeze(1), cand_w, eye_k)
    c_bar = torch.where(iso_coal.unsqueeze(1), cand_c, proto_c)
    cov_bar = torch.where(iso_coal.view(k, 1, 1), cand_cov, proto_cov)
    var_bar = torch.where(iso_coal, cand_var, proto_var)
    return c_bar, cov_bar, var_bar.sqrt(), w_mat, iso_coal, multicell_coal


@torch.no_grad()
def _sinkhorn_barycenter_fast(
    L_xp: torch.Tensor,
    log_a: float,
    log_b: torch.Tensor,
    c_bar_all: torch.Tensor,
    b: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """10-iteration log-domain Sinkhorn barycenter reusing one pre-allocated ``[B, K]`` workspace."""
    buf = torch.empty_like(L_xp)
    m0 = L_xp.amax(dim=0, keepdim=True)
    torch.sub(L_xp, m0, out=buf)
    buf.exp_()
    v = log_b - (m0.squeeze(0) + buf.sum(dim=0).log_())
    torch.add(L_xp, v.unsqueeze(0), out=buf)
    m1 = buf.amax(dim=1, keepdim=True)
    buf.sub_(m1).exp_()
    u = log_a - (m1.squeeze(1) + buf.sum(dim=1).log_())
    for _ in range(9):
        torch.add(L_xp, u.unsqueeze(1), out=buf)
        m0 = buf.amax(dim=0, keepdim=True)
        buf.sub_(m0).exp_()
        v = log_b - (m0.squeeze(0) + buf.sum(dim=0).log_())
        torch.add(L_xp, v.unsqueeze(0), out=buf)
        m1 = buf.amax(dim=1, keepdim=True)
        buf.sub_(m1).exp_()
        u = log_a - (m1.squeeze(1) + buf.sum(dim=1).log_())
    torch.add(L_xp, u.unsqueeze(1), out=buf)
    buf.add_(v.unsqueeze(0)).exp_().mul_(float(b))
    buf.div_(buf.sum(dim=1, keepdim=True).clamp_min_(1e-12))
    return (buf @ c_bar_all).to(out_dtype)


_TRIU_IDX_CACHE: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_triu_indices(m: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (m, device)
    cached = _TRIU_IDX_CACHE.get(key)
    if cached is None:
        idx = torch.triu_indices(m, m, offset=1, device=device)
        cached = (idx[0], idx[1])
        if len(_TRIU_IDX_CACHE) < 64:
            _TRIU_IDX_CACHE[key] = cached
    return cached


def _knn_sq_and_median_bandwidth_zerosync(z: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-sync exact k-NN squared distances and off-diagonal median squared bandwidth."""
    sq_dist = pairwise_sq_dist(z)
    with torch.no_grad():
        m = int(sq_dist.shape[0])
        if m < 2:
            r_med_sq = sq_dist.new_full((), 1e-12)
        else:
            iu, ju = _get_triu_indices(m, sq_dist.device)
            r_med_sq = torch.median(sq_dist[iu, ju]).clamp_min(1e-12)
    knn_sq_dist, _ = knn_excluding_self(sq_dist, k)
    return knn_sq_dist, r_med_sq


def _has_positive_prior_sigma(prior: nn.Module) -> bool:
    """Zero-sync intrinsic check matching ``_resolve_v15_prior_kernel(...)[1] > 0.0``."""
    base_prior = unwrap_prior(prior)
    if getattr(base_prior, "code_law", "table") != "table":
        return False
    if not bool(getattr(prior, "transforms_codes", False)):
        return isinstance(base_prior, MoGParticlePrior) and bool(getattr(base_prior, "_noise_enabled", False))
    manips = tuple(getattr(prior, "manipulators", ()))
    return (
        len(manips) == 1
        and type(manips[0]).__name__ == "NormalNoiseManipulator"
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
        and float(manips[0].sigma) > 0.0
    )


def _resolve_v15_prior_kernel(
    prior: nn.Module,
    latent: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, float, bool, bool]:
    """Zero-sync extraction of ``(mu_g, sigma_z, has_sigma, can_condense_table)``."""
    base_prior = unwrap_prior(prior)
    if getattr(base_prior, "code_law", "table") != "table":
        return latent.detach(), 0.0, False, False
    if not bool(getattr(prior, "transforms_codes", False)):
        if isinstance(base_prior, MoGParticlePrior):
            has_sigma = bool(getattr(base_prior, "_noise_enabled", False))
            sigma_z = base_prior._cached_sigma_float() if has_sigma else 0.0
            return base_prior.means()[indices], sigma_z, sigma_z > 0.0, True
        return base_prior.z[indices], 0.0, False, True
    manips = tuple(getattr(prior, "manipulators", ()))
    if (
        len(manips) == 1
        and type(manips[0]).__name__ == "NormalNoiseManipulator"
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
    ):
        sig_f = float(manips[0].sigma)
        return base_prior.z[indices], sig_f, sig_f > 0.0, True
    return latent.detach(), 0.0, False, False


def _validate_unified_v15_prior_state(state: Any) -> Dict[str, Any]:
    """Strictly validate schema-3 UnifiedV15PriorRegularizer state (rejecting v11/v12/v14 checkpoints)."""
    if not isinstance(state, dict) or set(state) != _PRIOR_V15_KEYS:
        raise ValueError("invalid UnifiedV15PriorRegularizer checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != 3:
        raise ValueError("unsupported UnifiedV15PriorRegularizer checkpoint schema")
    for bkey in ("quantizer_initialized", "iso_initialized", "rows_initialized", "ledger_initialized"):
        if type(state[bkey]) is not bool:
            raise ValueError(f"invalid UnifiedV15PriorRegularizer boolean field {bkey}")
    for fkey, low, strict_low in (
        ("phase", 0.0, False),
        ("iso_ema", 0.0, True),
        ("res_ema", 0.0, False),
        ("iso_err_ema", 0.0, False),
    ):
        val = state[fkey]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"invalid UnifiedV15PriorRegularizer scalar field {fkey}")
        if float(val) <= low if strict_low else float(val) < low:
            raise ValueError(f"out-of-range UnifiedV15PriorRegularizer scalar field {fkey}")
    if float(state["phase"]) >= 1.0:
        raise ValueError("UnifiedV15PriorRegularizer phase must lie in [0, 1)")

    q_tensors = ("proto_c", "proto_var", "proto_cov", "push_cov", "proto_w", "proto_hits", "v_push", "psi", "w_fake_ema")
    if not state["quantizer_initialized"]:
        if any(state[k] is not None for k in q_tensors):
            raise ValueError("uninitialized UnifiedV15PriorRegularizer quantizer must store None tensors")
    else:
        for k in q_tensors:
            t = state[k]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not bool(torch.isfinite(t).all()):
                raise ValueError(f"invalid UnifiedV15PriorRegularizer {k} tensor")
        c = state["proto_c"]
        if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
            raise ValueError("invalid UnifiedV15PriorRegularizer proto_c shape")
        k_len, dim = int(c.shape[0]), int(c.shape[1])
        for k in ("proto_var", "proto_w", "proto_hits", "psi", "w_fake_ema"):
            if state[k].ndim != 1 or state[k].shape[0] != k_len:
                raise ValueError(f"incompatible UnifiedV15PriorRegularizer {k} shape")
        if tuple(state["v_push"].shape) != (k_len, dim) or bool((state["v_push"] < 0.0).any()):
            raise ValueError("invalid UnifiedV15PriorRegularizer v_push tensor")
        for ck in ("proto_cov", "push_cov"):
            cov = state[ck]
            if tuple(cov.shape) != (k_len, dim, dim) or bool((torch.diagonal(cov, dim1=-2, dim2=-1) <= 0.0).any()):
                raise ValueError(f"invalid UnifiedV15PriorRegularizer {ck} tensor")
        if bool((state["proto_var"] <= 0.0).any()) or bool((state["proto_hits"] <= 0.0).any()):
            raise ValueError("UnifiedV15PriorRegularizer proto_var and proto_hits must be positive")
        for mass_k in ("proto_w", "w_fake_ema"):
            mt = state[mass_k]
            if bool((mt < 0.0).any()) or not math.isclose(float(mt.sum().item()), 1.0, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError(f"UnifiedV15PriorRegularizer {mass_k} must be nonnegative and sum to 1")

    r_tensors = ("row_m", "row_v", "row_n")
    if not state["rows_initialized"]:
        if any(state[k] is not None for k in r_tensors):
            raise ValueError("uninitialized UnifiedV15PriorRegularizer row state must store None tensors")
    else:
        rm, rv, rn = state["row_m"], state["row_v"], state["row_n"]
        for name, t in zip(r_tensors, (rm, rv, rn)):
            if not isinstance(t, torch.Tensor) or not t.is_floating_point() or not bool(torch.isfinite(t).all()):
                raise ValueError(f"invalid UnifiedV15PriorRegularizer {name} tensor")
        if rm.ndim != 2 or rm.shape[0] < 1 or rm.shape[1] < 1 or rv.shape != (rm.shape[0], 1) or rn.shape != (rm.shape[0], 1):
            raise ValueError("incompatible UnifiedV15PriorRegularizer row tensor shapes")
        if bool((rv < 0.0).any()) or bool((rn < 0.0).any()):
            raise ValueError("UnifiedV15PriorRegularizer row_v and row_n must be nonnegative")

    l_tensors = ("row_pos", "row_tgt_pos", "row_valid")
    if not state["ledger_initialized"]:
        if any(state[k] is not None for k in l_tensors):
            raise ValueError("uninitialized UnifiedV15PriorRegularizer ledger must store None tensors")
    else:
        rp, rtp, rval = state["row_pos"], state["row_tgt_pos"], state["row_valid"]
        for name, t in (("row_pos", rp), ("row_tgt_pos", rtp)):
            if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or t.ndim != 2 or not bool(torch.isfinite(t).all()):
                raise ValueError(f"invalid UnifiedV15PriorRegularizer {name} tensor")
        if not isinstance(rval, torch.Tensor) or rval.dtype != torch.bool or rval.shape != (rp.shape[0], 2):
            raise ValueError("invalid UnifiedV15PriorRegularizer row_valid tensor")
        if tuple(rtp.shape) != tuple(rp.shape):
            raise ValueError("incompatible UnifiedV15PriorRegularizer ledger tensor shapes")
    return state


class UnifiedV15PriorRegularizer(nn.Module):
    """Recipe-owned Unified-v15 prior regularizer (``prior_regularizer='unified_v15'``)."""

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
        max_prototypes: int = 256,
        gates: Any = None,
        bound_device: Optional[torch.device] = None,
        bound_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an integer >= 2")
        self.max_prototypes = int(max_prototypes)
        from dataclasses import replace as _dc_replace

        base_gates = BISPFGates() if gates is None else gates
        self._active_iso_enabled = bool(base_gates.active("sbnp_active_iso"))
        self._active_iso_above = int(base_gates.sbnp_active_iso_above)
        self._active_iso_weight = float(base_gates.sbnp_active_iso_weight)
        no_eig_gates = (
            _dc_replace(
                base_gates,
                sbnp_active_iso_enabled=False,
                sbnp_thermal_full_max_dim=max(int(base_gates.sbnp_thermal_full_max_dim), 1 << 30),
            )
            if isinstance(base_gates, BISPFGates)
            else base_gates
        )
        self.fallback = SBNPPriorRegularizer(
            reference, weight=weight, target_std=target_std, eps=eps, gates=no_eig_gates
        )
        self._bound_device = bound_device
        self._bound_shape = bound_shape
        self.unified_record: Any = None
        self.unified_v15_record: Any = None

        self.quantizer_initialized: bool = False
        self.proto_c: Optional[torch.Tensor] = None
        self.proto_var: Optional[torch.Tensor] = None
        self.proto_cov: Optional[torch.Tensor] = None
        self.push_cov: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.proto_hits: Optional[torch.Tensor] = None
        self.v_push: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.w_fake_ema: Optional[torch.Tensor] = None
        self.phase: float = 0.0
        self.iso_initialized: bool = False
        self._iso_ema_val: float = 1.0
        self._res_ema_val: float = 1.0
        self._iso_err_ema_val: float = 0.0
        self._iso_ema_t: Optional[torch.Tensor] = None
        self._res_ema_t: Optional[torch.Tensor] = None
        self._iso_err_ema_t: Optional[torch.Tensor] = None
        self._cached_ref_key: Optional[Tuple[Any, ...]] = None
        self._cached_ref_total_n: Optional[int] = None
        self._last_raw_var_cache: Optional[Tuple[int, float]] = None
        self._cache_k_key: Optional[Tuple[int, torch.device]] = None
        self._cache_idx_k: Optional[torch.Tensor] = None
        self._cache_eye_k: Optional[torch.Tensor] = None
        self._cache_eye2_key: Optional[Tuple[torch.device, torch.dtype]] = None
        self._cache_eye2: Optional[torch.Tensor] = None
        self.rows_initialized: bool = False
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None
        self.ledger_initialized: bool = False
        self.row_pos: Optional[torch.Tensor] = None
        self.row_tgt_pos: Optional[torch.Tensor] = None
        self.row_valid: Optional[torch.Tensor] = None

    def _get_k_helpers(self, k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (k, device)
        if self._cache_k_key != key or self._cache_idx_k is None or self._cache_eye_k is None:
            self._cache_idx_k = torch.arange(k, device=device, dtype=torch.long)
            self._cache_eye_k = torch.eye(k, device=device, dtype=torch.float32)
            self._cache_k_key = key
        return self._cache_idx_k, self._cache_eye_k

    def _get_eye2(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if self._cache_eye2_key != key or self._cache_eye2 is None:
            self._cache_eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
            self._cache_eye2_key = key
        return self._cache_eye2

    def _flush_ema_scalars(self) -> None:
        pairs = [
            (name, getattr(self, attr, None))
            for name, attr in (("iso_ema", "_iso_ema_t"), ("res_ema", "_res_ema_t"), ("iso_err_ema", "_iso_err_ema_t"))
            if getattr(self, attr, None) is not None
        ]
        if not pairs:
            return
        vals = torch.stack([t for _, t in pairs]).tolist()
        for (name, _), v in zip(pairs, vals):
            setattr(self, f"_{name}_val", float(v))
            setattr(self, f"_{name}_t", None)

    def _get_ema_t(self, name: str, device: torch.device) -> torch.Tensor:
        t = getattr(self, f"_{name}_t", None)
        if t is not None and t.device == device:
            return t
        if t is not None:
            self._flush_ema_scalars()
        return torch.full((), float(getattr(self, f"_{name}_val")), device=device, dtype=torch.float64)

    @property
    def iso_ema(self) -> float:
        if self._iso_ema_t is not None:
            self._flush_ema_scalars()
        return self._iso_ema_val

    @iso_ema.setter
    def iso_ema(self, val: float) -> None:
        self._iso_ema_val = float(val)
        self._iso_ema_t = None

    @property
    def res_ema(self) -> float:
        if self._res_ema_t is not None:
            self._flush_ema_scalars()
        return self._res_ema_val

    @res_ema.setter
    def res_ema(self, val: float) -> None:
        self._res_ema_val = float(val)
        self._res_ema_t = None

    @property
    def iso_err_ema(self) -> float:
        if self._iso_err_ema_t is not None:
            self._flush_ema_scalars()
        return self._iso_err_ema_val

    @iso_err_ema.setter
    def iso_err_ema(self, val: float) -> None:
        self._iso_err_ema_val = float(val)
        self._iso_err_ema_t = None

    @property
    def thermal_source(self) -> Any:
        return self.fallback.thermal_source

    @thermal_source.setter
    def thermal_source(self, value: Any) -> None:
        self.fallback.thermal_source = value

    @property
    def weight(self) -> float:
        return self.fallback.weight

    def _sbnp_fallback_zerosync_with_cov(self, rows: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        fb = self.fallback
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if fb.weight == 0.0 or not len(rows):
            return rows[:0].sum(), None
        gates = fb.gates
        n, d = rows.shape
        min_dof = fb._min_dof(d)

        cov: Optional[torch.Tensor] = None
        if fb.references:
            initial_mean, initial_covariance, count_t = pooled_reference_moments(fb.references)
            ref_key = tuple(id(r) for r in fb.references)
            if self._cached_ref_key != ref_key or self._cached_ref_total_n is None:
                self._cached_ref_key = ref_key
                self._cached_ref_total_n = int(count_t.item())
            if self._cached_ref_total_n <= min_dof or n <= min_dof or n < 2:
                return rows[:0].sum(), None
            if d != len(initial_mean):
                raise ValueError("sbnp rows and initial reference dimensions differ")
            mean = rows.mean(0)
            variance_scale = initial_covariance.diagonal().clamp_min(0) + fb.eps
            mean_error = ((mean - initial_mean).square() / variance_scale).mean()
            target_sigma = variance_scale.sqrt() * fb.target_std
            centered = rows - mean
            std = (centered.square().mean(0) + fb.eps).sqrt()
            var_loss = F.relu(target_sigma - std).square().mean()
            if d > 1:
                cov = (centered.T @ centered) / max(n - 1, 1)
                denom = variance_scale.unsqueeze(0) * variance_scale.unsqueeze(1) + 1e-6
                if gates.active("sbnp_full_cov") and d <= gates.sbnp_full_cov_max_dim and n <= gates.sbnp_full_cov_max_rows:
                    cov_loss = ((cov - initial_covariance).square() / denom).sum()
                else:
                    cov_loss = ((cov.square() / denom).sum() - (cov.diagonal().square() / denom.diagonal()).sum()) / float(d * d)
            else:
                cov_loss = rows.new_zeros(())
        else:
            if n <= min_dof or n < 2:
                return rows[:0].sum(), None
            mean = rows.mean(0)
            mean_error = rows.new_zeros(())
            centered = rows - mean
            std = (centered.square().mean(0) + fb.eps).sqrt()
            var_loss = F.relu(fb.target_std - std).square().mean()
            if d > 1:
                cov = (centered.T @ centered) / max(n - 1, 1)
                t_var_s = fb.target_std * fb.target_std + fb.eps
                cov_loss = (cov.square().sum() - cov.diagonal().square().sum()) / ((t_var_s * t_var_s + 1e-6) * float(d * d))
            else:
                cov_loss = rows.new_zeros(())

        m_sub = min(n, fb.repel_max_samples)
        z_sub = rows[(torch.arange(m_sub, device=rows.device) * n) // m_sub] if m_sub < n else rows

        local_iso_loss = rows.new_zeros(())
        if gates.active("sbnp_local_iso") and d == gates.sbnp_local_iso_dim and not fb.references and m_sub > gates.sbnp_local_iso_above:
            with torch.no_grad():
                k_iso = min(max(fb.k_nn, d + 1), m_sub - 1)
                _, iso_knn_idx = knn_excluding_self(pairwise_sq_dist(z_sub), k_iso)
            delta_z = z_sub[iso_knn_idx] - z_sub.unsqueeze(1)
            m_loc = torch.einsum("nkd,nke->nde", delta_z, delta_z) / max(1, k_iso)
            tr_m = torch.diagonal(m_loc, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            a_loc = m_loc - (tr_m / d) * torch.eye(d, device=rows.device, dtype=rows.dtype).unsqueeze(0)
            local_iso_loss = (a_loc.pow(2).sum(dim=(-2, -1)) / ((tr_m.squeeze(-1).squeeze(-1) / d) + 1e-6).pow(2)).mean()

        is_small_latent = fb._is_small_latent(n, d)
        if is_small_latent:
            eff_k_nn = min(m_sub - 1, max(fb.k_nn, gates.sbnp_small_latent_min_k_nn))
            eff_repel_weight = fb.repel_weight * gates.sbnp_small_latent_repel_mult
            effective_margin = gates.sbnp_small_latent_margin
        else:
            eff_k_nn = fb.k_nn
            eff_repel_weight = fb.repel_weight
            small_margin = gates.active("sbnp_small_margin") and m_sub <= gates.sbnp_small_margin_max_rows
            effective_margin = gates.sbnp_small_margin if small_margin else fb.repel_margin
        k = min(eff_k_nn, m_sub - 1)
        if k > 0 and eff_repel_weight > 0.0:
            knn_sq_dist, r_med_sq = _knn_sq_and_median_bandwidth_zerosync(z_sub, k)
            repel_loss = F.softplus(-torch.log((knn_sq_dist / (effective_margin * r_med_sq)).clamp_min(1e-12))).mean()
        else:
            repel_loss = rows.new_zeros(())

        global_moments = mean_error + var_loss + cov_loss
        scale_w = fb._applied_weight(is_small_latent)
        s = _read_thermal(fb.thermal_source, rows.device)
        if s <= FROZEN_THERMAL_SCALE:
            out = scale_w * (global_moments + eff_repel_weight * repel_loss + gates.sbnp_local_iso_weight * local_iso_loss)
            if gates.active("sbnp_thermal_full") and d <= gates.sbnp_thermal_full_max_dim and out.requires_grad:
                out = out * s + out.detach() * (1.0 - s)
            return out, cov
        if global_moments.requires_grad and s < 1.0:
            global_moments = global_moments * s + global_moments.detach() * (1.0 - s)
        return scale_w * (global_moments + eff_repel_weight * repel_loss + gates.sbnp_local_iso_weight * local_iso_loss), cov

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        reused_cov: Optional[torch.Tensor] = None
        if type(self.fallback) is SBNPPriorRegularizer:
            out, reused_cov = self._sbnp_fallback_zerosync_with_cov(rows)
        else:
            out = self.fallback(rows)
        if self.weight == 0.0 or rows.ndim != 2 or not len(rows):
            return out
        n, d = rows.shape
        if not self._active_iso_enabled or d < 2:
            return out
        m_sub = min(n, self.fallback.repel_max_samples)
        if m_sub <= self._active_iso_above or n <= self.fallback._min_dof(d):
            return out
        if m_sub == n and reused_cov is not None:
            cov_mat = reused_cov
        else:
            z_sub = rows[(torch.arange(m_sub, device=rows.device) * n) // m_sub] if m_sub < n else rows
            row_c = z_sub - z_sub.mean(dim=0, keepdim=True)
            cov_mat = (row_c.T @ row_c) / float(max(1, m_sub - 1))
        if d == 2:
            c00, c01, c11 = cov_mat[0, 0], cov_mat[0, 1], cov_mat[1, 1]
            active_iso = self._active_iso_weight * ((c00 - c11).square() + 4.0 * c01.square()) / (c00 + c11 + 1e-6).square()
        else:
            with torch.no_grad():
                cdet = cov_mat.detach().float()
                v1 = _principal_axis_power_zerosync(cdet, iters=6, canonicalize_sign=False)
                lam1_d = (v1 * (cdet @ v1)).sum().clamp_min(0.0)
                v2 = _principal_axis_power_zerosync(cdet - lam1_d * torch.outer(v1, v1), iters=6, canonicalize_sign=False)
                v2 = v2 - (v2 @ v1) * v1
                v2 = v2 / v2.norm().clamp_min(1e-12)
            v1_t, v2_t = v1.to(dtype=cov_mat.dtype), v2.to(dtype=cov_mat.dtype)
            l1 = (v1_t * (cov_mat @ v1_t)).sum().clamp_min(0.0)
            l2 = (v2_t * (cov_mat @ v2_t)).sum().clamp_min(0.0)
            active_iso = self._active_iso_weight * (l1 - l2).square() / (l1 + l2 + 1e-6).square()
        scale_w = self.fallback._applied_weight(self.fallback._is_small_latent(n, d))
        s = _read_thermal(self.thermal_source, rows.device)
        term = scale_w * active_iso
        if term.requires_grad and (
            (s <= FROZEN_THERMAL_SCALE and self.fallback.gates.active("sbnp_thermal_full") and d <= self.fallback.gates.sbnp_thermal_full_max_dim)
            or (s < 1.0 and d > 4)
        ):
            term = term * s + term.detach() * (1.0 - s)
        return out + term

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs

        def _cl(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t.detach().clone()

        return {
            "schema": 3,
            "quantizer_initialized": bool(self.quantizer_initialized),
            "proto_c": _cl(self.proto_c),
            "proto_var": _cl(self.proto_var),
            "proto_cov": _cl(self.proto_cov),
            "push_cov": _cl(self.push_cov),
            "proto_w": _cl(self.proto_w),
            "proto_hits": _cl(self.proto_hits),
            "v_push": _cl(self.v_push),
            "psi": _cl(self.psi),
            "w_fake_ema": _cl(self.w_fake_ema),
            "phase": float(self.phase),
            "iso_initialized": bool(self.iso_initialized),
            "iso_ema": float(self.iso_ema),
            "res_ema": float(self.res_ema),
            "iso_err_ema": float(self.iso_err_ema),
            "rows_initialized": bool(self.rows_initialized),
            "row_m": _cl(self.row_m),
            "row_v": _cl(self.row_v),
            "row_n": _cl(self.row_n),
            "ledger_initialized": bool(self.ledger_initialized),
            "row_pos": _cl(self.row_pos),
            "row_tgt_pos": _cl(self.row_tgt_pos),
            "row_valid": _cl(self.row_valid),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        del strict
        validated = _validate_unified_v15_prior_state(state_dict)
        if validated["rows_initialized"] and self._bound_shape is not None:
            if tuple(validated["row_m"].shape) != tuple(self._bound_shape):
                raise ValueError("UnifiedV15PriorRegularizer row_m shape does not match bound prior table")
        if validated["ledger_initialized"] and self._bound_shape is not None:
            if int(validated["row_pos"].shape[0]) != int(self._bound_shape[0]):
                raise ValueError("UnifiedV15PriorRegularizer row_pos count does not match bound prior table")
        if validated["quantizer_initialized"] and validated["proto_c"].shape[0] > self.max_prototypes:
            raise ValueError("UnifiedV15PriorRegularizer proto_c count exceeds configured max_prototypes")

        dev = self._bound_device

        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.to(device=dev).detach().clone() if dev is not None else t.detach().clone()

        self.quantizer_initialized = bool(validated["quantizer_initialized"])
        for key in (
            "proto_c", "proto_var", "proto_cov", "push_cov", "proto_w", "proto_hits",
            "v_push", "psi", "w_fake_ema", "row_m", "row_v", "row_n",
            "row_pos", "row_tgt_pos", "row_valid",
        ):
            setattr(self, key, _mv(validated[key]))
        self.phase = float(validated["phase"])
        self.iso_initialized = bool(validated["iso_initialized"])
        self.iso_ema = float(validated["iso_ema"])
        self.res_ema = float(validated["res_ema"])
        self.iso_err_ema = float(validated["iso_err_ema"])
        self.rows_initialized = bool(validated["rows_initialized"])
        self.ledger_initialized = bool(validated["ledger_initialized"])

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        if self._bound_device is None:
            self._bound_device = yr.device
        elif self.proto_c is not None and self.proto_c.device != yr.device:
            dev = yr.device
            for attr in (
                "proto_c", "proto_var", "proto_cov", "push_cov", "proto_w", "proto_hits",
                "v_push", "psi", "w_fake_ema", "row_m", "row_v", "row_n",
                "row_pos", "row_tgt_pos", "row_valid",
            ):
                val = getattr(self, attr, None)
                if isinstance(val, torch.Tensor) and val.device != dev:
                    setattr(self, attr, val.to(device=dev))

        b, dim = yr.shape
        x = yr.float()
        eye_d = torch.eye(dim, device=x.device, dtype=torch.float32)

        if not self.quantizer_initialized or self.proto_c is None or self.proto_c.shape[1] != dim:
            k = min(self.max_prototypes, max(4, (3 * b) // 4)) if b >= 8 else min(self.max_prototypes, b)
            idx_d = torch.arange(1, dim + 1, device=x.device, dtype=x.dtype)
            proj = torch.remainder(idx_d * _PHI1, 1.0) - 0.5
            proj = proj / proj.norm().clamp_min(1e-8)
            order = torch.argsort(x @ proj, stable=True)
            slots = ((torch.arange(k, device=x.device) * b + b // 2) // k).clamp(0, b - 1)
            self.proto_c = x[order[slots]].clone()
            if k >= 2:
                d_cc0 = torch.cdist(self.proto_c, self.proto_c)
                d_cc0.fill_diagonal_(float("inf"))
                d_nn = d_cc0.min(dim=1).values
                nn_tol = 1e-7 * float(d_nn.amax().clamp_min(1e-30).item())
                pos_nn = d_nn[d_nn > nn_tol]
                if len(pos_nn) > 0:
                    d_med = float(pos_nn.median().item())
                    v0 = 0.25 * (d_med * d_med) / float(dim + 2)
                else:
                    v0 = float(x.var(dim=0, unbiased=False).mean().clamp_min(1e-24).item()) / float(k)
            else:
                v0 = float(x.var(dim=0, unbiased=False).mean().clamp_min(1e-24).item())
            self.proto_var = torch.full((k,), max(v0, 1e-24), device=x.device, dtype=torch.float32)
            self.proto_cov = self.proto_var.view(k, 1, 1) * eye_d.unsqueeze(0)
            self.push_cov = self.proto_cov.clone()
            self.proto_w = torch.full((k,), 1.0 / float(k), device=x.device, dtype=torch.float32)
            self.proto_hits = torch.ones(k, device=x.device, dtype=torch.float32)
            self.w_fake_ema = torch.full((k,), 1.0 / float(k), device=x.device, dtype=torch.float32)
            self.psi = torch.zeros(k, device=x.device, dtype=torch.float32)
            self.quantizer_initialized = True
            self.v_push = self.proto_var.unsqueeze(1).expand_as(self.proto_c).clone()
            return

        k = len(self.proto_c)
        if self.proto_cov is None or tuple(self.proto_cov.shape) != (k, dim, dim) or self.proto_cov.device != x.device:
            self.proto_cov = self.proto_var.clamp_min(1e-24).view(k, 1, 1) * eye_d.unsqueeze(0)
        if self.push_cov is None or tuple(self.push_cov.shape) != (k, dim, dim) or self.push_cov.device != x.device:
            self.push_cov = self.proto_cov.clone()

        c_prev = self.proto_c.clone()
        d2 = torch.cdist(x, c_prev, p=2.0).square()
        d2_min, assign = d2.min(dim=1)

        diff_prev = x - c_prev[assign]
        outer_prev = (diff_prev.unsqueeze(2) * diff_prev.unsqueeze(1)).reshape(b, dim * dim)
        packed = torch.cat((x.new_ones((b, 1)), x, outer_prev), dim=1)
        sums = x.new_zeros((k, 1 + dim + dim * dim))
        sums.index_add_(0, assign, packed)
        b_cnt = sums[:, 0]
        b_sum = sums[:, 1 : 1 + dim]
        b_cov_sum = sums[:, 1 + dim :].view(k, dim, dim)
        b_w = b_cnt / float(b)

        raw_spread = 2.0 * x.var(dim=0, unbiased=False).sum()
        x_energy = x.square().sum(dim=-1).mean().clamp_min(1e-24)
        has_spread_t = raw_spread > (1e-11 * x_energy)
        pair_spread = raw_spread.clamp_min(1e-24)
        var_floor = (1e-10 * pair_spread).clamp_min(1e-24)
        r2p_t = torch.where(has_spread_t, torch.sqrt(d2_min.mean() / pair_spread).double(), raw_spread.new_zeros((), dtype=torch.float64))
        shock_t64 = ((r2p_t - 0.45) / 0.35).clamp(0.0, 1.0)
        self.proto_hits = torch.where(
            shock_t64 > 0.25,
            (self.proto_hits * (1.0 - 0.8 * shock_t64).float()).clamp_min(1.0),
            self.proto_hits,
        )

        alpha_w_t64 = 0.02 + 0.48 * (shock_t64 * shock_t64)
        alpha_w_t = alpha_w_t64.float()
        self.proto_w = (1.0 - alpha_w_t64).float() * self.proto_w + alpha_w_t * b_w
        self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

        hit = b_cnt > 0
        b_cnt_safe = b_cnt.clamp_min(1.0)
        cov_est = b_cov_sum / b_cnt_safe.view(k, 1, 1) + (var_floor * eye_d.unsqueeze(0))
        # Symmetric hit-annealed covariance EMA: fast (0.14-0.25) on newly initialized/split/teleported
        # prototypes, smoothly annealing to 0.045 once settled (zero one-sided shrinkage bias).
        alpha_base_c = (0.045 + 0.16 / torch.sqrt(self.proto_hits)).clamp_max(0.22)
        alpha_cov_t = (alpha_base_c + (0.48 * (shock_t64 * shock_t64)).float()).clamp_max(0.60).view(k, 1, 1)
        self.proto_cov = torch.where(
            hit.view(k, 1, 1),
            (1.0 - alpha_cov_t) * self.proto_cov + alpha_cov_t * cov_est,
            self.proto_cov,
        )
        self.proto_var = torch.diagonal(self.proto_cov, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(var_floor)

        b_mean = b_sum / b_cnt_safe.unsqueeze(1)
        new_hits = (self.proto_hits + b_cnt).clamp_max(2000.0)
        self.proto_hits = torch.where(hit, new_hits, self.proto_hits)
        base_c = (0.035 + 0.25 / torch.sqrt(self.proto_hits)) + (0.38 * (shock_t64 * shock_t64)).float()
        alpha_c = (base_c * (b_cnt / (b_cnt + 2.0))).clamp_max(0.50).unsqueeze(1)
        self.proto_c = torch.where(hit.unsqueeze(1), (1.0 - alpha_c) * c_prev + alpha_c * b_mean, c_prev)

        med_v = self.proto_var.median().clamp_min(var_floor)
        if dim == 2:
            lam_max_k = _sym_eig2x2_exact(self.proto_cov)[0]
        else:
            lam_max_k = torch.diagonal(self.proto_cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(var_floor)
        v_loc_max = torch.maximum(lam_max_k[assign], lam_max_k.median().clamp_min(var_floor))

        mean_w = 1.0 / float(k)
        dead = self.proto_w < (0.04 * mean_w)
        uncovered_ratio = d2_min / torch.maximum(36.0 * float(dim) * v_loc_max, 1e-3 * pair_spread)
        max_ur, worst_j = uncovered_ratio.max(dim=0)
        do_uncovered_t = (max_ur > 1.0) & has_spread_t if k >= 8 else raw_spread.new_zeros((), dtype=torch.bool)

        d_eff = float(min(dim, 4))
        gamma = d_eff / (d_eff + 2.0)
        gersho_e = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
        med_e = gersho_e.median().clamp_min(1e-24)
        max_e, pi_t = gersho_e.max(dim=0)
        w_pi = self.proto_w.index_select(0, pi_t.unsqueeze(0)).squeeze(0)
        need_gersho_split_t = (
            (max_e > 10.0 * med_e) & (w_pi > 0.25 * mean_w)
            if k >= 4
            else raw_spread.new_zeros((), dtype=torch.bool)
        )
        any_dead_t = dead.any()
        do_split_t = (
            (~do_uncovered_t) & has_spread_t & (any_dead_t | need_gersho_split_t)
            if k >= 2
            else raw_spread.new_zeros((), dtype=torch.bool)
        )

        u_flag, s_flag, dead_flag = torch.stack((do_uncovered_t, do_split_t, any_dead_t)).tolist()
        if u_flag:
            idx_k = torch.arange(k, device=x.device, dtype=torch.long)
            if dead_flag:
                slot_t = self.proto_w.argmin()
            else:
                d_cc = torch.cdist(self.proto_c, self.proto_c).square()
                d_cc.fill_diagonal_(float("inf"))
                slot_t = d_cc.min(dim=1).values.argmin()
            is_slot = idx_k == slot_t
            x_worst = x.index_select(0, worst_j.unsqueeze(0))
            self.proto_c = torch.where(is_slot.unsqueeze(1), x_worst, self.proto_c)
            self.proto_cov = torch.where(is_slot.view(k, 1, 1), (med_v * eye_d).unsqueeze(0), self.proto_cov)
            self.push_cov = torch.where(is_slot.view(k, 1, 1), (med_v * eye_d).unsqueeze(0), self.push_cov)
            self.proto_var = torch.where(is_slot, med_v, self.proto_var)
            self.proto_hits = torch.where(is_slot, self.proto_hits.new_full((), 2.0), self.proto_hits)
            new_w_t = torch.maximum(alpha_w_t / float(b), alpha_w_t.new_full((), 0.05 * mean_w))
            self.proto_w = torch.where(is_slot, new_w_t, self.proto_w)
            self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
            # D-07 fix: renormalize w_fake_ema cleanly to conserve exact unit mass.
            self.w_fake_ema = torch.where(is_slot, new_w_t, self.w_fake_ema)
            self.w_fake_ema = self.w_fake_ema / self.w_fake_ema.sum().clamp_min(1e-12)
            psi_new = torch.where(is_slot, self.psi.new_zeros(()), self.psi)
            self.psi = psi_new - psi_new.mean()
        elif s_flag:
            dim_idx = torch.arange(1, dim + 1, device=x.device, dtype=torch.float32)
            idx_k = torch.arange(k, device=x.device, dtype=torch.long)

            def _ltc_split_child(p_idx_1: torch.Tensor, phase_ang: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ang = dim_idx * (phase_ang * 2.0 * math.pi)
                fallback_dir = torch.cos(ang)
                fallback_dir = fallback_dir / fallback_dir.norm().clamp_min(1e-8)
                cov_pb = self.proto_cov.index_select(0, p_idx_1).squeeze(0)
                var_pb = torch.diagonal(cov_pb, dim1=-2, dim2=-1).mean().clamp_min(var_floor)
                lam1, v1 = _principal_eigenpair(cov_pb, fallback_dir)
                # Exact LTC split: Sigma_child = Sigma_pb - delta delta^T -> exact 0.84*v on isotropic 2D.
                alpha_sq = torch.minimum(var_pb.new_full((), 0.4225), (0.16 * float(dim) * var_pb) / lam1.clamp_min(var_floor))
                delta = (alpha_sq * lam1).clamp_min(var_floor).sqrt() * v1
                cov_child = cov_pb - torch.outer(delta, delta)
                cov_child = 0.5 * (cov_child + cov_child.transpose(0, 1))
                # Ensure strictly SPD diagonal floor while preserving exact LTC trace
                diag_c = torch.diagonal(cov_child, dim1=-2, dim2=-1)
                if bool((diag_c < var_floor).any()):
                    cov_child = cov_child + (var_floor - diag_c.amin()).clamp_min(0.0) * eye_d
                var_child = torch.diagonal(cov_child, dim1=-2, dim2=-1).mean().clamp_min(var_floor)
                return delta, cov_child, var_child

            if dead_flag:
                dead_ord = torch.where(dead, idx_k, idx_k.new_full((), k)).sort().values[:4]
                n_cand_slots = min(4, k)
                still_active = idx_k.new_ones((), dtype=torch.bool)
                for off in range(n_cand_slots):
                    di_t = dead_ord[off]
                    valid_slot = di_t < k
                    di_safe = di_t.clamp_max(k - 1)
                    is_di = idx_k == di_safe
                    ge = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
                    ge = torch.where(is_di | (self.proto_w <= (0.20 * mean_w)), ge.new_zeros(()), ge)
                    p_best_t = ge.argmax()
                    p_best_1 = p_best_t.unsqueeze(0)
                    di_1 = di_safe.unsqueeze(0)
                    w_best = self.proto_w.index_select(0, p_best_1).squeeze(0)
                    still_active = still_active & valid_slot & (p_best_t != di_safe) & (w_best > 0.20 * mean_w)
                    is_pb = idx_k == p_best_t
                    delta, cov_child, var_child = _ltc_split_child(p_best_1, self.phase + (off + 1) * _PHI1)
                    c_orig = self.proto_c.index_select(0, p_best_1).squeeze(0)
                    self.proto_c = torch.where((still_active & is_pb).unsqueeze(1), (c_orig - delta).unsqueeze(0), self.proto_c)
                    self.proto_c = torch.where((still_active & is_di).unsqueeze(1), (c_orig + delta).unsqueeze(0), self.proto_c)
                    hw = 0.5 * (w_best + self.proto_w.index_select(0, di_1).squeeze(0))
                    both_m = still_active & (is_pb | is_di)
                    self.proto_w = torch.where(both_m, hw, self.proto_w)
                    self.proto_cov = torch.where(both_m.view(k, 1, 1), cov_child.unsqueeze(0), self.proto_cov)
                    self.push_cov = torch.where(both_m.view(k, 1, 1), cov_child.unsqueeze(0), self.push_cov)
                    self.proto_var = torch.where(both_m, var_child, self.proto_var)
                    hits_pb = self.proto_hits.index_select(0, p_best_1).squeeze(0)
                    self.proto_hits = torch.where(both_m, (0.5 * hits_pb).clamp(2.0, 6.0), self.proto_hits)
                    apsi = 0.5 * (self.psi.index_select(0, p_best_1).squeeze(0) + self.psi.index_select(0, di_1).squeeze(0))
                    self.psi = torch.where(both_m, apsi, self.psi)
                    af = 0.5 * (self.w_fake_ema.index_select(0, p_best_1).squeeze(0) + self.w_fake_ema.index_select(0, di_1).squeeze(0))
                    self.w_fake_ema = torch.where(both_m, af, self.w_fake_ema)
                self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
                self.w_fake_ema = self.w_fake_ema / self.w_fake_ema.sum().clamp_min(1e-12)
                self.phase = (self.phase + _PHI1) % 1.0
            else:
                d_cc = torch.cdist(self.proto_c, self.proto_c).square()
                d_cc.fill_diagonal_(float("inf"))
                is_pi = idx_k == pi_t
                d_cc = d_cc.masked_fill(is_pi.unsqueeze(1) | is_pi.unsqueeze(0), float("inf"))
                min_pair_d2 = d_cc.min()
                do_merge = min_pair_d2 <= (16.0 * float(dim) * med_v)
                flat_min = d_cc.argmin()
                a_t = torch.div(flat_min, k, rounding_mode="floor")
                b_t = flat_min - a_t * k
                a_1 = a_t.unsqueeze(0)
                b_1 = b_t.unsqueeze(0)
                is_a = idx_k == a_t
                is_b = idx_k == b_t
                wa = self.proto_w.index_select(0, a_1).squeeze(0)
                wb = self.proto_w.index_select(0, b_1).squeeze(0)
                wtot64 = (wa.double() + wb.double()).clamp_min(1e-12)
                wtot = wtot64.float()
                inv_wtot = (1.0 / wtot64).float()
                ca = self.proto_c.index_select(0, a_1)
                cb = self.proto_c.index_select(0, b_1)
                # D-07 fix: exact Law of Total Covariance on Gersho merge-before-split.
                cova = self.proto_cov.index_select(0, a_1).squeeze(0)
                covb = self.proto_cov.index_select(0, b_1).squeeze(0)
                diff_ab = (ca - cb).squeeze(0)
                cov_merged = (wa * cova + wb * covb) * inv_wtot + ((wa * wb) * (inv_wtot * inv_wtot)) * torch.outer(diff_ab, diff_ab)
                var_merged = torch.diagonal(cov_merged, dim1=-2, dim2=-1).mean().clamp_min(var_floor)

                self.proto_c = torch.where((do_merge & is_a).unsqueeze(1), (wa * ca + wb * cb) * inv_wtot, self.proto_c)
                self.proto_cov = torch.where((do_merge & is_a).view(k, 1, 1), cov_merged.unsqueeze(0), self.proto_cov)
                self.push_cov = torch.where((do_merge & is_a).view(k, 1, 1), cov_merged.unsqueeze(0), self.push_cov)
                self.proto_var = torch.where(do_merge & is_a, var_merged, self.proto_var)
                self.proto_w = torch.where(do_merge & is_a, wtot, self.proto_w)
                self.proto_w = torch.where(do_merge & is_b, self.proto_w.new_zeros(()), self.proto_w)
                wf_sum = self.w_fake_ema.index_select(0, a_1).squeeze(0) + self.w_fake_ema.index_select(0, b_1).squeeze(0)
                self.w_fake_ema = torch.where(do_merge & is_a, wf_sum, self.w_fake_ema)
                self.w_fake_ema = torch.where(do_merge & is_b, self.w_fake_ema.new_zeros(()), self.w_fake_ema)

                ge = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
                ge = torch.where(is_b | (self.proto_w <= (0.20 * mean_w)), ge.new_zeros(()), ge)
                p_best_t = ge.argmax()
                p_best_1 = p_best_t.unsqueeze(0)
                w_best = self.proto_w.index_select(0, p_best_1).squeeze(0)
                do_sp = do_merge & (p_best_t != b_t) & (w_best > 0.20 * mean_w)
                is_pb = idx_k == p_best_t
                delta, cov_child, var_child = _ltc_split_child(p_best_1, self.phase + _PHI1)
                c_orig = self.proto_c.index_select(0, p_best_1).squeeze(0)
                self.proto_c = torch.where((do_sp & is_pb).unsqueeze(1), (c_orig - delta).unsqueeze(0), self.proto_c)
                self.proto_c = torch.where((do_sp & is_b).unsqueeze(1), (c_orig + delta).unsqueeze(0), self.proto_c)
                hw = 0.5 * (w_best + self.proto_w.index_select(0, b_1).squeeze(0))
                both_m = do_sp & (is_pb | is_b)
                self.proto_w = torch.where(both_m, hw, self.proto_w)
                self.proto_cov = torch.where(both_m.view(k, 1, 1), cov_child.unsqueeze(0), self.proto_cov)
                self.push_cov = torch.where(both_m.view(k, 1, 1), cov_child.unsqueeze(0), self.push_cov)
                self.proto_var = torch.where(both_m, var_child, self.proto_var)
                hits_pb = self.proto_hits.index_select(0, p_best_1).squeeze(0)
                self.proto_hits = torch.where(both_m, (0.5 * hits_pb).clamp(2.0, 6.0), self.proto_hits)
                apsi = 0.5 * (self.psi.index_select(0, p_best_1).squeeze(0) + self.psi.index_select(0, b_1).squeeze(0))
                self.psi = torch.where(both_m, apsi, self.psi)
                af = 0.5 * (self.w_fake_ema.index_select(0, p_best_1).squeeze(0) + self.w_fake_ema.index_select(0, b_1).squeeze(0))
                self.w_fake_ema = torch.where(both_m, af, self.w_fake_ema)
                self.proto_w = torch.where(do_merge, self.proto_w / self.proto_w.sum().clamp_min(1e-12), self.proto_w)
                self.w_fake_ema = torch.where(do_merge, self.w_fake_ema / self.w_fake_ema.sum().clamp_min(1e-12), self.w_fake_ema)
                if bool(do_merge.item()):
                    self.phase = (self.phase + _PHI1) % 1.0

        if (
            self.quantizer_initialized
            and self.proto_c is not None
            and (self.v_push is None or self.v_push.shape != self.proto_c.shape or self.v_push.device != self.proto_c.device)
        ):
            self.v_push = self.proto_var.unsqueeze(1).expand_as(self.proto_c).clone()

    _update_quantizer = update_quantizer


@torch.no_grad()
def _update_isolated_component_ledger(
    reg: UnifiedV15PriorRegularizer,
    x_ctr_det: torch.Tensor,
    indices: torch.Tensor,
    num_particles: int,
    c_bar_all: torch.Tensor,
    cov_bar_all: torch.Tensor,
    sigma_bar_all: torch.Tensor,
    w_mat_all: torch.Tensor,
    *,
    need_whiten: bool,
    real_batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Coordinate-anchored, quantization-exact 1-stays / surplus-moves ledger on isolated components.

    Eliminates both B < K minibatch Sinkhorn barycentric ghosts and discrete-capacity Laguerre
    limit-cycle thrashing while remaining 100% invariant to prototype re-indexing/splitting/merging.
    """
    dev = x_ctr_det.device
    dim = int(x_ctr_det.shape[1])
    p_total = int(num_particles)

    comp_rep = (w_mat_all > 0.0).float().argmax(dim=1)
    uniq_reps, inv_k = torch.unique(comp_rep, sorted=True, return_inverse=True)
    m_comp = int(uniq_reps.shape[0])
    comp_c = c_bar_all[uniq_reps]
    comp_cov = cov_bar_all[uniq_reps]
    comp_sig = sigma_bar_all[uniq_reps]
    if dim == 2:
        comp_sig_min = _sym_eig2x2_exact(comp_cov)[1].sqrt()
    else:
        comp_sig_min = torch.diagonal(comp_cov, dim1=-2, dim2=-1).amin(dim=-1).clamp_min(1e-24).sqrt()
    assert reg.proto_w is not None
    comp_w = torch.zeros(m_comp, device=dev, dtype=torch.float32)
    comp_w.index_add_(0, inv_k, reg.proto_w.float())
    comp_w = comp_w / comp_w.sum().clamp_min(1e-12)

    if (
        not reg.ledger_initialized
        or reg.row_pos is None
        or reg.row_tgt_pos is None
        or reg.row_valid is None
        or tuple(reg.row_pos.shape) != (p_total, dim)
        or reg.row_pos.device != dev
    ):
        reg.row_pos = torch.zeros((p_total, dim), device=dev, dtype=torch.float32)
        reg.row_tgt_pos = torch.zeros((p_total, dim), device=dev, dtype=torch.float32)
        reg.row_valid = torch.zeros((p_total, 2), device=dev, dtype=torch.bool)
        reg.ledger_initialized = True

    idx_dev = indices.to(device=dev, dtype=torch.long)
    reg.row_pos.index_copy_(0, idx_dev, x_ctr_det)
    reg.row_valid[idx_dev, 0] = True

    scale = torch.maximum(reg.row_pos.abs().amax(), comp_c.abs().amax()).clamp_min(1e-12)
    d_pc = torch.cdist(reg.row_pos / scale, comp_c / scale, p=2.0) * scale
    d_near, near_c = d_pc.min(dim=1)

    if m_comp >= 2:
        d_cc = torch.cdist(comp_c / scale, comp_c / scale, p=2.0) * scale
        d_cc.fill_diagonal_(float("inf"))
        nn_comp = d_cc.amin(dim=1).clamp_min(1e-12 * scale)
    else:
        nn_comp = torch.full((1,), float("inf"), device=dev, dtype=torch.float32)

    d_tc = torch.cdist(reg.row_tgt_pos / scale, comp_c / scale, p=2.0) * scale
    d_t_min, prev_t = d_tc.min(dim=1)
    match_tol = torch.minimum(
        0.45 * nn_comp[prev_t],
        torch.maximum(3.0 * math.sqrt(float(dim)) * comp_sig[prev_t], 0.25 * nn_comp[prev_t]),
    )
    prev_t = torch.where(reg.row_valid[:, 1] & (d_t_min <= match_tol), prev_t, prev_t.new_full((), -1))

    pos_valid_mask = reg.row_valid[:, 0]
    if bool(pos_valid_mask.any()):
        d_valid_min = d_pc[pos_valid_mask].amin(dim=0)
        aff_scale = nn_comp.clamp_max(float(scale.item())) if m_comp >= 2 else comp_sig.clamp_min(1e-12)
        chart_aff = torch.exp(-(d_valid_min / aff_scale.clamp_min(1e-12))).detach().cpu()
    else:
        chart_aff = None
    quotas = compute_atomic_quotas(p_total, comp_w, chart_affinity=chart_aff)
    q_list = quotas.tolist()

    if need_whiten and m_comp >= 4 and int(pos_valid_mask.sum().item()) >= 4:
        vp = reg.row_pos[pos_valid_mask]
        vp_mean = vp.mean(dim=0, keepdim=True)
        vp_std = vp.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
        c_mean = (comp_w.unsqueeze(1) * comp_c).sum(dim=0, keepdim=True)
        c_std = ((comp_w.unsqueeze(1) * (comp_c - c_mean).square()).sum(dim=0, keepdim=True)).sqrt().clamp_min(1e-12)
        d_assign_cpu = torch.cdist((reg.row_pos - vp_mean) / vp_std, (comp_c - c_mean) / c_std, p=2.0).detach().cpu()
        tie_scale = 1e-6
    else:
        d_assign_cpu = d_pc.detach().cpu()
        tie_scale = 1e-6 * float(scale.item())

    pv_list = pos_valid_mask.detach().cpu().tolist()
    nc_list = near_c.detach().cpu().tolist()
    dn_list = d_near.detach().cpu().tolist()
    pt_list = prev_t.detach().cpu().tolist()

    members: list[list[int]] = [[] for _ in range(m_comp)]
    for pid, cid in enumerate(pt_list):
        if 0 <= cid < m_comp:
            members[cid].append(pid)

    occ = [len(mem) for mem in members]
    # Hysteresis-guarded quota enforcement: immediately rescue any 0-occupancy component with q>=1,
    # and only evict established particles for mass rebalancing when excess/deficit exceeds the
    # minibatch EMA jitter band (prevents B < P rounding-boundary particle thrashing).
    cw_list = comp_w.detach().cpu().tolist()
    n_exp = [float(p_total) * cw_list[c] for c in range(m_comp)]
    # Exact Binomial-EMA standard error of proto_w (alpha_w=0.02) at batch size real_batch_size:
    # prevents B < P minibatch weight jitter from evicting settled particles across the space.
    b_real_f = float(max(1, real_batch_size))
    ema_var_fac = (0.02 / 1.98) / b_real_f
    hyst = [
        max(0.85, 2.6 * float(p_total) * math.sqrt(max(ema_var_fac * cw_list[c] * max(1.0 - cw_list[c], 0.0), 0.0)))
        for c in range(m_comp)
    ]
    n_zero_need = sum(1 for c in range(m_comp) if occ[c] == 0 and q_list[c] >= 1)
    n_unseen = sum(1 for pid in range(p_total) if pt_list[pid] < 0)

    # Effective release ceilings: keep particles within [n_exp - hyst, n_exp + hyst] unless needed
    # to cover an empty component.
    eff_cap = [max(q_list[c], int(math.floor(n_exp[c] + hyst[c]))) for c in range(m_comp)]
    if n_zero_need > n_unseen:
        need_more = n_zero_need - n_unseen
        # Release from components with largest surplus above max(1, q_list[c])
        donors = sorted(
            [c for c in range(m_comp) if occ[c] > max(1, q_list[c])],
            key=lambda c: (occ[c] - n_exp[c], occ[c]),
            reverse=True,
        )
        for c in donors:
            if need_more <= 0:
                break
            give = min(need_more, occ[c] - max(1, q_list[c]))
            eff_cap[c] = min(eff_cap[c], occ[c] - give)
            need_more -= give

    # Also release genuine large-surplus components when genuine large-deficit components exist.
    big_def = sum(max(0, q_list[c] - occ[c]) for c in range(m_comp) if (n_exp[c] - occ[c]) > hyst[c])
    if big_def > 0:
        for c in range(m_comp):
            if (occ[c] - n_exp[c]) > hyst[c]:
                eff_cap[c] = min(eff_cap[c], q_list[c])

    for cid in range(m_comp):
        cap = eff_cap[cid]
        if occ[cid] > cap:
            mem = members[cid]
            mem.sort(key=lambda pid: (0 if (pv_list[pid] and nc_list[pid] == cid) else 1, dn_list[pid] if pv_list[pid] else 1e30, pid))
            for pid in mem[cap:]:
                pt_list[pid] = -1
            occ[cid] = cap

    unassigned = [pid for pid in range(p_total) if pv_list[pid] and pt_list[pid] < 0]
    if unassigned:
        unassigned.sort(key=lambda pid: (dn_list[pid], pid))
        # Prioritize zero-occupancy components first, then deficit components below q_list, then any below eff_cap.
        tie_c = tie_scale * _weyl_tie(m_comp, device="cpu")
        for pid in unassigned:
            zero_mask = torch.tensor([occ[c] == 0 and q_list[c] >= 1 for c in range(m_comp)], dtype=torch.bool)
            if bool(zero_mask.any()):
                avail = zero_mask
            else:
                def_mask = torch.tensor([occ[c] < q_list[c] for c in range(m_comp)], dtype=torch.bool)
                if bool(def_mask.any()):
                    avail = def_mask
                else:
                    cap_mask = torch.tensor([occ[c] < eff_cap[c] for c in range(m_comp)], dtype=torch.bool)
                    avail = cap_mask
            if not bool(avail.any()):
                cid_best = nc_list[pid]
                pt_list[pid] = cid_best
                occ[cid_best] += 1
                continue
            scores = (d_assign_cpu[pid] + tie_c).masked_fill(~avail, float("inf"))
            cid_best = int(scores.argmin().item())
            pt_list[pid] = cid_best
            occ[cid_best] += 1

    assigned_all = torch.tensor(pt_list, device=dev, dtype=torch.long)
    valid_assign = assigned_all >= 0
    safe_assign = assigned_all.clamp(0, m_comp - 1)
    reg.row_tgt_pos = torch.where(valid_assign.unsqueeze(1), comp_c[safe_assign], reg.row_tgt_pos)
    reg.row_valid[:, 1] = valid_assign

    # Synchronize reg.w_fake_ema to the exact discrete component occupancy (sum == 1.0).
    occ_t = torch.tensor(occ, device=dev, dtype=torch.float32)
    occ_sum = occ_t.sum()
    if float(occ_sum.item()) > 0.0:
        comp_wf = occ_t / occ_sum
        proto_share = reg.proto_w.float() / comp_w[inv_k].clamp_min(1e-12)
        wf_new = comp_wf[inv_k] * proto_share
        reg.w_fake_ema = wf_new / wf_new.sum().clamp_min(1e-12)

    # Measure unsettled fraction using minor-axis halo so anisotropic modes condense cleanly.
    tgt_dist = (reg.row_pos - comp_c[safe_assign]).norm(dim=-1)
    in_halo = pos_valid_mask & valid_assign & (tgt_dist <= 2.5 * math.sqrt(float(dim)) * comp_sig_min[safe_assign])
    unsettled_frac = float(1.0 - in_halo.float().mean().item())
    return assigned_all, uniq_reps, comp_c, comp_cov, comp_sig_min, comp_w, unsettled_frac


def _is_point_mass_batch(real: torch.Tensor) -> bool:
    """Scale-invariant point-mass predicate (never false-triggers on 1e-5x continuous clouds)."""
    xf = real.detach().float().reshape(len(real), -1)
    raw_var = float(xf.var(dim=0, unbiased=False).sum().item())
    ref_sq = float(xf.square().sum(dim=-1).mean().item())
    return raw_var <= 1e-11 * max(ref_sq, 1e-24)


def should_use_unified_v15_pullback(
    trainer: Any,
    real: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Intrinsic geometric predicate for Unified-v15 Pullback + Rail-Escape + Laguerre Condensation."""
    del latent, indices
    recipe = trainer.recipe
    if recipe.prior_regularizer != "unified_v15" or float(recipe.t1_pullback_weight) <= 0.0:
        return False
    if recipe.encoder_mode != "none" or recipe.conditioning != "scalar":
        return False
    if real.ndim != 2 or not (1 <= int(real.shape[1]) <= 16):
        return False
    if int(real.shape[1]) >= 2 and _has_positive_prior_sigma(trainer.prior):
        return True
    is_pm = _is_point_mass_batch(real)
    reg = getattr(trainer, "prior_regularizer", None)
    if reg is not None and hasattr(reg, "_last_raw_var_cache"):
        reg._last_raw_var_cache = (id(real), 0.0 if is_pm else 1.0)
    return is_pm


def _point_mass_rail_escape_step(
    trainer: Any,
    real: torch.Tensor,
    generator_real: Any,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    sigma_out: Any,
    noise: torch.Generator,
    critic: nn.Module,
    loss_d: torch.Tensor,
    penalty: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Case A: Occam-pruned point-mass & tanh rail-escape step (zero forward-AD probe epicycle)."""
    recipe = trainer.recipe
    reg: UnifiedV15PriorRegularizer = trainer.prior_regularizer
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)

    fake_g = trainer._postprocess(
        trainer.prior,
        trainer._generate(trainer.G, latent, sigma_out, noise, rows=indices),
    )
    fake_flat = fake_g.reshape(b, -1)
    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(None)

    fake_logits = critic(fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:] or len(real_g) != len(real):
        raise ValueError("generator_real must match the real batch shape")
    real_logits = critic(real_g)

    rec_d = getattr(trainer.opt_d, "record", None)
    if rec_d is not None and hasattr(rec_d, "observe_margin_and_scale"):
        rec_d.observe_margin_and_scale(real_logits, fake_logits, real_g, fake_g)

    loss_gan = F.softplus(-fake_logits).mean() if recipe.loss == "non_saturating" else trainer.loss.g_loss(fake_logits, real_logits)
    yr = real_g.detach().reshape(b, -1)

    with torch.no_grad():
        err_vec = (fake_flat.detach().float() - yr.float()).norm(dim=-1)
        cur_res = max(float(err_vec.mean().item()), 0.35 * float(err_vec.amax().item()))
        if cur_res > max(3.5 * reg.res_ema, reg.res_ema + 0.12):
            reg.res_ema = min(cur_res, 10.0)
        else:
            reg.res_ema = 0.88 * reg.res_ema + 0.12 * min(cur_res, 10.0)
        g_scale_pt = max(0.015, min(1.0, reg.res_ema / 0.025))

    loss_pt = (torch.sqrt(1e-6 + (fake_flat - yr).square()) - 1e-3).mean()
    if base_prior.z.requires_grad:
        loss_pt = loss_pt + 0.004 * base_prior.z.float().square().mean().to(loss_pt.dtype)

    transport = recipe.kinetic_transport_loss(fake_g, real_g) if recipe.kinetic_transport_weight else loss_gan.new_zeros(())
    transport_local = (
        recipe.kinetic_transport_local_loss(fake_g, real_g) if recipe.kinetic_transport_local_weight else loss_gan.new_zeros(())
    )
    prior_reg_diag = loss_gan.new_zeros(())
    if base_prior.z.requires_grad and (recipe.prior_regularizer == "vicreg" or recipe.prior_reg > 0):
        raw = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        prior_reg_diag = trainer.prior_regularizer(raw).detach()

    gan_blend_pt = 0.02 * min(1.0, reg.res_ema / 0.25)
    loss_g = gan_blend_pt * loss_gan + float(recipe.t1_pullback_weight) * loss_pt
    if base_prior.z.requires_grad and recipe.prior_l2 > 0:
        loss_g = loss_g + recipe.prior_l2 * base_prior.z.square().mean()

    prior_manip_loss = None
    if trainer._prior_penalty:
        raw_rows = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        prior_manip_loss = trainer.prior.regularize(
            raw_rows, sampled_codes=latent, indices=indices, generated=fake_g, logits=fake_logits
        )
        loss_g = loss_g + prior_manip_loss
    if recipe.kinetic_transport_weight:
        loss_g = loss_g + transport
    if recipe.kinetic_transport_local_weight:
        loss_g = loss_g + transport_local

    if rec_d is not None and hasattr(rec_d, "_t1c_disc_gate"):
        rec_d._t1c_disc_gate = 0.0
    if hasattr(trainer.opt_g, "bind_v4_modulation"):
        trainer.opt_g.bind_v4_modulation(g_scale=g_scale_pt, wakeup_boost=1.0, prior_scale=g_scale_pt)

    trainer.opt_g.zero_grad()
    if getattr(trainer.opt_g, "direction_blend", False) and hasattr(trainer.opt_g, "bind_protected_from_grads"):
        loss_g.backward()
        trainer.opt_g.bind_protected_from_grads()
    else:
        constraint_geometry_backward(loss_g, trainer.opt_g, (loss_g,))
    trainer.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
    trainer.opt_g.step()
    trainer.policy.after_generator_step()
    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


def unified_v15_pullback_step(
    trainer: Any,
    real: torch.Tensor,
    generator_real: Any,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    sigma_out: Any,
    noise: torch.Generator,
    critic: nn.Module,
    loss_d: torch.Tensor,
    penalty: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Execute Unified-v15 4-Law Anisotropic LTC Pullback + Starvation-Free Laguerre Condensation."""
    reg: UnifiedV15PriorRegularizer = trainer.prior_regularizer
    rv_cache = getattr(reg, "_last_raw_var_cache", None)
    reg._last_raw_var_cache = None
    if rv_cache is not None and rv_cache[0] == id(real):
        is_pm = rv_cache[1] < 0.5
    else:
        is_pm = _is_point_mass_batch(real)
    if is_pm:
        return _point_mass_rail_escape_step(
            trainer, real, generator_real, latent, indices,
            sigma_out=sigma_out, noise=noise, critic=critic, loss_d=loss_d, penalty=penalty,
        )

    recipe = trainer.recipe
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)
    mu_g, sigma_z, has_sigma, can_condense_table = _resolve_v15_prior_kernel(trainer.prior, latent, indices)

    # --- PART A: Smooth Generator Chart G_theta + Anisotropic Whitened Pullback Covariance ---
    mu_det = mu_g.detach()
    if has_sigma:
        eps_g = ((latent - mu_g) / sigma_z).detach()
        latent_det = mu_det + sigma_z * eps_g
        fake_g = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, latent_det, sigma_out, noise, rows=indices))
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_flat = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, mu_det, 0.0, noise)).reshape(b, -1)
    else:
        eps_g = None
        latent_det = mu_det
        fake_g = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, mu_det, sigma_out, noise, rows=indices))
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_flat = fake_flat

    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(fake_g)

    fake_logits = critic(fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:] or len(real_g) != len(real):
        raise ValueError("generator_real must match the real batch shape")
    real_logits = critic(real_g)

    rec_d = getattr(trainer.opt_d, "record", None)
    if rec_d is not None and hasattr(rec_d, "observe_margin_and_scale"):
        rec_d.observe_margin_and_scale(real_logits, fake_logits, real_g, fake_g)

    loss_gan = F.softplus(-fake_logits).mean() if recipe.loss == "non_saturating" else trainer.loss.g_loss(fake_logits, real_logits)
    yr = real_g.detach().reshape(b, -1)
    dim = int(yr.shape[1])

    with torch.no_grad():
        reg.update_quantizer(yr)
        yf = yr.float()
        r_macro = yf.std(dim=0, unbiased=False).mean()
        y_abs_ref = yf.abs().amax().clamp_min(1e-12)
        is_point_t = r_macro <= (1e-6 * y_abs_ref)
        r_macro_safe = r_macro.clamp_min(1e-12)

        k_proto = len(reg.proto_c)
        idx_k, eye_k = reg._get_k_helpers(k_proto, yr.device)
        assert reg.proto_cov is not None
        c_bar_all, cov_bar_all, sigma_bar_all, w_mat_all, iso_coal, multicell_coal = _compute_pythagorean_smoothed_cells_v15(
            reg.proto_c, reg.proto_cov, reg.proto_w, eye_k=eye_k
        )
        sig_floor_t = (1e-5 * r_macro_safe).clamp_min(1e-12)
        sigma_cell_med = _quantile_1d_fast(sigma_bar_all, 0.35).clamp_min(sig_floor_t)

        x_ctr_det = x_ctr_flat.detach().float()
        fake_det = fake_flat.detach().float()
        d_ctr = torch.cdist(x_ctr_det, c_bar_all, p=2.0)
        near_p_G = d_ctr.argmin(dim=1)
        # Clean-center OT applies to ALL externally isolated modes (1 <= |C| <= 4), eliminating
        # contractive L2 shrinkage on both coalesced and singleton isolated modes.
        is_coal_i = iso_coal[near_p_G]
        has_iso_coal_t = iso_coal.any()
        all_proto_iso_t = iso_coal.all()
        has_multicell_coal_t = multicell_coal.any()
        any_coal_i_t = is_coal_i.any()
        all_coal_t = is_coal_i.all()
        xff = torch.where(is_coal_i.unsqueeze(1), x_ctr_det, fake_det)

        f_std = xff.std(dim=0, unbiased=False).mean().clamp_min(1e-15)
        scale_ratio_t = f_std / r_macro_safe
        wakeup_boost_t = torch.where(
            scale_ratio_t < 0.50,
            scale_ratio_t.new_full((), 5.0),
            torch.where(scale_ratio_t < 0.80, scale_ratio_t.new_full((), 2.0), scale_ratio_t.new_full((), 1.0)),
        )

        # v_push isotropic dispersion applies strictly to dim == 2, d_z == 2 near-isotropic modes
        # (where v_push is actively tracked); never inject isotropic dispersion into d_z != 2 or
        # anisotropic ellipses!
        v_mu_star_cand: Optional[torch.Tensor] = None
        if reg.v_push is not None and dim == 2 and int(mu_det.shape[1]) == 2:
            lam1_c, lam2_c, _, _ = _sym_eig2x2_exact(cov_bar_all)
            near_iso_k = (lam1_c / lam2_c) <= 1.8
            v_bar_sq = sigma_bar_all.square().unsqueeze(1)
            v_push_bar = (w_mat_all @ reg.v_push).clamp_min(0.0)
            deficit_ratio = ((v_bar_sq - v_push_bar) / v_bar_sq.clamp_min(1e-24)).clamp(0.0, 1.0)
            u_def = ((deficit_ratio - 0.15) / 0.12).clamp(0.0, 1.0)
            v_mu_star_cand = torch.where(
                near_iso_k.unsqueeze(1),
                (u_def * u_def * (3.0 - 2.0 * u_def)) * (v_bar_sq - v_push_bar).clamp_min(0.0),
                v_bar_sq.new_zeros((k_proto, dim)),
            )
            any_v_mu_star_t = has_multicell_coal_t & (v_mu_star_cand.amax() > 0.0)
        else:
            any_v_mu_star_t = has_multicell_coal_t.new_zeros((), dtype=torch.bool)

        (
            is_point_flag,
            has_iso_flag,
            all_proto_iso_flag,
            has_coal_flag,
            any_coal_i_flag,
            all_coal_flag,
            need_whiten_flag,
            scale_ge_065_flag,
            any_v_mu_star_flag,
        ) = torch.stack((
            is_point_t,
            has_iso_coal_t,
            all_proto_iso_t,
            has_multicell_coal_t,
            any_coal_i_t,
            all_coal_t,
            scale_ratio_t < 0.35,
            scale_ratio_t >= 0.65,
            any_v_mu_star_t,
        )).tolist()

        if has_sigma and not is_point_flag:
            iso_t = (sigma_z * r_macro_safe / sigma_cell_med).double().clamp(1.0, 4.5)
            if not reg.iso_initialized:
                reg._iso_ema_t = iso_t
                reg.iso_initialized = True
            else:
                reg._iso_ema_t = 0.95 * reg._get_ema_t("iso_ema", yr.device) + 0.05 * iso_t
        else:
            reg.iso_ema = 1.0
            reg.iso_initialized = True

        # Coordinate-anchored discrete component ledger when ALL prototypes are externally isolated
        # and we have a finite learnable particle table (P <= 2048): eliminates B < K Sinkhorn
        # barycentric ghosts and discrete-capacity Laguerre limit-cycle thrashing!
        use_ledger = (
            bool(all_proto_iso_flag)
            and not is_point_flag
            and can_condense_table
            and base_prior.z.requires_grad
            and indices is not None
            and len(base_prior.z) <= 2048
        )
        full_table_ledger = use_ledger and len(base_prior.z) <= 512
        ledger_assigned: Optional[torch.Tensor] = None
        ledger_uniq_reps: Optional[torch.Tensor] = None
        ledger_comp_c: Optional[torch.Tensor] = None
        ledger_comp_cov: Optional[torch.Tensor] = None
        ledger_comp_sig: Optional[torch.Tensor] = None
        ledger_comp_w: Optional[torch.Tensor] = None
        ledger_unsettled: float = 0.0
        x_ctr_all_grad: Optional[torch.Tensor] = None

        prev_res64 = reg._get_ema_t("res_ema", yr.device)
        if use_ledger:
            assert indices is not None
            if full_table_ledger:
                c_all_det = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z).detach()
                with torch.enable_grad():
                    x_ctr_all_grad = trainer._postprocess(
                        trainer.prior, trainer._generate(trainer.G, c_all_det, 0.0, noise)
                    ).reshape(len(base_prior.z), -1)
                ledger_pos_in = x_ctr_all_grad.detach().float()
                ledger_idx_in = torch.arange(len(base_prior.z), device=yr.device, dtype=torch.long)
            else:
                ledger_pos_in = x_ctr_det
                ledger_idx_in = indices
            (
                ledger_assigned,
                ledger_uniq_reps,
                ledger_comp_c,
                ledger_comp_cov,
                ledger_comp_sig,
                ledger_comp_w,
                ledger_unsettled,
            ) = _update_isolated_component_ledger(
                reg,
                ledger_pos_in,
                ledger_idx_in,
                len(base_prior.z),
                c_bar_all,
                cov_bar_all,
                sigma_bar_all,
                w_mat_all,
                need_whiten=bool(need_whiten_flag),
                real_batch_size=b,
            )
            batch_comp = ledger_assigned[indices.to(device=yr.device, dtype=torch.long)]
            target_p_G = ledger_uniq_reps[batch_comp]
            y_ot = ledger_comp_c[batch_comp].to(fake_flat.dtype)
            # Full-table geometric residual across all observed particles eliminates B < K noise
            # without feeding minor-axis settling bounce back into G's learning rate.
            assert reg.row_pos is not None and reg.row_valid is not None
            pos_v = reg.row_valid[:, 0]
            d_full = torch.cdist(reg.row_pos[pos_v], ledger_comp_c, p=2.0)
            p2f = d_full.min(dim=0).values
            w_p2f = (ledger_comp_w * p2f).sum()
            f2p = d_full.min(dim=1).values.median()
            cell_ref = 1.45 * math.sqrt(float(dim)) * sigma_cell_med
            cur_res_t = ((torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro_safe).double()
            res_ema64 = 0.92 * prev_res64 + 0.08 * cur_res_t.clamp_max(10.0)
            reg._res_ema_t = res_ema64
        else:
            target_p_G = near_p_G
            d_xp_phys = d_ctr if all_coal_flag else torch.cdist(xff, c_bar_all, p=2.0)
            if need_whiten_flag and k_proto >= 4:
                xff_w = (xff - xff.mean(dim=0, keepdim=True)) / xff.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
                cb_mean = (reg.proto_w.unsqueeze(1) * c_bar_all).sum(dim=0, keepdim=True)
                cb_std = ((reg.proto_w.unsqueeze(1) * (c_bar_all - cb_mean).square()).sum(dim=0, keepdim=True)).sqrt().clamp_min(1e-12)
                d_xp_route = torch.cdist(xff_w, (c_bar_all - cb_mean) / cb_std, p=2.0)
            else:
                d_xp_route = d_xp_phys

            c_scale = d_xp_route.mean().clamp_min(1e-12)
            eps_s_t64 = (0.25 * prev_res64).clamp(0.010, 0.06)
            L_xp = -(d_xp_route / c_scale) * (1.0 / eps_s_t64).float()
            y_ot = _sinkhorn_barycenter_fast(L_xp, -math.log(float(b)), torch.log(reg.proto_w.clamp_min(1e-12)), c_bar_all, b, fake_flat.dtype)

            p2f = d_xp_phys.min(dim=0).values
            w_p2f = (reg.proto_w * p2f).sum()
            f2p = d_xp_phys.min(dim=1).values.median()
            cell_ref = 1.45 * math.sqrt(float(dim)) * sigma_cell_med
            cur_res_t = d_xp_phys.mean().double() if is_point_flag else ((torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro_safe).double()
            res_ema64 = 0.92 * prev_res64 + 0.08 * cur_res_t.clamp_max(10.0)
            reg._res_ema_t = res_ema64

        comp_rep: Optional[torch.Tensor] = None
        cid_i: Optional[torch.Tensor] = None
        cnt_f: Optional[torch.Tensor] = None
        sum_c: Optional[torch.Tensor] = None
        xi_i: Optional[torch.Tensor] = None
        xi_f_sum: Optional[torch.Tensor] = None
        need_vpush_reuse = (
            has_coal_flag and has_sigma and recipe.t1_iso_weight > 0.0 and not is_point_flag
            and dim == 2 and mu_det.shape[1] == 2 and reg.v_push is not None
        )
        v_mu_star_all = v_mu_star_cand if (has_coal_flag and any_v_mu_star_flag and v_mu_star_cand is not None) else None
        condense_part_a = (
            v_mu_star_all is not None and any_coal_i_flag and scale_ge_065_flag
            and can_condense_table and not is_point_flag and indices is not None
        )
        if any_coal_i_flag or need_vpush_reuse or (v_mu_star_all is not None) or (has_sigma and not is_point_flag):
            comp_rep = (w_mat_all > 0.0).float().argmax(dim=1)
            cid_i = comp_rep[target_p_G]
            if any_coal_i_flag:
                if condense_part_a:
                    assert indices is not None
                    xi_i = _deterministic_box_muller(indices, dim, torch.float32).to(yr.device)
                    packed_cid = torch.cat((x_ctr_det.new_ones((b, 1)), x_ctr_det, xi_i), dim=1)
                    sums_cid = x_ctr_det.new_zeros((k_proto, 1 + 2 * dim))
                    sums_cid.index_add_(0, cid_i, packed_cid)
                    cnt_f, sum_c, xi_f_sum = sums_cid[:, 0], sums_cid[:, 1 : 1 + dim], sums_cid[:, 1 + dim :]
                else:
                    packed_cid = torch.cat((x_ctr_det.new_ones((b, 1)), x_ctr_det), dim=1)
                    sums_cid = x_ctr_det.new_zeros((k_proto, 1 + dim))
                    sums_cid.index_add_(0, cid_i, packed_cid)
                    cnt_f, sum_c = sums_cid[:, 0], sums_cid[:, 1:]
            elif need_vpush_reuse:
                cnt_f = (cid_i.unsqueeze(0) == idx_k.unsqueeze(1)).float().sum(dim=1)

        if condense_part_a:
            assert v_mu_star_all is not None and cid_i is not None and cnt_f is not None and xi_i is not None and xi_f_sum is not None
            need_conc_i = multicell_coal[target_p_G] & (v_mu_star_all[target_p_G].amax(dim=-1) > 0.0)
            d_near_i = (xff - c_bar_all[target_p_G]).norm(dim=-1)
            halo_r = 3.5 * math.sqrt(float(dim)) * sigma_bar_all[target_p_G]
            halo_blend = ((d_near_i / halo_r.clamp_min(1e-12)).clamp(0.0, 1.0) ** 2).unsqueeze(1).to(fake_flat.dtype)
            xi_f_mean = torch.where(cnt_f.unsqueeze(1) >= 2.0, xi_f_sum / cnt_f.clamp_min(1.0).unsqueeze(1), torch.zeros_like(xi_f_sum))
            c_exact_i = (c_bar_all[target_p_G] + v_mu_star_all[target_p_G].sqrt() * (xi_i - xi_f_mean[cid_i])).to(fake_flat.dtype)
            y_ot = torch.where(need_conc_i.unsqueeze(1), halo_blend * y_ot + (1.0 - halo_blend) * c_exact_i, y_ot)

        # Law of Total Variance + 0.995 Quartic-Frobenius Jensen Centering (only when intentional
        # center dispersion v_mu_star_all is active or non-ledger isotropic coalescence is used;
        # when the ledger condenses x_ctr -> comp_c, full raw_cov_i must be produced by dx).
        raw_sig_i = sigma_bar_all[target_p_G]
        raw_cov_i = cov_bar_all[target_p_G]
        shrink_i = torch.ones(b, device=yr.device, dtype=torch.float32)
        if any_coal_i_flag and (not use_ledger or v_mu_star_all is not None):
            assert cid_i is not None and cnt_f is not None and sum_c is not None
            mean_c = sum_c / cnt_f.clamp_min(1.0).unsqueeze(1)
            n_i = cnt_f[cid_i]
            valid_idx = torch.nonzero(is_coal_i & (n_i >= 2.0), as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                n_v = n_i.index_select(0, valid_idx)
                x_v = x_ctr_det.index_select(0, valid_idx)
                c_v = cid_i.index_select(0, valid_idx)
                sig_v = raw_sig_i.index_select(0, valid_idx)
                v_mu_i = (n_v / (n_v - 1.0)) * ((x_v - mean_c[c_v]).square().sum(dim=-1) / float(dim))
                rho_mu = (v_mu_i / sig_v.square().clamp_min(1e-24)).clamp(0.0, 0.75).mean()
                shrink = torch.sqrt((0.995 - rho_mu).clamp_min(0.25))
                shrink_i = torch.where(is_coal_i, shrink.expand(b), shrink_i)
        eff_raw_sig_i = raw_sig_i * shrink_i
        cov_eff_i = raw_cov_i * shrink_i.square().view(b, 1, 1)
        sigma_eff_i = eff_raw_sig_i.to(fake_flat.dtype)

        if use_ledger and ledger_comp_w is not None:
            m_comp_f = float(len(ledger_comp_w))
            inv_w_i = (1.0 / (m_comp_f * ledger_comp_w[batch_comp]).clamp_min(1e-4)).clamp(0.25, 6.0).to(fake_flat.dtype)
        else:
            inv_w_i = ((1.0 / (float(k_proto) * reg.proto_w[target_p_G]).clamp_min(1e-4)).sqrt().clamp(0.5, 4.0).to(fake_flat.dtype))

        w_sat_vec = torch.ones(b, device=yr.device, dtype=fake_flat.dtype)
        if has_sigma and eps_g is not None:
            e_norm_raw = eps_g.float().norm(dim=-1)
            valid_e = e_norm_raw >= 0.45
            s_obs = (fake_det - x_ctr_det).norm(dim=-1) / (sigma_z * e_norm_raw.clamp_min(0.45))
            if has_iso_flag:
                n_valid_e = valid_e.sum()
                s_sorted = torch.where(valid_e, s_obs, s_obs.new_full((), float("inf"))).sort().values
                s_med = s_sorted.index_select(0, ((n_valid_e - 1).clamp_min(0) // 2).unsqueeze(0)).squeeze(0).clamp_min(1e-12)
                w_boost = (s_med / s_obs.clamp_min(1e-12)).clamp(1.0, 8.0)
                w_sat_vec = torch.where(valid_e & (s_obs < (0.50 * s_med)), w_boost, torch.ones_like(w_boost)).to(fake_flat.dtype)
            else:
                d_z_f = float(max(1, mu_det.shape[1]))
                s_target = (eff_raw_sig_i * math.sqrt(float(dim) / d_z_f)) / max(sigma_z, 1e-4)
                drift_d = (xff - y_ot.detach().float()).norm(dim=-1)
                is_rail = valid_e & (drift_d > (1.35 * math.sqrt(float(dim)) * raw_sig_i)) & (s_obs < (0.40 * s_target))
                w_boost = (0.45 * s_target / s_obs.clamp_min(1e-12)).clamp(1.0, 8.0)
                w_sat_vec = torch.where(is_rail, w_boost, torch.ones_like(w_boost)).to(fake_flat.dtype)

        y_bound = yf.abs().amax(dim=0, keepdim=True) + 3.5 * sigma_cell_med
        ot_scale_dt = (y_abs_ref if is_point_flag else r_macro_safe).to(fake_flat.dtype)

    sc_med_dt = sigma_cell_med.to(fake_flat.dtype)
    if x_ctr_all_grad is not None and ledger_assigned is not None and ledger_comp_c is not None and ledger_comp_sig is not None and ledger_comp_w is not None:
        # Pin ALL P isolated-component centers simultaneously -> zero B < P whiplash on unvisited modes.
        with torch.no_grad():
            m_comp_f = float(len(ledger_comp_w))
            inv_w_all = (1.0 / (m_comp_f * ledger_comp_w[ledger_assigned]).clamp_min(1e-4)).clamp(0.25, 6.0).to(fake_flat.dtype)
            sat_all = torch.ones(len(ledger_assigned), device=yr.device, dtype=fake_flat.dtype)
            if indices is not None:
                sat_all.index_copy_(0, indices.to(device=yr.device, dtype=torch.long), w_sat_vec)
            y_ot_all = ledger_comp_c[ledger_assigned].to(fake_flat.dtype)
            sc_ot_all = ledger_comp_sig[ledger_assigned].to(fake_flat.dtype)
        sq_ot_all = (x_ctr_all_grad - y_ot_all).square().sum(dim=-1)
        loss_g_ot = (sat_all * inv_w_all * (torch.sqrt(sc_ot_all.square() + sq_ot_all) - sc_ot_all) / ot_scale_dt).mean()
    else:
        x_ot_src = x_ctr_flat if all_coal_flag else (fake_flat if not any_coal_i_flag else torch.where(is_coal_i.unsqueeze(1), x_ctr_flat, fake_flat))
        sq_ot = (x_ot_src - y_ot).square().sum(dim=-1)
        sc_ot_dt = (ledger_comp_sig[batch_comp].to(fake_flat.dtype) if (use_ledger and ledger_comp_sig is not None) else sc_med_dt)
        ot_row_w = (w_sat_vec * inv_w_i) if use_ledger else w_sat_vec
        loss_g_ot = (ot_row_w * (torch.sqrt(sc_ot_dt.square() + sq_ot) - sc_ot_dt) / ot_scale_dt).mean()
    excess_bound = F.relu(fake_flat.float().abs() - y_bound) / ot_scale_dt.float()
    loss_g_ot = loss_g_ot + (4.0 * (torch.sqrt(1.0 + excess_bound.square()) - 1.0).mean()).to(fake_flat.dtype)

    if has_sigma and recipe.t1_iso_weight > 0.0:
        if is_point_flag:
            loss_g_iso = ((fake_flat - x_ctr_flat) / ot_scale_dt).square().sum(dim=-1).mean()
        else:
            d_z = int(mu_det.shape[1])
            prev_iso_err64 = reg._get_ema_t("iso_err_ema", yr.device)
            if dim == 2 and d_z == 2:
                m_iso = b if has_iso_flag else min(b, 128)
                e1_u = eps_g[:m_iso] / eps_g[:m_iso].norm(dim=-1, keepdim=True).clamp_min(1e-6)
                e2_u = torch.stack((-e1_u[:, 1], e1_u[:, 0]), dim=1)
                z_u_cat = torch.cat([mu_det[:m_iso] + sigma_z * e1_u, mu_det[:m_iso] + sigma_z * e2_u], dim=0)
                x_u1, x_u2 = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_u_cat, 0.0, noise)).reshape(2 * m_iso, -1).chunk(2, dim=0)
                dx_0 = x_u1 - x_ctr_flat[:m_iso]
                dx_p = x_u2 - x_ctr_flat[:m_iso]
                sc_sub2 = sigma_eff_i[:m_iso]
                dx_0_n = dx_0.norm(dim=-1)
                dx_p_n = dx_p.norm(dim=-1)
                with torch.no_grad():
                    min_dx_u = torch.minimum(dx_0_n.detach(), dx_p_n.detach())
                    w_iso_sat = (sc_sub2 / min_dx_u.clamp_min(0.15 * sc_sub2)).clamp(1.0, 4.0)
                    lam1_2, lam2_2, _, _ = _sym_eig2x2_exact(cov_eff_i[:m_iso])
                    is_aniso_2d = bool(((lam1_2 / lam2_2).amax() > 1.8).item())
                w_iso_sub = inv_w_i[:m_iso] * w_iso_sat
                M_dx2 = torch.stack((dx_0, dx_p), dim=-1)
                Sigma_push2 = M_dx2 @ M_dx2.transpose(1, 2)
                eye2 = reg._get_eye2(yr.device, fake_flat.dtype)
                target_cov2 = cov_eff_i[:m_iso].to(fake_flat.dtype) if is_aniso_2d else ((sc_sub2.square().view(m_iso, 1, 1)) * eye2)
                denom_cov2 = (sc_sub2 * sc_med_dt).clamp_min(1e-24).view(m_iso, 1, 1)

                if has_iso_flag:
                    with torch.no_grad():
                        if reg.v_push is not None and has_coal_flag:
                            assert comp_rep is not None and cid_i is not None and cnt_f is not None
                            vp_diag = dx_0.detach().float().square() + dx_p.detach().float().square()
                            sum_vp = vp_diag.new_zeros((k_proto, dim))
                            sum_vp.index_add_(0, cid_i, vp_diag)
                            cnt_k = cnt_f[comp_rep]
                            hit_k = (cnt_k > 0) & multicell_coal
                            mean_vp_k = sum_vp[comp_rep] / cnt_k.clamp_min(1.0).unsqueeze(1)
                            reg.v_push = torch.where(hit_k.unsqueeze(1), 0.88 * reg.v_push + 0.12 * mean_vp_k, reg.v_push)
                        t_tr1 = sc_sub2.detach().float().square().clamp_min(1e-24).view(m_iso, 1, 1)
                        iso_frob_i = ((Sigma_push2.detach().float() - target_cov2.detach().float()) / t_tr1).square().sum(dim=(1, 2)).sqrt()
                        cur_frob64 = 0.60 * iso_frob_i.mean().double() + 0.40 * _quantile_1d_fast(iso_frob_i, 0.92).double()
                        reg._iso_err_ema_t = 0.92 * prev_iso_err64 + 0.08 * cur_frob64.clamp_max(5.0)
                    err_cov2 = (Sigma_push2 - target_cov2) / denom_cov2
                    iso_cov2 = (w_iso_sub * (4.0 * torch.asinh(0.5 * err_cov2).square()).sum(dim=(1, 2))).mean() * 0.5
                    if is_aniso_2d:
                        loss_g_iso = iso_cov2
                    else:
                        iso_u1 = (w_iso_sub * (4.0 * torch.asinh(0.5 * (dx_0_n - sc_sub2) / sc_med_dt).square())).mean()
                        iso_u2 = (w_iso_sub * (4.0 * torch.asinh(0.5 * (dx_p_n - sc_sub2) / sc_med_dt).square())).mean()
                        loss_g_iso = 0.45 * (iso_u1 + iso_u2) + 0.55 * iso_cov2
                else:
                    with torch.no_grad():
                        eps_norm = eps_g.norm(dim=-1).clamp_min(0.25)
                        rel_iso_err = (((fake_det - x_ctr_det).norm(dim=-1) - eff_raw_sig_i * eps_norm) / sigma_cell_med.clamp_min(1e-12)).abs()
                        reg._iso_err_ema_t = 0.94 * prev_iso_err64 + 0.06 * rel_iso_err.mean().clamp_max(5.0).double()
                    if is_aniso_2d:
                        err_cov2 = (Sigma_push2 - target_cov2) / denom_cov2
                        loss_g_iso = (w_iso_sub * (4.0 * torch.asinh(0.5 * err_cov2).square()).sum(dim=(1, 2))).mean() * 0.5
                    else:
                        iso_u1 = (w_iso_sub * (4.0 * torch.asinh(0.5 * (dx_0_n - sc_sub2) / sc_med_dt).square())).mean()
                        iso_u2 = (w_iso_sub * (4.0 * torch.asinh(0.5 * (dx_p_n - sc_sub2) / sc_med_dt).square())).mean()
                        cos_sq = ((dx_0 * dx_p).sum(dim=-1) / (dx_0_n * dx_p_n + (1e-6 * sc_sub2.square()).clamp_min(1e-24))).square().mean()
                        loss_g_iso = 0.5 * (iso_u1 + iso_u2) + 0.25 * cos_sq
            else:
                # General (dim, d_z): zero nullspace bias, zero eigh, zero d_z==4 epicycle!
                # Tracks per-component pushforward covariance reg.push_cov[K, D, D] and applies
                # exact Riemannian log-covariance straight-through pullback (sigma_target cancels
                # identically in d/d(dx) -> perfectly conditioned across 10:1 ellipses & 1e-5x scale).
                use_qmc_iso = (
                    x_ctr_all_grad is not None
                    and ledger_assigned is not None
                    and ledger_uniq_reps is not None
                    and ledger_comp_cov is not None
                    and ledger_comp_w is not None
                    and b < len(base_prior.z)
                )
                if use_qmc_iso:
                    assert x_ctr_all_grad is not None and ledger_assigned is not None and ledger_uniq_reps is not None and ledger_comp_cov is not None and ledger_comp_w is not None
                    p_iso = len(base_prior.z)
                    c_all_iso = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z).detach()
                    qmc_rows = torch.arange(p_iso, device=yr.device, dtype=torch.float64) + float(reg.phase) * 997.0
                    reg.phase = (reg.phase + _PHI1) % 1.0
                    eps_iso = (_deterministic_box_muller(qmc_rows, d_z, mu_det.dtype).to(yr.device) * 1.0066)
                    fake_iso = trainer._postprocess(
                        trainer.prior, trainer._generate(trainer.G, c_all_iso + sigma_z * eps_iso, 0.0, noise)
                    ).reshape(p_iso, -1)
                    dx_ker = fake_iso - x_ctr_all_grad
                    iso_target_p = ledger_uniq_reps[ledger_assigned]
                    iso_cid = comp_rep[iso_target_p]
                    iso_cov_eff = ledger_comp_cov[ledger_assigned]
                    m_cf = float(len(ledger_comp_w))
                    iso_inv_w = (1.0 / (m_cf * ledger_comp_w[ledger_assigned]).clamp_min(1e-4)).clamp(0.25, 6.0).to(dx_ker.dtype)
                    n_iso = p_iso
                else:
                    dx_ker = fake_flat - x_ctr_flat
                    iso_target_p = target_p_G
                    iso_cid = cid_i
                    iso_cov_eff = cov_eff_i
                    iso_inv_w = inv_w_i
                    n_iso = b

                with torch.no_grad():
                    assert reg.push_cov is not None and comp_rep is not None and iso_cid is not None
                    dx_det = dx_ker.detach().float()
                    dx_outer = (dx_det.unsqueeze(2) * dx_det.unsqueeze(1)).reshape(n_iso, dim * dim)
                    eye_d_p = torch.eye(dim, device=yr.device, dtype=torch.float32)
                    # Exclude particles currently in flight across modes from polluting settled mode push_cov/loss_g_iso.
                    if use_qmc_iso and x_ctr_all_grad is not None and ledger_assigned is not None and ledger_comp_c is not None:
                        d_to_c = (x_ctr_all_grad.detach().float() - ledger_comp_c[ledger_assigned]).norm(dim=-1)
                        in_mode_m = d_to_c <= (3.5 * math.sqrt(float(dim)) * sigma_bar_all[iso_target_p])
                        in_cnt = dx_det.new_zeros((k_proto,))
                        in_cnt.index_add_(0, iso_cid, in_mode_m.float())
                        use_in_mask = in_cnt[iso_cid] >= 2.0
                        iso_valid_w = torch.where(use_in_mask, in_mode_m.float(), dx_det.new_ones((n_iso,)))
                    else:
                        iso_valid_w = dx_det.new_ones((n_iso,))
                    p_sums = dx_det.new_zeros((k_proto, 1 + dim + dim * dim))
                    p_sums.index_add_(
                        0,
                        iso_cid,
                        torch.cat(
                            (iso_valid_w.unsqueeze(1), iso_valid_w.unsqueeze(1) * dx_det, iso_valid_w.unsqueeze(1) * dx_outer),
                            dim=1,
                        ),
                    )
                    cnt_p = p_sums[comp_rep, 0]
                    hit_p = cnt_p > 0.0
                    mean_dx_k = p_sums[comp_rep, 1 : 1 + dim] / cnt_p.clamp_min(1.0).unsqueeze(1)
                    raw_sec_k = p_sums[comp_rep, 1 + dim :].view(k_proto, dim, dim) / cnt_p.clamp_min(1.0).view(k_proto, 1, 1)
                    # Unbiased centered sample covariance when a component has >= 3 samples (eliminates
                    # generator curvature mean-shift bias E[dx]E[dx]^T and finite-sample Bessel bias).
                    bessel_scale = torch.where(cnt_p >= 3.0, cnt_p / (cnt_p - 1.0).clamp_min(1.0), cnt_p.new_ones((k_proto,)))
                    cov_cen_k = bessel_scale.view(k_proto, 1, 1) * (
                        raw_sec_k - mean_dx_k.unsqueeze(2) * mean_dx_k.unsqueeze(1)
                    )
                    mean_outer_k = torch.where((cnt_p >= 3.0).view(k_proto, 1, 1), cov_cen_k, raw_sec_k)
                    mean_outer_k = 0.5 * (mean_outer_k + mean_outer_k.transpose(-2, -1))
                    var_fl_p = (1e-10 * r_macro_safe.square()).clamp_min(1e-24)
                    mean_outer_k = mean_outer_k + var_fl_p * eye_d_p.unsqueeze(0)

                    alpha_base_p = (0.18 * cnt_p / (cnt_p + 2.0)).clamp_max(0.25) if use_qmc_iso else (0.08 * cnt_p / (cnt_p + 0.5)).clamp_max(0.20)
                    tr_push = torch.diagonal(reg.push_cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-24)
                    tr_obs = torch.diagonal(mean_outer_k, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-24)
                    ratio_tr = torch.maximum(tr_push / tr_obs, tr_obs / tr_push)
                    shock_p = ((torch.log(ratio_tr.clamp_min(1.0)) - 2.0) / 3.0).clamp(0.0, 1.0)
                    alpha_p = (alpha_base_p + (0.35 - alpha_base_p) * shock_p).view(k_proto, 1, 1)
                    reg.push_cov = torch.where(
                        hit_p.view(k_proto, 1, 1),
                        (1.0 - alpha_p) * reg.push_cov + alpha_p * mean_outer_k,
                        reg.push_cov,
                    )
                    s_bar_all = torch.einsum("ij,jde->ide", w_mat_all, reg.push_cov)
                    s_bar_all = 0.5 * (s_bar_all + s_bar_all.transpose(-2, -1))
                    s_i = s_bar_all[iso_target_p]
                    iso_row_w = iso_inv_w * iso_valid_w.to(dx_ker.dtype)

                if dim == 2:
                    with torch.no_grad():
                        lam1, lam2, c_th, s_th = _sym_eig2x2_exact(iso_cov_eff)
                        inv_s1 = lam1.rsqrt()
                        inv_s2 = lam2.rsqrt()
                        s00, s01, s11 = s_i[:, 0, 0], s_i[:, 0, 1], s_i[:, 1, 1]
                        # Whiten S_i into target eigenframe: U_i = W_i S_i W_i^T
                        u00 = (c_th * c_th * s00 + 2.0 * c_th * s_th * s01 + s_th * s_th * s11) / lam1
                        u11 = (s_th * s_th * s00 - 2.0 * c_th * s_th * s01 + c_th * c_th * s11) / lam2
                        u01 = (-c_th * s_th * s00 + (c_th * c_th - s_th * s_th) * s01 + c_th * s_th * s11) * (inv_s1 * inv_s2)
                        u_mat = torch.stack((torch.stack((u00, u01), dim=-1), torch.stack((u01, u11), dim=-1)), dim=-2)
                        mu1, mu2, cu, su = _sym_eig2x2_exact(u_mat)
                        lm1 = torch.log(mu1.clamp(1e-3, 1e3)).clamp(-2.5, 2.5)
                        lm2 = torch.log(mu2.clamp(1e-3, 1e3)).clamp(-2.5, 2.5)
                        # Exact Riemannian Newton step: m_eff_j = log(mu_j) / max(mu_j, 0.05)
                        me1 = lm1 / mu1.clamp_min(0.05)
                        me2 = lm2 / mu2.clamp_min(0.05)
                        m00 = (me1 * cu * cu + me2 * su * su).to(dx_ker.dtype)
                        m11 = (me1 * su * su + me2 * cu * cu).to(dx_ker.dtype)
                        m01 = ((me1 - me2) * cu * su).to(dx_ker.dtype)
                        riem_val = (0.5 * (lm1.square() + lm2.square())).to(dx_ker.dtype)
                        cur_riem64 = torch.sqrt(lm1.square() + lm2.square()).mean().clamp_max(5.0).double()
                        reg._iso_err_ema_t = 0.94 * prev_iso_err64 + 0.06 * cur_riem64
                        c_dt, s_dt = c_th.to(dx_ker.dtype), s_th.to(dx_ker.dtype)
                        is1_dt, is2_dt = inv_s1.to(dx_ker.dtype), inv_s2.to(dx_ker.dtype)
                    u0 = (c_dt * dx_ker[:, 0] + s_dt * dx_ker[:, 1]) * is1_dt
                    u1 = (-s_dt * dx_ker[:, 0] + c_dt * dx_ker[:, 1]) * is2_dt
                    quad_i = 0.5 * (m00 * u0.square() + 2.0 * m01 * u0 * u1 + m11 * u1.square())
                    loss_g_iso = (iso_row_w * (riem_val + (quad_i - quad_i.detach()))).mean()
                else:
                    with torch.no_grad():
                        t_diag = torch.diagonal(iso_cov_eff, dim1=-2, dim2=-1).clamp_min(1e-24)
                        s_diag = torch.diagonal(s_i, dim1=-2, dim2=-1).clamp_min(1e-24)
                        mu_d = s_diag / t_diag
                        lm_d = torch.log(mu_d.clamp(1e-3, 1e3)).clamp(-2.5, 2.5)
                        me_d = (lm_d / mu_d.clamp_min(0.05)).to(dx_ker.dtype)
                        inv_sd = t_diag.rsqrt().to(dx_ker.dtype)
                        riem_val = (0.5 * lm_d.square().mean(dim=-1)).to(dx_ker.dtype)
                        cur_riem64 = lm_d.square().mean(dim=-1).sqrt().mean().clamp_max(5.0).double()
                        reg._iso_err_ema_t = 0.94 * prev_iso_err64 + 0.06 * cur_riem64
                    u_d = dx_ker * inv_sd
                    quad_i = 0.5 * (me_d * u_d.square()).mean(dim=-1)
                    loss_g_iso = (iso_row_w * (riem_val + (quad_i - quad_i.detach()))).mean()
    else:
        loss_g_iso = fake_flat.new_zeros(())

    transport = recipe.kinetic_transport_loss(fake_g, real_g) if recipe.kinetic_transport_weight else loss_gan.new_zeros(())
    transport_local = (
        recipe.kinetic_transport_local_loss(fake_g, real_g) if recipe.kinetic_transport_local_weight else loss_gan.new_zeros(())
    )

    with torch.no_grad():
        if has_iso_flag:
            eq_gate64 = ((0.030 - res_ema64) / 0.020).clamp(0.0, 1.0)
            adv_w_dt = (0.038 * (1.0 - eq_gate64) + 0.008 * eq_gate64).to(loss_gan.dtype)
            g_floor64 = 0.022 * (1.0 - eq_gate64) + 0.012 * eq_gate64
            row_floor64 = 0.15 * (1.0 - eq_gate64) + 0.38 * eq_gate64
        else:
            adv_w_dt = 0.15
            g_floor64 = res_ema64.new_full((), 0.04)
            row_floor64 = res_ema64.new_full((), 0.15)

    loss_g = adv_w_dt * loss_gan + float(recipe.t1_pullback_weight) * loss_g_ot + float(recipe.t1_iso_weight) * loss_g_iso
    if base_prior.z.requires_grad and recipe.prior_l2 > 0:
        loss_g = loss_g + recipe.prior_l2 * base_prior.z.square().mean()

    prior_manip_loss = None
    if trainer._prior_penalty:
        raw_rows = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        prior_manip_loss = trainer.prior.regularize(
            raw_rows, sampled_codes=latent, indices=indices, generated=fake_g, logits=fake_logits
        )
        loss_g = loss_g + prior_manip_loss
    if recipe.kinetic_transport_weight:
        loss_g = loss_g + transport
    if recipe.kinetic_transport_local_weight:
        loss_g = loss_g + transport_local

    if rec_d is not None and hasattr(rec_d, "_t1c_disc_gate"):
        rec_d._t1c_disc_gate = 0.0

    trainer.opt_g.zero_grad()
    protected_evaluator = None
    if recipe.constraint_geometry_mode == "strict_progress":
        def protected_evaluator() -> Tuple[torch.Tensor]:
            rec = getattr(trainer.opt_d, "record", None)
            prev_probe = getattr(rec, "_in_protected_probe", False) if rec is not None else False
            if rec is not None:
                rec._in_protected_probe = True
            try:
                generated = trainer._postprocess(
                    trainer.prior, trainer._generate(trainer.G, latent_det if has_sigma else mu_det, sigma_out, noise, rows=indices)
                )
                return (F.softplus(-critic(generated)).mean(),)
            finally:
                if rec is not None:
                    rec._in_protected_probe = prev_probe

    if getattr(trainer.opt_g, "direction_blend", False) and hasattr(trainer.opt_g, "bind_protected_from_grads"):
        loss_g.backward()
        trainer.opt_g.bind_protected_from_grads()
    else:
        constraint_geometry_backward(loss_g, trainer.opt_g, (loss_g,), protected_evaluator=protected_evaluator)
    if base_prior.z.requires_grad and base_prior.z.grad is None:
        base_prior.z.grad = torch.zeros_like(base_prior.z)

    trainer.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())

    with torch.no_grad():
        if is_point_flag:
            g_scale_t = res_ema64.new_full((), 1.0)
        else:
            iso_err64 = reg._get_ema_t("iso_err_ema", yr.device)
            iso_gate64 = ((iso_err64 - 0.18) / 0.45).clamp(0.0, 1.0)
            g_scale_norm64 = torch.maximum(
                g_floor64,
                torch.minimum(res_ema64.new_full((), 1.0), torch.maximum(res_ema64 / 0.10, 0.35 * iso_gate64)),
            )
            g_scale_t = torch.where(scale_ratio_t < 0.60, res_ema64.new_full((), 1.0), g_scale_norm64)

    has_v4_mod = hasattr(trainer.opt_g, "bind_v4_modulation")
    pre_u_rows: Optional[torch.Tensor] = None
    if (
        can_condense_table and base_prior.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0 and not is_point_flag
        and has_v4_mod and hasattr(trainer.opt_g, "_supports_zerosync_v4_tensors")
        and not trainer.opt_g._supports_zerosync_v4_tensors()
    ):
        cand_rows = getattr(trainer.opt_g, "_sampled_rows", {}).get(base_prior.z)
        if cand_rows is not None and cand_rows.device == indices.device:
            pre_u_rows = cand_rows

    if has_v4_mod:
        trainer.opt_g.bind_v4_modulation(g_scale=g_scale_t, wakeup_boost=wakeup_boost_t, prior_scale=1.0)
        if trainer._gradient_response is not None:
            params = list(trainer.G.parameters())
            gain = trainer._gradient_response.observe([torch.zeros_like(p) if p.grad is None else p.grad for p in params])
            for p, m in zip(params, trainer._gradient_response.packet["means"]):
                p.grad = (m / trainer._gradient_response.packet["weight"]).to(p).clone()
            trainer.opt_g.bind_v4_modulation(g_scale=g_scale_t * gain, wakeup_boost=wakeup_boost_t, prior_scale=1.0)
            trainer.opt_g.step()
            trainer._gradient_response.response["applied_steps"] += 1
        else:
            trainer.opt_g.step()
    else:
        g_scale = float(g_scale_t.item())
        saved_lrs = [grp["lr"] for grp in trainer.opt_g.param_groups]
        for grp in trainer.opt_g.param_groups:
            grp["lr"] = grp["lr"] * g_scale
        try:
            if trainer._gradient_response is not None:
                params = list(trainer.G.parameters())
                gain = trainer._gradient_response.observe([torch.zeros_like(p) if p.grad is None else p.grad for p in params])
                for p, m in zip(params, trainer._gradient_response.packet["means"]):
                    p.grad = (m / trainer._gradient_response.packet["weight"]).to(p).clone()
                for group in trainer.opt_g.param_groups:
                    if group.get("role") == "generator":
                        group["lr"] = group["lr"] * gain
                trainer.opt_g.step()
                trainer._gradient_response.response["applied_steps"] += 1
            else:
                trainer.opt_g.step()
        finally:
            for grp, lr0 in zip(trainer.opt_g.param_groups, saved_lrs):
                grp["lr"] = lr0

    trainer.policy.after_generator_step()

    # --- PART B: Starvation-Free Kantorovich-Laguerre Condensation + Outward Dead-Clamp Rescue ---
    prior_reg_diag = loss_gan.new_zeros(())
    if can_condense_table and base_prior.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0 and not is_point_flag:
        if full_table_ledger and ledger_assigned is not None:
            u_rows = torch.arange(len(base_prior.z), device=yr.device, dtype=torch.long)
        else:
            u_rows = pre_u_rows if pre_u_rows is not None else torch.unique(indices)
        g_flags = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)

            n_u = len(u_rows)
            n_eff = float(n_u)
            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[u_rows]
            x_u = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, c_u, 0.0, noise)).reshape(n_u, -1)

            with torch.no_grad():
                xu_det = x_u.detach().float()
                if use_ledger and ledger_assigned is not None and ledger_uniq_reps is not None and ledger_comp_c is not None and ledger_comp_sig is not None:
                    u_dev = u_rows.to(device=yr.device, dtype=torch.long)
                    u_comp = ledger_assigned[u_dev]
                    k_star = ledger_uniq_reps[u_comp]
                    c_bar_u = ledger_comp_c[u_comp]
                    sc_u = ledger_comp_sig[u_comp].to(x_u.dtype)
                    psi_gate64 = res_ema64.new_full((), min(1.0, max(0.0, ledger_unsettled)))
                else:
                    psi_eff = w_mat_all @ reg.psi
                    d2_up = torch.cdist(xu_det, c_bar_all, p=2.0).square()
                    k_star = (d2_up - psi_eff.unsqueeze(0)).argmin(dim=1)

                    w_hard_u = w_mat_all[k_star].mean(dim=0)
                    reg.w_fake_ema = 0.88 * reg.w_fake_ema + 0.12 * w_hard_u
                    reg.w_fake_ema = reg.w_fake_ema / reg.w_fake_ema.sum().clamp_min(1e-12)
                    w_feedback = 0.35 * w_hard_u + 0.65 * reg.w_fake_ema

                    k_float = float(k_proto)
                    tv_now64 = 0.5 * (reg.proto_w - reg.w_fake_ema).abs().sum().double()
                    tv_floor = 0.35 * math.sqrt(k_float / max(n_eff * 8.0, 1.0))
                    tv_gate64 = ((tv_now64 - tv_floor) / max(0.08, tv_floor)).clamp(0.0, 1.0)

                    # Component-level starvation override: never freeze psi_gate while any mode/atom
                    # expecting >= 0.75 particles is starved below 65% of its target mass.
                    comp_mask = (w_mat_all > 0.0).float()
                    comp_w = comp_mask @ reg.proto_w
                    comp_wf = comp_mask @ reg.w_fake_ema
                    starv_rel = F.relu(comp_w - comp_wf) / comp_w.clamp_min(1e-6)
                    starv_elig = (comp_w * float(len(base_prior.z))) >= 0.75
                    max_starv64 = torch.where(starv_elig, starv_rel, starv_rel.new_zeros(())).amax().double()
                    starv_gate64 = ((max_starv64 - 0.35) / 0.35).clamp(0.0, 1.0)
                    psi_gate64 = torch.maximum(tv_gate64, starv_gate64)

                    eta_scale = min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * k_float))
                    r_mac_sq64 = r_macro_safe.square().double()
                    eta_psi64 = ((psi_gate64 * eta_scale) * float(dim)) * r_mac_sq64
                    psi_next = reg.psi * (1.0 - 0.008 * psi_gate64).float()
                    psi_next = torch.addcmul(psi_next, reg.proto_w - w_feedback, eta_psi64.float())
                    psi_next = w_mat_all @ psi_next
                    psi_next = psi_next - psi_next.mean()
                    psi_cap_t = (2.5 * float(dim) * r_mac_sq64).clamp_min(1e-12).float()
                    psi_clamped = torch.clamp(psi_next, -psi_cap_t, psi_cap_t)
                    reg.psi = psi_clamped - psi_clamped.mean()

                    c_bar_u = c_bar_all[k_star]
                    sc_u = sigma_bar_all[k_star].to(x_u.dtype)

                if v_mu_star_all is not None:
                    assert comp_rep is not None
                    coal_u = multicell_coal[k_star]
                    xi_u = _deterministic_box_muller(u_rows, dim, torch.float32).to(yr.device)
                    cid_u = comp_rep[k_star]
                    packed_u = torch.cat((xi_u.new_ones((n_u, 1)), xi_u), dim=1)
                    sums_u = xi_u.new_zeros((k_proto, 1 + dim))
                    sums_u.index_add_(0, cid_u, packed_u)
                    cnt_u, xi_u_sum = sums_u[:, 0], sums_u[:, 1:]
                    xi_u_mean = torch.where(cnt_u.unsqueeze(1) >= 2.0, xi_u_sum / cnt_u.clamp_min(1.0).unsqueeze(1), torch.zeros_like(xi_u_sum))
                    c_conc_u = c_bar_u + v_mu_star_all[k_star].sqrt() * (xi_u - xi_u_mean[cid_u])
                    c_target = torch.where(coal_u.unsqueeze(1), c_conc_u, c_bar_u).to(x_u.dtype)
                else:
                    c_target = c_bar_u.to(x_u.dtype)
                row_settle64 = torch.maximum(
                    row_floor64,
                    torch.minimum(res_ema64.new_full((), 1.0), torch.maximum(res_ema64 / 0.10, psi_gate64)),
                )

            sq_u = (x_u - c_target).square().sum(dim=-1)
            loss_mu_condense = ((torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u) / ot_scale_dt.to(x_u.dtype)).mean()

            mu_raw_u = base_prior.z[u_rows]
            if has_sigma and n_u >= 2:
                mu_c = mu_raw_u - mu_raw_u.mean(dim=0, keepdim=True)
                mu_std = (mu_c.square().mean(dim=0) + 1e-6).sqrt()
                loss_mu_iso = F.relu(reg._get_ema_t("iso_ema", yr.device).to(mu_std.dtype) - mu_std).square().mean()
            else:
                loss_mu_iso = x_u.new_zeros(())

            loss_mu_total = loss_mu_condense + 0.15 * loss_mu_iso
            prior_reg_diag = loss_mu_total.detach()

            if base_prior.z.grad is not None:
                base_prior.z.grad.zero_()
            loss_mu_total.backward()

            if not reg.rows_initialized or reg.row_m is None or reg.row_m.shape != base_prior.z.shape:
                reg.row_m = torch.zeros_like(base_prior.z)
                reg.row_v = torch.zeros(len(base_prior.z), 1, device=base_prior.z.device, dtype=base_prior.z.dtype)
                reg.row_n = torch.zeros(len(base_prior.z), 1, device=base_prior.z.device, dtype=base_prior.z.dtype)
                reg.rows_initialized = True

            with torch.no_grad():
                g_rows = (base_prior.z.grad[u_rows] * n_eff).clone()
                base_prior.z.grad.zero_()
                dist_u = sq_u.detach().float().sqrt()
                part_scale = (dist_u.to(sq_u.dtype).unsqueeze(1) / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2)

                far_mask_1d = dist_u > (2.8 * math.sqrt(float(dim)) * sc_u.float())
                d_z_u = int(c_u.shape[1])
                if d_z_u <= 8 and bool(far_mask_1d.any()):
                    far_idx = torch.nonzero(far_mask_1d, as_tuple=False).squeeze(-1)
                    in_idx = torch.nonzero(~far_mask_1d, as_tuple=False).squeeze(-1)[:64]
                    if in_idx.numel() < 4:
                        in_idx = torch.arange(min(64, n_u), device=c_u.device, dtype=torch.long)
                    probe_idx = torch.cat([far_idx, in_idx], dim=0)
                    c_probe = c_u[probe_idx].detach()
                    x_probe = xu_det[probe_idx]
                    out_sgn = torch.where(c_probe >= 0.0, 1.0, -1.0)
                    eye_z = torch.eye(d_z_u, device=c_u.device, dtype=c_u.dtype)
                    c_pert = (c_probe.unsqueeze(1) + (1e-3 * out_sgn).unsqueeze(2) * eye_z.unsqueeze(0)).reshape(-1, d_z_u)
                    x_pert = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, c_pert, 0.0, noise)).float().reshape(len(probe_idx), d_z_u, dim)
                    col_sens = (x_pert - x_probe.unsqueeze(1)).norm(dim=-1) / 1e-3
                    s_ref_col = torch.quantile(col_sens[len(far_idx) :], 0.5, dim=0, keepdim=True).clamp_min(1e-12)
                    dead_far = col_sens[: len(far_idx)] < (0.04 * s_ref_col)
                    if bool(dead_far.any()):
                        mu_ref = c_u[in_idx].detach().mean(dim=0, keepdim=True)
                        z_far = mu_raw_u[far_idx].detach()
                        gu_far = g_rows[far_idx]
                        g_mag = torch.maximum(gu_far.float().norm(dim=-1, keepdim=True), torch.ones_like(z_far[:, :1], dtype=torch.float32))
                        inward_grad = (torch.sign(z_far - mu_ref).float() * g_mag).to(g_rows.dtype)
                        g_rows[far_idx] = torch.where(dead_far, inward_grad, gu_far)
                        part_scale[far_idx] = torch.where(dead_far.any(dim=1).unsqueeze(1), torch.ones_like(part_scale[far_idx]), part_scale[far_idx])

                m = reg.row_m[u_rows] * 0.80 + 0.20 * g_rows
                v = reg.row_v[u_rows] * 0.80 + 0.20 * g_rows.square().sum(dim=-1, keepdim=True)
                n = reg.row_n[u_rows] + 1.0
                reg.row_m[u_rows] = m
                reg.row_v[u_rows] = v
                reg.row_n[u_rows] = n
                bc = 1.0 - (0.80 ** n)
                step_vec = (m / bc) / ((v / bc).sqrt() + 1e-8)
                row_alpha = ((-float(recipe.t1_prior_row_lr)) * row_settle64).to(base_prior.z.dtype)
                base_prior.z.index_add_(0, u_rows, (step_vec * part_scale) * row_alpha, alpha=1.0)
        finally:
            for p, flag in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(flag)

    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


__all__ = [
    "UnifiedV15PriorRegularizer",
    "_compute_pythagorean_smoothed_cells_v15",
    "_sym_eig2x2_exact",
    "should_use_unified_v15_pullback",
    "unified_v15_pullback_step",
]
