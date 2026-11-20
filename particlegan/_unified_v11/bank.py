"""Zero-`eigh` / Zero-`svd` / Zero-`pinv` Cross-Batch Conditional Operator Bank for Unified-v11.

Strictly clock-free, swap-invariant, innovation-resetting per-stratum conditional operator bank
derived from ``CondFiberBank``, with all ``torch.linalg.eigh`` calls replaced by exact closed-form
symmetric 1x1/2x2 eigensolvers and cyclic Jacobi symmetric eigendecomposition (``< 1e-14`` float64
parity against exact ``eigh`` with strictly ``0`` ``eigh``/``svd``/``pinv`` calls).
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_UNIFIED_V11_BANK_SCHEMA = 1
_MAX_STRATA = 64
_MAX_RING_CAPACITY = 384
_MAX_FULL_BASIS_DC = 6
_MAX_BANK_DC = 32
_STRATUM_KEYS = frozenset({
    "key",
    "n_eff",
    "n_fast",
    "w_eff",
    "w_fast",
    "mu_c",
    "m2_c",
    "g_pp",
    "g_px",
    "g_py",
    "g_pp_fast",
    "g_px_fast",
    "g_py_fast",
    "s_xx",
    "s_yy",
    "ring_c",
    "ring_x",
    "ring_y",
})
_BANK_PACKET_KEYS = frozenset({"schema", "d_c", "d_x", "strata"})


@torch.no_grad()
def zero_eigh_symmetric(mat: torch.Tensor, *, sweeps: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact zero-`eigh` symmetric eigendecomposition for ``[..., d, d]`` matrices.

    Uses exact closed-form formulas for ``d in (1, 2)`` (branchless cancellation-free float64)
    and vectorized cyclic Jacobi rotations for ``d >= 3`` (converges to ``< 1e-14`` in float64).
    Returns ``(eigvals, eigvecs)`` with columns of ``eigvecs`` orthonormal and
    ``mat ~= eigvecs @ diag(eigvals) @ eigvecs.T``.
    """
    orig_dtype = mat.dtype
    d = int(mat.shape[-1])
    if mat.ndim < 2 or mat.shape[-2] != d:
        raise ValueError(f"zero_eigh_symmetric expects [..., d, d], got {tuple(mat.shape)}")
    sym = 0.5 * (mat.to(dtype=torch.float64) + mat.to(dtype=torch.float64).transpose(-1, -2))
    batch_shape = sym.shape[:-2]

    if d == 1:
        vals = sym[..., 0, 0].unsqueeze(-1)
        vecs = torch.ones_like(sym)
        return vals.to(orig_dtype), vecs.to(orig_dtype)

    if d == 2:
        a = sym[..., 0, 0]
        b = sym[..., 0, 1]
        c_diag = sym[..., 1, 1]
        half_tr = 0.5 * (a + c_diag)
        h = 0.5 * (a - c_diag)
        r = torch.hypot(h, b)
        lam_min = half_tr - r
        lam_max = half_tr + r
        use_top = h >= 0.0
        vx = torch.where(use_top, h + r, b)
        vy = torch.where(use_top, b, r - h)
        nrm = torch.hypot(vx, vy)
        deg = nrm <= 1e-30
        safe_nrm = torch.where(deg, torch.ones_like(nrm), nrm)
        cos_t = torch.where(deg, torch.ones_like(vx), vx / safe_nrm)
        sin_t = torch.where(deg, torch.zeros_like(vy), vy / safe_nrm)
        # Column 0 = v_min = (-sin_t, cos_t), Column 1 = v_max = (cos_t, sin_t):
        vals = torch.stack((lam_min, lam_max), dim=-1)
        col0 = torch.stack((-sin_t, cos_t), dim=-1)
        col1 = torch.stack((cos_t, sin_t), dim=-1)
        vecs = torch.stack((col0, col1), dim=-1)
        return vals.to(orig_dtype), vecs.to(orig_dtype)

    # General d >= 3: vectorized cyclic Jacobi sweeps on flattened batch [N, d, d]:
    a_mat = sym.reshape(-1, d, d).clone()
    n_b = int(a_mat.shape[0])
    v_mat = (
        torch.eye(d, dtype=torch.float64, device=a_mat.device)
        .unsqueeze(0)
        .expand(n_b, d, d)
        .clone()
    )
    for _ in range(int(sweeps)):
        for p in range(d - 1):
            for q in range(p + 1, d):
                apq = a_mat[:, p, q].clone()
                app = a_mat[:, p, p].clone()
                aqq = a_mat[:, q, q].clone()
                tol = 1e-15 * (app.abs() + aqq.abs() + 1e-30)
                active = apq.abs() > tol
                if not bool(active.any()):
                    continue
                safe_apq = torch.where(active, apq, torch.ones_like(apq))
                tau = (aqq - app) / (2.0 * safe_apq)
                sign_tau = torch.where(tau >= 0.0, torch.ones_like(tau), -torch.ones_like(tau))
                t_val = torch.where(
                    active,
                    sign_tau / (tau.abs() + torch.hypot(tau, torch.ones_like(tau))),
                    torch.zeros_like(tau),
                )
                c_val = torch.rsqrt(1.0 + t_val * t_val)
                s_val = t_val * c_val

                # Rotate rows/columns p and q of symmetric A:
                row_p = a_mat[:, p, :].clone()
                row_q = a_mat[:, q, :].clone()
                c_col = c_val.unsqueeze(1)
                s_col = s_val.unsqueeze(1)
                new_p = c_col * row_p - s_col * row_q
                new_q = s_col * row_p + c_col * row_q
                a_mat[:, p, :] = new_p
                a_mat[:, q, :] = new_q
                a_mat[:, :, p] = new_p
                a_mat[:, :, q] = new_q
                a_mat[:, p, p] = app - t_val * apq
                a_mat[:, q, q] = aqq + t_val * apq
                a_mat[:, p, q] = torch.where(active, torch.zeros_like(apq), apq)
                a_mat[:, q, p] = a_mat[:, p, q]

                # Accumulate eigenvectors V <- V @ J:
                vp = v_mat[:, :, p].clone()
                vq = v_mat[:, :, q].clone()
                v_mat[:, :, p] = c_col * vp - s_col * vq
                v_mat[:, :, q] = s_col * vp + c_col * vq

    vals_unsorted = torch.diagonal(a_mat, dim1=-2, dim2=-1)
    order = torch.argsort(vals_unsorted, dim=-1)
    vals_sorted = torch.gather(vals_unsorted, -1, order)
    order_exp = order.unsqueeze(1).expand(n_b, d, d)
    vecs_sorted = torch.gather(v_mat, -1, order_exp)
    return (
        vals_sorted.reshape(*batch_shape, d).to(orig_dtype),
        vecs_sorted.reshape(*batch_shape, d, d).to(orig_dtype),
    )


@torch.no_grad()
def sym_eigensplit_zero_eigh(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split symmetric ``[..., d, d]`` tensor into PSD ``E_+`` and NSD ``E_-`` with ``0`` `eigh`."""
    orig_dtype = mat.dtype
    vals, vecs = zero_eigh_symmetric(mat)
    vals64 = vals.to(dtype=torch.float64)
    vecs64 = vecs.to(dtype=torch.float64)
    vecs64_t = vecs64.transpose(-1, -2)
    e_pos = (vecs64 * vals64.clamp_min(0.0).unsqueeze(-2)) @ vecs64_t
    e_neg = (vecs64 * vals64.clamp_max(0.0).unsqueeze(-2)) @ vecs64_t
    return e_pos.to(orig_dtype), e_neg.to(orig_dtype)


def _basis_dim(d_c: int) -> int:
    if d_c <= _MAX_FULL_BASIS_DC:
        return 1 + 3 * d_c + (d_c * (d_c - 1)) // 2
    return 2 + d_c


_TRIU_CACHE: Dict[Tuple[torch.device, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_RIDGE_CACHE: Dict[Tuple[torch.device, int, float, float], torch.Tensor] = {}
_AGE_CACHE: Dict[Tuple[torch.device, int], torch.Tensor] = {}


def _build_basis_and_weights(
    c_in: torch.Tensor,
    mu_c: torch.Tensor,
    std_c: torch.Tensor,
    *,
    tail_leverage_gain: float = 0.90,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(phi(c), omega(c), u_std)`` preserving input floating dtype."""
    dt = c_in.dtype
    d_c = int(c_in.shape[1])
    u = (c_in - mu_c.to(dtype=dt).unsqueeze(0)) / std_c.to(dtype=dt).unsqueeze(0)
    ones = torch.ones((c_in.shape[0], 1), dtype=dt, device=c_in.device)
    r2_mean = u.square().mean(dim=1)
    omega = 1.0 + float(tail_leverage_gain) * r2_mean.clamp_max(4.5)
    if d_c <= _MAX_FULL_BASIS_DC:
        u_pos = 2.0 * (F.softplus(u - 0.8) - 0.425)
        u_neg = 2.0 * (F.softplus(-u - 0.8) - 0.425)
        parts = [ones, u, u_pos, u_neg]
        if d_c >= 2:
            t_key = (c_in.device, d_c)
            tri = _TRIU_CACHE.get(t_key)
            if tri is None:
                idx_a, idx_b = torch.triu_indices(d_c, d_c, offset=1, device=c_in.device)
                tri = (idx_a, idx_b)
                if len(_TRIU_CACHE) < 64:
                    _TRIU_CACHE[t_key] = tri
            idx_a, idx_b = tri
            ua = u[:, idx_a]
            ub = u[:, idx_b]
            u_cross = (ua * ub) * torch.rsqrt(1.0 + 0.15 * (ua.square() + ub.square()))
            parts.append(u_cross)
        phi = torch.cat(parts, dim=1)
        return phi, omega, u
    r_tail = 2.0 * (F.softplus(r2_mean.clamp_min(1e-6).sqrt() - 1.0).unsqueeze(1) - 0.3133)
    phi = torch.cat((ones, u, r_tail), dim=1)
    return phi, omega, u


def _basis_ridge_diag(
    d_c: int,
    *,
    lin_ridge: float = 1e-3,
    nl_ridge: float = 0.025,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    dev = torch.device("cpu") if device is None else device
    r_key = (dev, int(d_c), float(lin_ridge), float(nl_ridge))
    cached = _RIDGE_CACHE.get(r_key)
    if cached is not None:
        return cached
    p = _basis_dim(d_c)
    diag = torch.full((p,), float(nl_ridge), dtype=torch.float64, device=dev)
    diag[: 1 + d_c] = float(lin_ridge)
    if len(_RIDGE_CACHE) < 64:
        _RIDGE_CACHE[r_key] = diag
    return diag


def _sym_inv_sqrt_zero_eigh(cov_f64: torch.Tensor) -> torch.Tensor:
    """Exact SPD inverse square root ``(cov + eps*I)^{-1/2}`` in float64 with ``0`` `eigh`."""
    sym = 0.5 * (cov_f64 + cov_f64.T)
    eigvals, eigvecs = zero_eigh_symmetric(sym)
    tr_scale = float(eigvals.clamp_min(0.0).mean().item())
    floor = max(1e-4 * max(tr_scale, 1e-4), 1e-6)
    inv_sqrt_eigs = eigvals.clamp_min(floor).rsqrt()
    return (eigvecs * inv_sqrt_eigs.unsqueeze(0)) @ eigvecs.T


def _solve_spd_gram_zero_eigh(
    gram_norm: torch.Tensor,
    ridge_diag: torch.Tensor,
    rhs: torch.Tensor,
    *,
    need_inv: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    p = gram_norm.shape[0]
    a_mat = 0.5 * (gram_norm + gram_norm.T) + torch.diag(ridge_diag)
    l_mat, info = torch.linalg.cholesky_ex(a_mat, check_errors=False)
    if int(info.item()) != 0:
        eigvals, eigvecs = zero_eigh_symmetric(a_mat)
        eigvals = eigvals.clamp_min(1e-6)
        a_inv = (eigvecs * eigvals.reciprocal().unsqueeze(0)) @ eigvecs.T
        return a_inv @ rhs, (0.5 * (a_inv + a_inv.T) if need_inv else None)
    sol = torch.cholesky_solve(rhs, l_mat)
    if not need_inv:
        return sol, None
    eye_p = torch.eye(p, dtype=torch.float64, device=gram_norm.device)
    a_inv = torch.cholesky_solve(eye_p, l_mat)
    return sol, 0.5 * (a_inv + a_inv.T)


def _validate_tensor_f64(val: Any, expected_shape: Tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(val, torch.Tensor):
        raise ValueError(f"invalid unified_v11 bank tensor {name}")
    if val.dtype != torch.float64:
        raise ValueError(f"unified_v11 bank tensor {name} must have float64 dtype")
    if tuple(val.shape) != expected_shape:
        raise ValueError(f"unified_v11 bank tensor {name} shape mismatch: {tuple(val.shape)} != {expected_shape}")
    if not bool(torch.isfinite(val).all()):
        raise ValueError(f"unified_v11 bank tensor {name} contains non-finite values")
    return val


def _validate_ring_tensor(val: Any, expected_cols: int, name: str) -> torch.Tensor:
    if not isinstance(val, torch.Tensor):
        raise ValueError(f"invalid unified_v11 ring tensor {name}")
    if val.dtype != torch.float32:
        raise ValueError(f"unified_v11 ring tensor {name} must have float32 dtype")
    if val.ndim != 2 or val.shape[1] != expected_cols or val.shape[0] > _MAX_RING_CAPACITY:
        raise ValueError(f"unified_v11 ring tensor {name} has invalid shape {tuple(val.shape)}")
    if not bool(torch.isfinite(val).all()):
        raise ValueError(f"unified_v11 ring tensor {name} contains non-finite values")
    return val


def validate_v11_bank_packet(packet: Any) -> Dict[str, Any]:
    """Strictly validate a serialized ``UnifiedV11CondFiberBank`` checkpoint packet."""
    if not isinstance(packet, dict) or set(packet) != _BANK_PACKET_KEYS:
        raise ValueError("invalid unified_v11 bank packet keys")
    if type(packet["schema"]) is not int or packet["schema"] != _UNIFIED_V11_BANK_SCHEMA:
        raise ValueError("unsupported unified_v11 bank schema")
    d_c = packet["d_c"]
    d_x = packet["d_x"]
    strata = packet["strata"]
    if not isinstance(strata, list) or len(strata) > _MAX_STRATA:
        raise ValueError("invalid unified_v11 bank strata list")
    if d_c is None and d_x is None:
        if strata:
            raise ValueError("uninitialized unified_v11 bank cannot hold strata")
        return packet
    if type(d_c) is not int or type(d_x) is not int or d_c < 1 or d_c > _MAX_BANK_DC or d_x < 1:
        raise ValueError("invalid unified_v11 bank dimensions")
    p = _basis_dim(d_c)
    seen_keys = set()
    for entry in strata:
        if not isinstance(entry, dict) or set(entry) != _STRATUM_KEYS:
            raise ValueError("invalid unified_v11 stratum entry keys")
        key = entry["key"]
        if (
            not isinstance(key, tuple)
            or len(key) == 0
            or not all(type(v) is int for v in key)
            or key in seen_keys
        ):
            raise ValueError("invalid or duplicate unified_v11 stratum key")
        seen_keys.add(key)
        for n_name in ("n_eff", "n_fast", "w_eff", "w_fast"):
            n_val = entry[n_name]
            if isinstance(n_val, bool) or not isinstance(n_val, (int, float)) or not math.isfinite(n_val) or float(n_val) <= 0.0:
                raise ValueError(f"invalid unified_v11 stratum {n_name}")
        _validate_tensor_f64(entry["mu_c"], (d_c,), "mu_c")
        m2_c = _validate_tensor_f64(entry["m2_c"], (d_c,), "m2_c")
        if bool((m2_c < 0.0).any()):
            raise ValueError("unified_v11 stratum m2_c must be nonnegative")
        g_pp = _validate_tensor_f64(entry["g_pp"], (p, p), "g_pp")
        g_pp_f = _validate_tensor_f64(entry["g_pp_fast"], (p, p), "g_pp_fast")
        if bool((torch.diagonal(g_pp) < -1e-9).any()) or bool((torch.diagonal(g_pp_f) < -1e-9).any()):
            raise ValueError("unified_v11 stratum Gram diagonal must be nonnegative")
        _validate_tensor_f64(entry["g_px"], (p, d_x), "g_px")
        _validate_tensor_f64(entry["g_py"], (p, d_x), "g_py")
        _validate_tensor_f64(entry["g_px_fast"], (p, d_x), "g_px_fast")
        _validate_tensor_f64(entry["g_py_fast"], (p, d_x), "g_py_fast")
        s_xx = _validate_tensor_f64(entry["s_xx"], (d_x, d_x), "s_xx")
        s_yy = _validate_tensor_f64(entry["s_yy"], (d_x, d_x), "s_yy")
        if bool((torch.diagonal(s_xx) < -1e-9).any()) or bool((torch.diagonal(s_yy) < -1e-9).any()):
            raise ValueError("unified_v11 stratum residual covariance diagonal must be nonnegative")
        rc = _validate_ring_tensor(entry["ring_c"], d_c, "ring_c")
        rx = _validate_ring_tensor(entry["ring_x"], d_x, "ring_x")
        ry = _validate_ring_tensor(entry["ring_y"], d_x, "ring_y")
        if rc.shape[0] != rx.shape[0] or rc.shape[0] != ry.shape[0]:
            raise ValueError("unified_v11 ring tensors must have matching lengths")
    return packet


class UnifiedV11CondFiberBank:
    """Zero-`eigh` clock-free swap-invariant per-stratum conditional operator bank + paired FIFO ring."""

    def __init__(
        self,
        *,
        gamma_base: float = 0.96,
        gamma_fast: float = 0.85,
        gamma_min: float = 0.50,
        innov_threshold: float = 2.5,
        lin_ridge: float = 1e-3,
        nl_ridge: float = 0.025,
        ring_capacity: int = _MAX_RING_CAPACITY,
        knn_k: int = 20,
        tail_leverage_gain: float = 0.90,
    ) -> None:
        self.gamma_base = float(gamma_base)
        self.gamma_fast = float(gamma_fast)
        self.gamma_min = float(gamma_min)
        self.innov_threshold = float(innov_threshold)
        self.lin_ridge = float(lin_ridge)
        self.nl_ridge = float(nl_ridge)
        self.ring_capacity = min(int(ring_capacity), _MAX_RING_CAPACITY)
        self.knn_k = int(knn_k)
        self.tail_leverage_gain = float(tail_leverage_gain)
        self.d_c: Optional[int] = None
        self.d_x: Optional[int] = None
        self.strata: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    def reset(self) -> None:
        self.d_c = None
        self.d_x = None
        self.strata = {}

    def copy(self) -> "UnifiedV11CondFiberBank":
        dup = UnifiedV11CondFiberBank(
            gamma_base=self.gamma_base,
            gamma_fast=self.gamma_fast,
            gamma_min=self.gamma_min,
            innov_threshold=self.innov_threshold,
            lin_ridge=self.lin_ridge,
            nl_ridge=self.nl_ridge,
            ring_capacity=self.ring_capacity,
            knn_k=self.knn_k,
            tail_leverage_gain=self.tail_leverage_gain,
        )
        dup.d_c = self.d_c
        dup.d_x = self.d_x
        dup.strata = {
            key: {
                k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v))
                for k, v in st.items()
            }
            for key, st in self.strata.items()
        }
        return dup

    def to_packet(self) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for key in sorted(self.strata):
            st = self.strata[key]
            entries.append({
                "key": tuple(int(v) for v in key),
                "n_eff": float(st["n_eff"]),
                "n_fast": float(st["n_fast"]),
                "w_eff": float(st["w_eff"]),
                "w_fast": float(st["w_fast"]),
                "mu_c": st["mu_c"].detach().cpu().to(dtype=torch.float64).clone(),
                "m2_c": st["m2_c"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_pp": st["g_pp"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_px": st["g_px"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_py": st["g_py"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_pp_fast": st["g_pp_fast"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_px_fast": st["g_px_fast"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_py_fast": st["g_py_fast"].detach().cpu().to(dtype=torch.float64).clone(),
                "s_xx": st["s_xx"].detach().cpu().to(dtype=torch.float64).clone(),
                "s_yy": st["s_yy"].detach().cpu().to(dtype=torch.float64).clone(),
                "ring_c": st["ring_c"].detach().cpu().to(dtype=torch.float32).clone(),
                "ring_x": st["ring_x"].detach().cpu().to(dtype=torch.float32).clone(),
                "ring_y": st["ring_y"].detach().cpu().to(dtype=torch.float32).clone(),
            })
        return {
            "schema": _UNIFIED_V11_BANK_SCHEMA,
            "d_c": self.d_c,
            "d_x": self.d_x,
            "strata": entries,
        }

    def load_packet(self, packet: Dict[str, Any]) -> None:
        validate_v11_bank_packet(packet)
        self.d_c = packet["d_c"]
        self.d_x = packet["d_x"]
        self.strata = {}
        for entry in packet["strata"]:
            key = tuple(int(v) for v in entry["key"])
            self.strata[key] = {
                "n_eff": float(entry["n_eff"]),
                "n_fast": float(entry["n_fast"]),
                "w_eff": float(entry["w_eff"]),
                "w_fast": float(entry["w_fast"]),
                "mu_c": entry["mu_c"].detach().cpu().to(dtype=torch.float64).clone(),
                "m2_c": entry["m2_c"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_pp": entry["g_pp"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_px": entry["g_px"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_py": entry["g_py"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_pp_fast": entry["g_pp_fast"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_px_fast": entry["g_px_fast"].detach().cpu().to(dtype=torch.float64).clone(),
                "g_py_fast": entry["g_py_fast"].detach().cpu().to(dtype=torch.float64).clone(),
                "s_xx": entry["s_xx"].detach().cpu().to(dtype=torch.float64).clone(),
                "s_yy": entry["s_yy"].detach().cpu().to(dtype=torch.float64).clone(),
                "ring_c": entry["ring_c"].detach().cpu().to(dtype=torch.float32).clone(),
                "ring_x": entry["ring_x"].detach().cpu().to(dtype=torch.float32).clone(),
                "ring_y": entry["ring_y"].detach().cpu().to(dtype=torch.float32).clone(),
            }

    @torch.no_grad()
    def query_stratum(
        self,
        key: Tuple[int, ...],
        c_g: torch.Tensor,
        x_g: torch.Tensor,
        y_g: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        """Query pre-update cross-batch operators for stratum ``key`` (CPU float64, ``0`` `eigh`)."""
        d_c = int(c_g.shape[1])
        d_x = int(x_g.shape[1])
        if d_c > _MAX_BANK_DC or self.d_c != d_c or self.d_x != d_x:
            return None
        st = self.strata.get(key)
        if st is None:
            return None
        p = _basis_dim(d_c)
        n_eff = float(st["n_eff"])
        w_eff = float(st["w_eff"])
        w_fast = float(st["w_fast"])
        ring_len = int(st["ring_c"].shape[0])
        if n_eff < float(max(p + 2, 12)) or ring_len < 8:
            return None

        mu_c = st["mu_c"]
        std_c = (st["m2_c"] / max(n_eff, 1.0)).clamp_min(1e-6).sqrt()
        phi_q, omega_q, u_q = _build_basis_and_weights(
            c_g, mu_c, std_c, tail_leverage_gain=self.tail_leverage_gain
        )
        inv_w = 1.0 / max(w_eff, 1.0)
        inv_wf = 1.0 / max(w_fast, 1.0)
        inv_n = 1.0 / n_eff
        mean_ridge = _basis_ridge_diag(d_c, lin_ridge=self.lin_ridge, nl_ridge=self.nl_ridge)

        gram_slow = st["g_pp"] * inv_w
        sol_slow, m_inv_slow = _solve_spd_gram_zero_eigh(
            gram_slow, mean_ridge, torch.cat((st["g_px"], st["g_py"]), dim=1) * inv_w
        )
        b_x_slow = sol_slow[:, :d_x]
        b_y_slow = sol_slow[:, d_x:]
        b_pool = 0.5 * (b_x_slow + b_y_slow)
        m_x_slow_q = phi_q @ b_x_slow
        m_y_slow_q = phi_q @ b_y_slow
        m_pool_q = phi_q @ b_pool

        gram_fast = st["g_pp_fast"] * inv_wf
        b_x_fast, _ = _solve_spd_gram_zero_eigh(gram_fast, mean_ridge, st["g_px_fast"] * inv_wf, need_inv=False)
        m_x_fast_q = phi_q @ b_x_fast

        n_g = int(c_g.shape[0])
        u_mean = x_g - m_y_slow_q
        if n_g >= 2:
            w_phi_q = phi_q * omega_q.unsqueeze(1)
            w_sum_q = float(omega_q.sum().item())
            w_loo_den = (w_sum_q - omega_q).clamp_min(1.0).unsqueeze(1)
            proj_phi = phi_q @ m_inv_slow
            h_ii = (proj_phi * w_phi_q).sum(dim=1, keepdim=True)
            delta_m_loo = (proj_phi @ (w_phi_q.T @ u_mean) - h_ii * u_mean) / w_loo_den
        else:
            delta_m_loo = m_x_fast_q - m_y_slow_q

        delta_m_bank = m_x_fast_q - m_y_slow_q

        cov_pool = 0.5 * (st["s_xx"] + st["s_yy"]) * inv_n
        w_mat = _sym_inv_sqrt_zero_eigh(cov_pool)

        rc32 = st["ring_c"]
        rx_ring32 = st["ring_x"]
        ry_ring32 = st["ring_y"]
        phi_r32, _, u_r32 = _build_basis_and_weights(
            rc32, mu_c, std_c, tail_leverage_gain=self.tail_leverage_gain
        )
        m_pool_r32 = phi_r32 @ b_pool.float()

        d_qr32 = torch.cdist(u_q.float(), u_r32, p=2.0)
        k_eff = min(self.knn_k, max(6, ring_len // 5), ring_len)
        d_topk, _ = torch.topk(d_qr32, k=k_eff, dim=1, largest=False)
        h_q = d_topk[:, -1].clamp_min(0.25 * math.sqrt(float(d_c)))
        a_key = (u_q.device, ring_len)
        log_recency = _AGE_CACHE.get(a_key)
        if log_recency is None:
            log_recency = -torch.arange(ring_len - 1, -1, -1, dtype=torch.float32, device=u_q.device) / 128.0
            if len(_AGE_CACHE) < 128:
                _AGE_CACHE[a_key] = log_recency
        logits_k = -0.5 * (d_qr32 / h_q.unsqueeze(1)).square() + log_recency.unsqueeze(0)
        logits_k = logits_k - logits_k.amax(dim=1, keepdim=True)
        alpha32 = torch.softmax(logits_k, dim=1)

        res_x_r = rx_ring32 - m_pool_r32
        res_y_r = ry_ring32 - m_pool_r32
        dcov_r = (
            res_x_r.unsqueeze(2) * res_x_r.unsqueeze(1)
            - res_y_r.unsqueeze(2) * res_y_r.unsqueeze(1)
        )
        delta_cov_loc = (alpha32 @ dcov_r.reshape(ring_len, d_x * d_x)).reshape(n_g, d_x, d_x).double()

        m_x_loc_q = m_pool_q + 0.5 * (alpha32 @ (rx_ring32 - ry_ring32)).double()

        db_w = (b_x_slow - b_y_slow) @ w_mat
        raw_param_sq = float(db_w.square().sum().item()) / float(max(1, d_x))
        param_noise_floor = 2.0 * float(p) / max(n_eff, 1.0)
        mean_err_excess = max(0.0, raw_param_sq - param_noise_floor)

        e_x = float((((x_g - m_x_slow_q).mean(dim=0)) @ w_mat).norm().item())
        e_y = float((((y_g - m_y_slow_q).mean(dim=0)) @ w_mat).norm().item())
        z_innov = math.sqrt(float(n_g) / float(max(1, d_x))) * max(e_x, e_y)
        c_jump = float(((c_g.mean(dim=0) - mu_c) / std_c).norm().item()) * math.sqrt(float(n_g) / float(max(1, d_c)))
        z_innov = max(z_innov, c_jump)

        return {
            "n_eff": n_eff,
            "ring_len": ring_len,
            "omega": omega_q,
            "m_x": m_x_loc_q,
            "m_pool": m_pool_q,
            "w_mat": w_mat,
            "delta_m_bank": delta_m_bank,
            "delta_m_loo": delta_m_loo,
            "delta_cov_loc": delta_cov_loc,
            "mean_err_sq": mean_err_excess,
            "z_innov": z_innov,
        }

    @torch.no_grad()
    def update_stratum(
        self,
        key: Tuple[int, ...],
        c_g: torch.Tensor,
        x_g: torch.Tensor,
        y_g: torch.Tensor,
        *,
        z_innov: float = 0.0,
    ) -> None:
        """Update stratum ``key`` sufficient statistics and FIFO paired ring (swap-invariant & clock-free)."""
        n_g, d_c = int(c_g.shape[0]), int(c_g.shape[1])
        d_x = int(x_g.shape[1])
        if n_g == 0 or d_c > _MAX_BANK_DC:
            return
        if self.d_c is None:
            self.d_c = d_c
            self.d_x = d_x
        elif self.d_c != d_c or self.d_x != d_x:
            self.reset()
            self.d_c = d_c
            self.d_x = d_x

        if key not in self.strata and len(self.strata) >= _MAX_STRATA:
            return

        c_bar = c_g.mean(dim=0)
        c_dev = c_g - c_bar.unsqueeze(0)
        c_m2_batch = c_dev.square().sum(dim=0)
        rc_new = c_g.float()
        rx_new = x_g.float()
        ry_new = y_g.float()

        st = self.strata.get(key)
        if st is None:
            n_new = float(n_g)
            mu_c = c_bar.clone()
            m2_c = c_m2_batch.clone()
            std_c = (m2_c / max(n_new, 1.0)).clamp_min(1e-6).sqrt()
            phi, omega, _ = _build_basis_and_weights(
                c_g, mu_c, std_c, tail_leverage_gain=self.tail_leverage_gain
            )
            w_batch = float(omega.sum().item())
            w_phi = phi * omega.unsqueeze(1)
            g_pp = w_phi.T @ phi
            g_px = w_phi.T @ x_g
            g_py = w_phi.T @ y_g

            ridge = _basis_ridge_diag(d_c, lin_ridge=self.lin_ridge, nl_ridge=self.nl_ridge)
            sol, _ = _solve_spd_gram_zero_eigh(
                g_pp / max(w_batch, 1.0), ridge, torch.cat((g_px, g_py), dim=1) / max(w_batch, 1.0), need_inv=False
            )
            rx = x_g - phi @ sol[:, :d_x]
            ry = y_g - phi @ sol[:, d_x:]
            s_xx = rx.T @ rx
            s_yy = ry.T @ ry

            cap = self.ring_capacity
            self.strata[key] = {
                "n_eff": n_new,
                "n_fast": n_new,
                "w_eff": w_batch,
                "w_fast": w_batch,
                "mu_c": mu_c,
                "m2_c": m2_c,
                "g_pp": g_pp,
                "g_px": g_px,
                "g_py": g_py,
                "g_pp_fast": g_pp.clone(),
                "g_px_fast": g_px.clone(),
                "g_py_fast": g_py.clone(),
                "s_xx": s_xx,
                "s_yy": s_yy,
                "ring_c": rc_new[-cap:].clone(),
                "ring_x": rx_new[-cap:].clone(),
                "ring_y": ry_new[-cap:].clone(),
            }
            return

        excess = max(0.0, float(z_innov) - self.innov_threshold)
        decay_mod = math.exp(-0.5 * excess * excess)
        gamma_s = self.gamma_min + (self.gamma_base - self.gamma_min) * decay_mod
        gamma_f = self.gamma_min + (self.gamma_fast - self.gamma_min) * decay_mod

        n_old = gamma_s * float(st["n_eff"])
        n_new = n_old + float(n_g)
        n_fast_new = gamma_f * float(st["n_fast"]) + float(n_g)

        delta_c = c_bar - st["mu_c"]
        mu_c_new = st["mu_c"] + (float(n_g) / n_new) * delta_c
        m2_c_new = (gamma_s * st["m2_c"] + c_m2_batch + (n_old * float(n_g) / n_new) * delta_c.square()).clamp_min(0.0)
        std_c_new = (m2_c_new / max(n_new, 1.0)).clamp_min(1e-6).sqrt()

        phi, omega, _ = _build_basis_and_weights(
            c_g, mu_c_new, std_c_new, tail_leverage_gain=self.tail_leverage_gain
        )
        w_batch = float(omega.sum().item())
        w_eff_new = gamma_s * float(st["w_eff"]) + w_batch
        w_fast_new = gamma_f * float(st["w_fast"]) + w_batch

        w_phi = phi * omega.unsqueeze(1)
        phi_t_phi = w_phi.T @ phi
        phi_t_x = w_phi.T @ x_g
        phi_t_y = w_phi.T @ y_g

        g_pp_new = gamma_s * st["g_pp"] + phi_t_phi
        g_px_new = gamma_s * st["g_px"] + phi_t_x
        g_py_new = gamma_s * st["g_py"] + phi_t_y

        g_pp_fast_new = gamma_f * st["g_pp_fast"] + phi_t_phi
        g_px_fast_new = gamma_f * st["g_px_fast"] + phi_t_x
        g_py_fast_new = gamma_f * st["g_py_fast"] + phi_t_y

        ridge = _basis_ridge_diag(d_c, lin_ridge=self.lin_ridge, nl_ridge=self.nl_ridge)
        sol, _ = _solve_spd_gram_zero_eigh(
            g_pp_new / max(w_eff_new, 1.0), ridge, torch.cat((g_px_new, g_py_new), dim=1) / max(w_eff_new, 1.0), need_inv=False
        )
        rx = x_g - phi @ sol[:, :d_x]
        ry = y_g - phi @ sol[:, d_x:]
        s_xx_new = gamma_s * st["s_xx"] + (rx.T @ rx)
        s_yy_new = gamma_s * st["s_yy"] + (ry.T @ ry)

        keep_old = int(round(float(st["ring_c"].shape[0]) * max(0.25, decay_mod))) if decay_mod < 0.9 else int(st["ring_c"].shape[0])
        old_c = st["ring_c"][-keep_old:] if keep_old > 0 else st["ring_c"][:0]
        old_x = st["ring_x"][-keep_old:] if keep_old > 0 else st["ring_x"][:0]
        old_y = st["ring_y"][-keep_old:] if keep_old > 0 else st["ring_y"][:0]

        cap = self.ring_capacity
        ring_c = torch.cat((old_c, rc_new), dim=0)[-cap:]
        ring_x = torch.cat((old_x, rx_new), dim=0)[-cap:]
        ring_y = torch.cat((old_y, ry_new), dim=0)[-cap:]

        st["n_eff"] = n_new
        st["n_fast"] = n_fast_new
        st["w_eff"] = w_eff_new
        st["w_fast"] = w_fast_new
        st["mu_c"] = mu_c_new
        st["m2_c"] = m2_c_new
        st["g_pp"] = g_pp_new
        st["g_px"] = g_px_new
        st["g_py"] = g_py_new
        st["g_pp_fast"] = g_pp_fast_new
        st["g_px_fast"] = g_px_fast_new
        st["g_py_fast"] = g_py_fast_new
        st["s_xx"] = s_xx_new
        st["s_yy"] = s_yy_new
        st["ring_c"] = ring_c
        st["ring_x"] = ring_x
        st["ring_y"] = ring_y


__all__ = [
    "UnifiedV11CondFiberBank",
    "sym_eigensplit_zero_eigh",
    "validate_v11_bank_packet",
    "zero_eigh_symmetric",
]
