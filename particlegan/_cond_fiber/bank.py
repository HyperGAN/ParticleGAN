"""Strictly clock-free, swap-invariant, innovation-resetting per-stratum conditional operator bank.

Combines two strictly pre-update, fully checkpointed cross-batch structures per discrete stratum ``t``:

1. **Cross-Batch Sufficient-Statistics Mean & Precision Bank** (``gamma_base=0.96``, ~400 samples/stratum;
   fast tracking stream ``gamma_fast=0.85``, ~105 samples/stratum):
   - Exact parallel-Welford EMA context standardization ``(n_eff, mu_c, m2_c)``.
   - Symmetric WLS Gram & cross-moments over the bounded piecewise-hinge + bilinear-interaction basis
     ``phi(c) = [1, u, u_pos, u_neg, u_cross]`` (resolving affine slopes, outer-tail hinges, and
     2D diagonal MLP corner kinks with near-zero affine shrinkage and low finite-sample variance).
   - Swap-invariant pooled drift ``m_pool(c, t)`` and residual whitening
     ``W_t = (0.5 * (Sigma_XX + Sigma_YY) + eps * I)^{-1/2}``.
2. **Paired-Context Nonparametric FIFO Ring** (``M_max=256`` recent ``(c_j, X_j, Y_j)`` per stratum):
   - Centering by the swap-invariant pooled midpoint ``m_pool(c_j)``, the recency-weighted adaptive-kNN
     local covariance difference ``E_cov_loc(c_i) = sum_j alpha_ij * (rx_j rx_j^T - ry_j ry_j^T)``
     cancels the mean-bias outer product ``0.25 * Delta Delta^T`` identically by symmetry, isolating
     pure local conditional covariance error ``Sigma_X(c_i) - Sigma_Y(c_i)`` with zero variance collapse.
3. **Swap-Invariant Innovation Reset**:
   - Strictly clock-free adaptive forgetting triggered by ``max(e_X, e_Y, c_jump)``.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_COND_FIBER_BANK_SCHEMA = 5
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


def _basis_dim(d_c: int) -> int:
    if d_c <= _MAX_FULL_BASIS_DC:
        return 1 + 3 * d_c + (d_c * (d_c - 1)) // 2
    return 2 + d_c


def _build_basis_and_weights(
    c_f64: torch.Tensor,
    mu_c: torch.Tensor,
    std_c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(phi(c), omega(c), u_std)`` in float64."""
    d_c = int(c_f64.shape[1])
    u = (c_f64 - mu_c.unsqueeze(0)) / std_c.unsqueeze(0)
    ones = torch.ones((c_f64.shape[0], 1), dtype=torch.float64, device=c_f64.device)
    r2_mean = u.square().mean(dim=1)
    omega = 1.0 + 0.90 * r2_mean.clamp_max(4.5)
    if d_c <= _MAX_FULL_BASIS_DC:
        u_pos = 2.0 * (F.softplus(u - 0.8) - 0.425)
        u_neg = 2.0 * (F.softplus(-u - 0.8) - 0.425)
        parts = [ones, u, u_pos, u_neg]
        if d_c >= 2:
            idx_a, idx_b = torch.triu_indices(d_c, d_c, offset=1, device=c_f64.device)
            ua = u[:, idx_a]
            ub = u[:, idx_b]
            # Bounded bilinear interaction (resolves 2D diagonal MLP corner kinks with O(|u|) tail growth):
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
    p = _basis_dim(d_c)
    dev = torch.device("cpu") if device is None else device
    diag = torch.full((p,), float(nl_ridge), dtype=torch.float64, device=dev)
    diag[: 1 + d_c] = float(lin_ridge)
    return diag


def _sym_inv_sqrt(cov_f64: torch.Tensor) -> torch.Tensor:
    """Exact SPD inverse square root ``(cov + eps*I)^{-1/2}`` in float64."""
    sym = 0.5 * (cov_f64 + cov_f64.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    tr_scale = float(eigvals.clamp_min(0.0).mean().item())
    floor = max(1e-4 * max(tr_scale, 1e-4), 1e-6)
    inv_sqrt_eigs = eigvals.clamp_min(floor).rsqrt()
    return (eigvecs * inv_sqrt_eigs.unsqueeze(0)) @ eigvecs.T


def _solve_spd_gram(gram_norm: torch.Tensor, ridge_diag: torch.Tensor, rhs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    p = gram_norm.shape[0]
    a_mat = 0.5 * (gram_norm + gram_norm.T) + torch.diag(ridge_diag)
    eye_p = torch.eye(p, dtype=torch.float64, device=gram_norm.device)
    l_mat, info = torch.linalg.cholesky_ex(a_mat, check_errors=False)
    if int(info.item()) != 0:
        eigvals, eigvecs = torch.linalg.eigh(a_mat)
        eigvals = eigvals.clamp_min(1e-6)
        a_inv = (eigvecs * eigvals.reciprocal().unsqueeze(0)) @ eigvecs.T
        return a_inv @ rhs, a_inv
    sol = torch.cholesky_solve(rhs, l_mat)
    a_inv = torch.cholesky_solve(eye_p, l_mat)
    return sol, 0.5 * (a_inv + a_inv.T)


def _validate_tensor_f64(val: Any, expected_shape: Tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(val, torch.Tensor):
        raise ValueError(f"invalid cond_fiber bank tensor {name}")
    if val.dtype != torch.float64:
        raise ValueError(f"cond_fiber bank tensor {name} must have float64 dtype")
    if tuple(val.shape) != expected_shape:
        raise ValueError(f"cond_fiber bank tensor {name} shape mismatch: {tuple(val.shape)} != {expected_shape}")
    if not bool(torch.isfinite(val).all()):
        raise ValueError(f"cond_fiber bank tensor {name} contains non-finite values")
    return val


def _validate_ring_tensor(val: Any, expected_cols: int, name: str) -> torch.Tensor:
    if not isinstance(val, torch.Tensor):
        raise ValueError(f"invalid cond_fiber ring tensor {name}")
    if val.dtype != torch.float32:
        raise ValueError(f"cond_fiber ring tensor {name} must have float32 dtype")
    if val.ndim != 2 or val.shape[1] != expected_cols or val.shape[0] > _MAX_RING_CAPACITY:
        raise ValueError(f"cond_fiber ring tensor {name} has invalid shape {tuple(val.shape)}")
    if not bool(torch.isfinite(val).all()):
        raise ValueError(f"cond_fiber ring tensor {name} contains non-finite values")
    return val


def validate_bank_packet(packet: Any) -> Dict[str, Any]:
    """Strictly validate a serialized ``CondFiberBank`` checkpoint packet."""
    if not isinstance(packet, dict) or set(packet) != _BANK_PACKET_KEYS:
        raise ValueError("invalid cond_fiber bank packet keys")
    if type(packet["schema"]) is not int or packet["schema"] != _COND_FIBER_BANK_SCHEMA:
        raise ValueError("unsupported cond_fiber bank schema")
    d_c = packet["d_c"]
    d_x = packet["d_x"]
    strata = packet["strata"]
    if not isinstance(strata, list) or len(strata) > _MAX_STRATA:
        raise ValueError("invalid cond_fiber bank strata list")
    if d_c is None and d_x is None:
        if strata:
            raise ValueError("uninitialized cond_fiber bank cannot hold strata")
        return packet
    if type(d_c) is not int or type(d_x) is not int or d_c < 1 or d_c > _MAX_BANK_DC or d_x < 1:
        raise ValueError("invalid cond_fiber bank dimensions")
    p = _basis_dim(d_c)
    seen_keys = set()
    for entry in strata:
        if not isinstance(entry, dict) or set(entry) != _STRATUM_KEYS:
            raise ValueError("invalid cond_fiber stratum entry keys")
        key = entry["key"]
        if (
            not isinstance(key, tuple)
            or len(key) == 0
            or not all(type(v) is int for v in key)
            or key in seen_keys
        ):
            raise ValueError("invalid or duplicate cond_fiber stratum key")
        seen_keys.add(key)
        for n_name in ("n_eff", "n_fast", "w_eff", "w_fast"):
            n_val = entry[n_name]
            if isinstance(n_val, bool) or not isinstance(n_val, (int, float)) or not math.isfinite(n_val) or float(n_val) <= 0.0:
                raise ValueError(f"invalid cond_fiber stratum {n_name}")
        _validate_tensor_f64(entry["mu_c"], (d_c,), "mu_c")
        m2_c = _validate_tensor_f64(entry["m2_c"], (d_c,), "m2_c")
        if bool((m2_c < 0.0).any()):
            raise ValueError("cond_fiber stratum m2_c must be nonnegative")
        g_pp = _validate_tensor_f64(entry["g_pp"], (p, p), "g_pp")
        g_pp_f = _validate_tensor_f64(entry["g_pp_fast"], (p, p), "g_pp_fast")
        if bool((torch.diagonal(g_pp) < -1e-9).any()) or bool((torch.diagonal(g_pp_f) < -1e-9).any()):
            raise ValueError("cond_fiber stratum Gram diagonal must be nonnegative")
        _validate_tensor_f64(entry["g_px"], (p, d_x), "g_px")
        _validate_tensor_f64(entry["g_py"], (p, d_x), "g_py")
        _validate_tensor_f64(entry["g_px_fast"], (p, d_x), "g_px_fast")
        _validate_tensor_f64(entry["g_py_fast"], (p, d_x), "g_py_fast")
        s_xx = _validate_tensor_f64(entry["s_xx"], (d_x, d_x), "s_xx")
        s_yy = _validate_tensor_f64(entry["s_yy"], (d_x, d_x), "s_yy")
        if bool((torch.diagonal(s_xx) < -1e-9).any()) or bool((torch.diagonal(s_yy) < -1e-9).any()):
            raise ValueError("cond_fiber stratum residual covariance diagonal must be nonnegative")
        rc = _validate_ring_tensor(entry["ring_c"], d_c, "ring_c")
        rx = _validate_ring_tensor(entry["ring_x"], d_x, "ring_x")
        ry = _validate_ring_tensor(entry["ring_y"], d_x, "ring_y")
        if rc.shape[0] != rx.shape[0] or rc.shape[0] != ry.shape[0]:
            raise ValueError("cond_fiber ring tensors must have matching lengths")
    return packet


class CondFiberBank:
    """Clock-free swap-invariant per-stratum sufficient-statistics + paired-ring conditional operator bank."""

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
    ) -> None:
        self.gamma_base = float(gamma_base)
        self.gamma_fast = float(gamma_fast)
        self.gamma_min = float(gamma_min)
        self.innov_threshold = float(innov_threshold)
        self.lin_ridge = float(lin_ridge)
        self.nl_ridge = float(nl_ridge)
        self.ring_capacity = min(int(ring_capacity), _MAX_RING_CAPACITY)
        self.knn_k = int(knn_k)
        self.d_c: Optional[int] = None
        self.d_x: Optional[int] = None
        self.strata: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    def reset(self) -> None:
        self.d_c = None
        self.d_x = None
        self.strata = {}

    def copy(self) -> "CondFiberBank":
        dup = CondFiberBank(
            gamma_base=self.gamma_base,
            gamma_fast=self.gamma_fast,
            gamma_min=self.gamma_min,
            innov_threshold=self.innov_threshold,
            lin_ridge=self.lin_ridge,
            nl_ridge=self.nl_ridge,
            ring_capacity=self.ring_capacity,
            knn_k=self.knn_k,
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
            "schema": _COND_FIBER_BANK_SCHEMA,
            "d_c": self.d_c,
            "d_x": self.d_x,
            "strata": entries,
        }

    def load_packet(self, packet: Dict[str, Any]) -> None:
        validate_bank_packet(packet)
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
        """Query pre-update cross-batch operators for stratum ``key`` (CPU float64)."""
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
        phi_q, omega_q, u_q = _build_basis_and_weights(c_g, mu_c, std_c)
        inv_w = 1.0 / max(w_eff, 1.0)
        inv_wf = 1.0 / max(w_fast, 1.0)
        inv_n = 1.0 / n_eff
        mean_ridge = _basis_ridge_diag(d_c, lin_ridge=self.lin_ridge, nl_ridge=self.nl_ridge)

        # 1. Slow symmetric WLS bank: swap-invariant pooled drift m_pool & high-precision target m_Y:
        gram_slow = st["g_pp"] * inv_w
        sol_slow, m_inv_slow = _solve_spd_gram(
            gram_slow, mean_ridge, torch.cat((st["g_px"], st["g_py"]), dim=1) * inv_w
        )
        b_x_slow = sol_slow[:, :d_x]
        b_y_slow = sol_slow[:, d_x:]
        b_pool = 0.5 * (b_x_slow + b_y_slow)
        m_x_slow_q = phi_q @ b_x_slow
        m_y_slow_q = phi_q @ b_y_slow
        m_pool_q = phi_q @ b_pool

        # Fast WLS bank: critically-damped tracking of moving generator mean m_X (zero ringing!):
        gram_fast = st["g_pp_fast"] * inv_wf
        b_x_fast, _ = _solve_spd_gram(gram_fast, mean_ridge, st["g_px_fast"] * inv_wf)
        m_x_fast_q = phi_q @ b_x_fast

        # Zero-lag exact WLS LOO current-batch mean error (instant damping, zero variance collapse!):
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

        # Swap-invariant pooled residual whitening W_t = (0.5*(Sigma_XX + Sigma_YY) + eps*I)^{-1/2}:
        cov_pool = 0.5 * (st["s_xx"] + st["s_yy"]) * inv_n
        w_mat = _sym_inv_sqrt(cov_pool)

        # 2. Nonparametric paired-context FIFO ring for exact local conditional covariance discrepancy
        #    (midpoint-centered => exact cancellation of 0.25 * Delta Delta^T, zero 2D-corner blind spots!):
        rc = st["ring_c"].double()
        rx_ring = st["ring_x"].double()
        ry_ring = st["ring_y"].double()
        phi_r, _, u_r = _build_basis_and_weights(rc, mu_c, std_c)
        m_pool_r = phi_r @ b_pool

        d_qr = torch.cdist(u_q, u_r, p=2.0)
        k_eff = min(self.knn_k, max(6, ring_len // 5), ring_len)
        d_topk, _ = torch.topk(d_qr, k=k_eff, dim=1, largest=False)
        h_q = d_topk[:, -1].clamp_min(0.25 * math.sqrt(float(d_c)))
        age = torch.arange(ring_len - 1, -1, -1, dtype=torch.float64, device=u_q.device)
        log_recency = -age / 128.0
        logits_k = -0.5 * (d_qr / h_q.unsqueeze(1)).square() + log_recency.unsqueeze(0)
        logits_k = logits_k - logits_k.amax(dim=1, keepdim=True)
        alpha = torch.softmax(logits_k, dim=1)

        res_x_r = rx_ring - m_pool_r
        res_y_r = ry_ring - m_pool_r
        dcov_r = (
            res_x_r.unsqueeze(2) * res_x_r.unsqueeze(1)
            - res_y_r.unsqueeze(2) * res_y_r.unsqueeze(1)
        )
        delta_cov_loc = torch.einsum("nm,mij->nij", alpha, dcov_r)

        # True local X mean around q for exact covariance residual centering rx_q = X_q - m_x_loc_q:
        m_x_loc_q = m_pool_q + 0.5 * (alpha @ (rx_ring - ry_ring))

        # Parameter-space whitened operator discrepancy (weights affine, tail hinges, and 2D corners
        # equally instead of letting 95% core samples hide an unresolved 5% outer-tail error!):
        db_w = (b_x_slow - b_y_slow) @ w_mat  # [p, d_x]
        raw_param_sq = float(db_w.square().sum().item()) / float(max(1, d_x))
        param_noise_floor = 2.0 * float(p) / max(n_eff, 1.0)
        mean_err_excess = max(0.0, raw_param_sq - param_noise_floor)

        # Swap-invariant innovation score (strictly clock-free):
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
            phi, omega, _ = _build_basis_and_weights(c_g, mu_c, std_c)
            w_batch = float(omega.sum().item())
            w_phi = phi * omega.unsqueeze(1)
            g_pp = w_phi.T @ phi
            g_px = w_phi.T @ x_g
            g_py = w_phi.T @ y_g

            ridge = _basis_ridge_diag(d_c, lin_ridge=self.lin_ridge, nl_ridge=self.nl_ridge)
            sol, _ = _solve_spd_gram(
                g_pp / max(w_batch, 1.0), ridge, torch.cat((g_px, g_py), dim=1) / max(w_batch, 1.0)
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

        # Clock-free swap-invariant innovation reset:
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

        phi, omega, _ = _build_basis_and_weights(c_g, mu_c_new, std_c_new)
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
        sol, _ = _solve_spd_gram(
            g_pp_new / max(w_eff_new, 1.0), ridge, torch.cat((g_px_new, g_py_new), dim=1) / max(w_eff_new, 1.0)
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
    "CondFiberBank",
    "validate_bank_packet",
]
