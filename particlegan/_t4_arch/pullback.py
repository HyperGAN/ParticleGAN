"""Valley-Coalesced Concentric Pullback & fwAD Saturation Preconditioning (Theory 4B).

Solves Output-Head Saturation Deadlock (``tanh`` rail trapping & bounded scaled
``tanh`` explosion), Deep Collapsed Jacobian Wakeup (16-layer unnormalized MLPs),
and Total-Variance Disconnected-Mode Accuracy without modifying any model
architecture or custom backward derivative:

1. **Valley-Coalesced Dynamic Mode Atlas** ``(C_k, v_k, w_k)``:
   Discovers separated modes up to ``max_prototypes`` via valley-separated
   novelty insertion, coalesces intra-basin duplicates so unimodal Gaussians are
   never split into multi-lobe sub-cells, and tracks mathematically unbiased
   within-mode per-coordinate variance ``v_k``.
2. **Exact Forward-AD (``fwAD``) Saturation-Aware Pullback & Pre-Rail Envelope**:
   Evaluates ``(G(z), J_G(z) dz)`` in one 1st-order forward-AD pass (keeping
   ``G``'s declared backward derivative 100% unaltered, with automatic central
   secant fallback for custom quantized ops without ``jvp`` rules) to rescale
   attenuated ``sech^2`` tail gradients while gently bounding pre-rail excursions.
3. **Concentric Total-Variance Decomposition & Dead-Jacobian Recovery**:
   Matches smooth pushforward covariance ``Sigma_push`` on companion orthonormal
   frames to ``0.72 * v_k I`` in Part A (with halo-debiased fixed-point targets
   eliminating entropic Sinkhorn inward shrinkage) and supplies exact complementary
   intra-mode particle dispersion ``v_mu* = max(0, v_k - v_push)`` via zero-RNG
   Weyl Box-Muller offsets in Part B, plus inward recovery for particles trapped
   in flat zero-Jacobian clamp/ReLU zones.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.autograd import forward_ad as fwAD
import torch.nn.functional as F

from .._empf.prior import EMPFPriorRegularizer
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior

_PRIMES_RAD = (
    0.4142135623730951,  # sqrt(2) - 1
    0.7320508075688772,  # sqrt(3) - 1
    0.2360679774997897,  # sqrt(5) - 2
    0.6457513110645906,  # sqrt(7) - 2
    0.3166247903554000,  # sqrt(11) - 3
    0.6055512754639893,  # sqrt(13) - 3
    0.1231056256176606,  # sqrt(17) - 4
    0.3588989435406740,  # sqrt(19) - 4
)

_T4_STATE_KEYS = frozenset({
    "schema",
    "atlas_initialized",
    "is_point",
    "proto_c",
    "proto_var",
    "proto_w",
    "proto_hits",
    "psi",
    "w_fake",
    "v_push",
    "res_ema",
    "iso_initialized",
    "iso_ema",
    "iso_err_ema",
})


def _deterministic_box_muller(rows: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Zero-RNG Weyl-sequence Box-Muller Gaussian vectors indexed solely by particle row ID."""
    r_f = rows.to(torch.float64) + 0.5
    cols = []
    n_pairs = (dim + 1) // 2
    for j in range(n_pairs):
        a1 = _PRIMES_RAD[(2 * j) % len(_PRIMES_RAD)]
        a2 = _PRIMES_RAD[(2 * j + 1) % len(_PRIMES_RAD)]
        u1 = torch.remainder(r_f * a1, 1.0).clamp(1e-4, 1.0 - 1e-4)
        u2 = torch.remainder(r_f * a2, 1.0)
        rad = torch.sqrt(-2.0 * torch.log(u1)).clamp_max(2.95)
        ang = (2.0 * math.pi) * u2
        cols.append(rad * torch.cos(ang))
        if len(cols) < dim:
            cols.append(rad * torch.sin(ang))
    return torch.stack(cols[:dim], dim=1).to(dtype=dtype)


def _project_for_transport(x: torch.Tensor) -> torch.Tensor:
    """Exact deterministic spatial block-mean projection for >8x8 4D image tensors."""
    b = len(x)
    if x.ndim == 4 and (x.shape[2] > 8 or x.shape[3] > 8):
        _, c, h, w = x.shape
        th, tw = min(h, 8), min(w, 8)
        scale = math.sqrt(float(h * w) / float(th * tw))
        if h % th == 0 and w % tw == 0:
            pooled = x.reshape(b, c, th, h // th, tw, w // tw).mean(dim=(3, 5))
            return pooled.reshape(b, -1) * scale
        pooled = F.adaptive_avg_pool2d(x.float(), (th, tw)).to(x.dtype)
        return pooled.reshape(b, -1) * scale
    return x.reshape(b, -1)


def _feydy_sinkhorn_barycenter(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    iters: int = 6,
    ref_mean: Optional[torch.Tensor] = None,
    ref_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Scale-normalized FP32 Feydy-debiased Sinkhorn target (with streaming-stabilized 1D Monge quantile OT when D=1)."""
    with torch.no_grad():
        fx = x.detach().float()
        ry = y.detach().float()
        b, m = fx.shape[0], ry.shape[0]
        r_std = ry.std(dim=0, unbiased=False).mean().clamp_min(1e-5)
        f_std = fx.std(dim=0, unbiased=False).mean().clamp_min(1e-12)
        if fx.shape[1] == 1 and b == m:
            _, perm_x = torch.sort(fx[:, 0], stable=True)
            y_sorted, _ = torch.sort(ry[:, 0], stable=True)
            if ref_mean is not None and ref_std is not None and b >= 4:
                yb_mean = y_sorted.mean()
                yb_std = y_sorted.std(unbiased=True).clamp_min(1e-6)
                y_sorted = ref_mean.reshape(()) + ref_std.reshape(()) * ((y_sorted - yb_mean) / yb_std)
            targ_1d = torch.empty_like(fx)
            targ_1d[perm_x, 0] = y_sorted
            res_1d = float((fx - targ_1d).abs().mean().item() / float(r_std.item()))
            return targ_1d.to(x.dtype), res_1d

        if float((f_std / r_std).item()) < 0.30:
            fw = (fx - fx.mean(0, keepdim=True)) / fx.std(0, keepdim=True).clamp_min(1e-12)
            rw = (ry - ry.mean(0, keepdim=True)) / ry.std(0, keepdim=True).clamp_min(1e-6)
            d2 = 0.5 * torch.cdist(fw, rw).pow(2)
            eps = 0.08
            u = torch.zeros(b, device=fx.device)
            v = torch.zeros(m, device=fx.device)
            log_b, log_m = -math.log(float(b)), -math.log(float(m))
            for _ in range(iters):
                v = log_m - torch.logsumexp(-d2 / eps + u.unsqueeze(1), dim=0)
                u = log_b - torch.logsumexp(-d2 / eps + v.unsqueeze(0), dim=1)
            pi = torch.exp(-d2 / eps + u.unsqueeze(1) + v.unsqueeze(0))
            targ = (float(b) * pi) @ ry
            res = float((fx - targ).norm(dim=-1).mean().item() / float(r_std.item()))
            return targ.to(x.dtype), res

        scale = (2.0 * ry.var(dim=0, unbiased=False).sum()).clamp_min(1e-8)
        d2_xy = 0.5 * torch.cdist(fx, ry).pow(2) / scale
        d2_xx = 0.5 * torch.cdist(fx, fx).pow(2) / scale
        eps = 0.05
        log_b, log_m = -math.log(float(b)), -math.log(float(m))
        u = torch.zeros(b, device=fx.device)
        v = torch.zeros(m, device=fx.device)
        for _ in range(iters):
            v = log_m - torch.logsumexp(-d2_xy / eps + u.unsqueeze(1), dim=0)
            u = log_b - torch.logsumexp(-d2_xy / eps + v.unsqueeze(0), dim=1)
        pi_xy = torch.exp(-d2_xy / eps + u.unsqueeze(1) + v.unsqueeze(0))
        bary_xy = (float(b) * pi_xy) @ ry

        u_xx = torch.zeros(b, device=fx.device)
        v_xx = torch.zeros(b, device=fx.device)
        for _ in range(iters):
            v_xx = log_b - torch.logsumexp(-d2_xx / eps + u_xx.unsqueeze(1), dim=0)
            u_xx = log_b - torch.logsumexp(-d2_xx / eps + v_xx.unsqueeze(0), dim=1)
        pi_xx = torch.exp(-d2_xx / eps + u_xx.unsqueeze(1) + v_xx.unsqueeze(0))
        bary_xx = (float(b) * pi_xx) @ fx

        drift = bary_xy - bary_xx
        y_deb = fx + drift
        res = float(drift.norm(dim=-1).mean().item() / float(r_std.item()))
        return y_deb.to(x.dtype), res


def _validate_t4_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict) or set(state) != _T4_STATE_KEYS:
        raise ValueError("invalid T4ArchPriorRegularizer checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != 1:
        raise ValueError("unsupported T4ArchPriorRegularizer checkpoint schema")
    for bkey in ("atlas_initialized", "is_point", "iso_initialized"):
        if type(state[bkey]) is not bool:
            raise ValueError(f"invalid T4 boolean field {bkey}")
    for fkey, strict_pos in (("res_ema", False), ("iso_ema", True), ("iso_err_ema", False)):
        val = state[fkey]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"invalid T4 scalar field {fkey}")
        if float(val) <= 0.0 if strict_pos else float(val) < 0.0:
            raise ValueError(f"out-of-range T4 scalar field {fkey}")

    t_keys = ("proto_c", "proto_var", "proto_w", "proto_hits", "psi", "w_fake", "v_push")
    if not state["atlas_initialized"]:
        if any(state[k] is not None for k in t_keys):
            raise ValueError("uninitialized T4 atlas must store None tensors")
    else:
        for k in t_keys:
            t = state[k]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not bool(torch.isfinite(t).all()):
                raise ValueError(f"invalid T4 {k} tensor")
        c = state["proto_c"]
        if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
            raise ValueError("invalid T4 proto_c shape")
        k_len, d_len = c.shape
        for k in ("proto_var", "proto_w", "proto_hits", "psi", "w_fake"):
            if state[k].ndim != 1 or state[k].shape[0] != k_len:
                raise ValueError(f"incompatible T4 {k} shape")
        if state["v_push"].ndim != 2 or state["v_push"].shape != (k_len, d_len):
            raise ValueError("incompatible T4 v_push shape")
        if bool((state["proto_var"] <= 0).any()) or bool((state["proto_hits"] <= 0).any()):
            raise ValueError("T4 proto_var and proto_hits must be positive")
        if bool((state["proto_w"] < 0).any()) or not math.isclose(float(state["proto_w"].sum().item()), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("T4 proto_w must be nonnegative and sum to 1")
        if bool((state["w_fake"] < 0).any()) or bool((state["v_push"] < 0).any()):
            raise ValueError("T4 w_fake and v_push must be nonnegative")
    return state


class T4ArchPriorRegularizer(nn.Module):
    """Recipe-owned Theory 4B Valley-Coalesced Pullback & Saturation module (``prior_regularizer='t4_arch'``)."""

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
        max_prototypes: int = 256,
        empf_record: Any = None,
        bound_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an integer >= 2")
        self.max_prototypes = int(max_prototypes)
        self.fallback = EMPFPriorRegularizer(
            reference,
            weight=weight,
            target_std=target_std,
            eps=eps,
            empf_record=empf_record,
        )
        self._bound_device = bound_device

        self.atlas_initialized: bool = False
        self.is_point: bool = False
        self.proto_c: Optional[torch.Tensor] = None
        self.proto_var: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.proto_hits: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.w_fake: Optional[torch.Tensor] = None
        self.v_push: Optional[torch.Tensor] = None
        self.res_ema: float = 1.0
        self.iso_initialized: bool = False
        self.iso_ema: float = 1.0
        self.iso_err_ema: float = 0.0

    @property
    def empf_record(self) -> Any:
        return self.fallback.empf_record

    @empf_record.setter
    def empf_record(self, value: Any) -> None:
        self.fallback.empf_record = value

    @property
    def weight(self) -> float:
        return self.fallback.weight

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.fallback(rows)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        return {
            "schema": 1,
            "atlas_initialized": bool(self.atlas_initialized),
            "is_point": bool(self.is_point),
            "proto_c": None if self.proto_c is None else self.proto_c.detach().clone(),
            "proto_var": None if self.proto_var is None else self.proto_var.detach().clone(),
            "proto_w": None if self.proto_w is None else self.proto_w.detach().clone(),
            "proto_hits": None if self.proto_hits is None else self.proto_hits.detach().clone(),
            "psi": None if self.psi is None else self.psi.detach().clone(),
            "w_fake": None if self.w_fake is None else self.w_fake.detach().clone(),
            "v_push": None if self.v_push is None else self.v_push.detach().clone(),
            "res_ema": float(self.res_ema),
            "iso_initialized": bool(self.iso_initialized),
            "iso_ema": float(self.iso_ema),
            "iso_err_ema": float(self.iso_err_ema),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        del strict
        validated = _validate_t4_state(state_dict)
        if validated["atlas_initialized"] and validated["proto_c"].shape[0] > self.max_prototypes:
            raise ValueError("T4 proto_c count exceeds configured max_prototypes")

        dev = self._bound_device

        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.to(device=dev).detach().clone() if dev is not None else t.detach().clone()

        self.atlas_initialized = bool(validated["atlas_initialized"])
        self.is_point = bool(validated["is_point"])
        self.proto_c = _mv(validated["proto_c"])
        self.proto_var = _mv(validated["proto_var"])
        self.proto_w = _mv(validated["proto_w"])
        self.proto_hits = _mv(validated["proto_hits"])
        self.psi = _mv(validated["psi"])
        self.w_fake = _mv(validated["w_fake"])
        self.v_push = _mv(validated["v_push"])
        self.res_ema = float(validated["res_ema"])
        self.iso_initialized = bool(validated["iso_initialized"])
        self.iso_ema = float(validated["iso_ema"])
        self.iso_err_ema = float(validated["iso_err_ema"])

    def _filter_atlas(self, mask: torch.Tensor) -> None:
        self.proto_c = self.proto_c[mask]
        self.proto_var = self.proto_var[mask]
        self.proto_w = self.proto_w[mask] / self.proto_w[mask].sum().clamp_min(1e-12)
        self.proto_hits = self.proto_hits[mask]
        self.psi = self.psi[mask] - self.psi[mask].mean()
        self.w_fake = self.w_fake[mask] / self.w_fake[mask].sum().clamp_min(1e-12)
        self.v_push = self.v_push[mask]

    @torch.no_grad()
    def update_atlas(self, yr: torch.Tensor) -> None:
        x = yr.detach().float()
        b, dim = x.shape
        raw_var = x.var(dim=0, unbiased=False).mean()
        is_point = float(raw_var.item()) < 1e-11
        if is_point:
            self.is_point = True
            self.proto_c = x[:1].clone()
            self.proto_var = torch.full((1,), 1e-10, device=x.device, dtype=torch.float32)
            self.proto_w = torch.ones(1, device=x.device, dtype=torch.float32)
            self.proto_hits = torch.ones(1, device=x.device, dtype=torch.float32)
            self.psi = torch.zeros(1, device=x.device, dtype=torch.float32)
            self.w_fake = torch.ones(1, device=x.device, dtype=torch.float32)
            self.v_push = torch.zeros((1, dim), device=x.device, dtype=torch.float32)
            self.atlas_initialized = True
            return

        self.is_point = False
        if not self.atlas_initialized or self.proto_c is None or self.proto_c.shape[1] != dim:
            sub = x[: min(b, 512)]
            if len(sub) >= 2:
                d2_sub = torch.cdist(sub, sub).pow(2)
                d2_sub.fill_diagonal_(float("inf"))
                k_nn = max(1, min(6, len(sub) // 64))
                nn_d2 = torch.topk(d2_sub, k=k_nn, dim=1, largest=False).values[:, -1]
                pos_nn = nn_d2[nn_d2 > 1e-12]
                v0 = float(pos_nn.quantile(0.35).item()) / float(dim) if len(pos_nn) > 0 else float(raw_var.item())
            else:
                v0 = float(raw_var.item())
            v0 = max(v0, 1e-8)

            centers = [x[0]]
            min_d2 = (x - x[0]).pow(2).sum(dim=-1)
            sep_thresh = max(25.0, 14.0 * float(dim)) * v0
            while len(centers) < self.max_prototypes:
                if int((min_d2 > sep_thresh).sum().item()) < 2:
                    break
                max_val, idx = min_d2.max(dim=0)
                if float(max_val.item()) <= sep_thresh:
                    break
                c_new = x[idx]
                centers.append(c_new)
                min_d2 = torch.minimum(min_d2, (x - c_new).pow(2).sum(dim=-1))

            self.proto_c = torch.stack(centers, dim=0).clone()
            k = len(self.proto_c)
            d2_init = torch.cdist(x, self.proto_c).pow(2)
            assign = d2_init.argmin(dim=1)
            cnt = torch.bincount(assign, minlength=k).float()
            self.proto_w = (cnt + 0.5) / float(b + 0.5 * k)
            self.proto_hits = cnt.clamp_min(1.0)
            self.proto_var = torch.full((k,), v0, device=x.device, dtype=torch.float32)
            for idx_k in range(k):
                pts = x[assign == idx_k]
                if len(pts) >= 1:
                    self.proto_c[idx_k] = pts.mean(dim=0)
                if len(pts) >= 3:
                    self.proto_var[idx_k] = (
                        (pts - self.proto_c[idx_k]).pow(2).sum(dim=-1).sum() / float(dim * (len(pts) - 1))
                    ).clamp_min(1e-8)
            self.psi = torch.zeros(k, device=x.device, dtype=torch.float32)
            self.w_fake = self.proto_w.clone()
            self.v_push = (0.70 * self.proto_var).unsqueeze(1).expand(k, dim).clone()
            self.atlas_initialized = True
            return

        d2 = torch.cdist(x, self.proto_c).pow(2)
        d2_min, assign = d2.min(dim=1)
        pair_spread = (2.0 * x.var(dim=0, unbiased=False).sum()).clamp_min(1e-10)
        r2p = float(torch.sqrt(d2_min.mean() / pair_spread).item())
        shock = max(0.0, min(1.0, (r2p - 0.45) / 0.35))
        if shock > 0.25:
            self.proto_hits.mul_(1.0 - 0.85 * shock).clamp_min_(1.0)

        k = len(self.proto_c)
        cnt = torch.bincount(assign, minlength=k).float()
        b_w = cnt / float(b)
        alpha_w = 0.02 + 0.48 * (shock * shock)
        self.proto_w = (1.0 - alpha_w) * self.proto_w + alpha_w * b_w
        self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

        hit = cnt > 0
        if bool(hit.any()):
            b_sum = torch.zeros((k, dim), device=x.device, dtype=torch.float32).index_add_(0, assign, x)
            b_mean = b_sum[hit] / cnt[hit].unsqueeze(1)
            self.proto_hits[hit] = (self.proto_hits[hit] + cnt[hit]).clamp_max(8000.0)
            base_c = (0.012 + 0.60 / torch.sqrt(self.proto_hits[hit])) + 0.45 * (shock * shock)
            alpha_c = (base_c * (cnt[hit] / (cnt[hit] + 1.5))).clamp_max(0.65).unsqueeze(1)
            self.proto_c[hit] = (1.0 - alpha_c) * self.proto_c[hit] + alpha_c * b_mean

            b_mean_full = self.proto_c.clone()
            b_mean_full[hit] = b_mean
            resid_sq = (x - b_mean_full[assign]).pow(2).sum(dim=1) / float(dim)
            b_vsum = torch.zeros(k, device=x.device, dtype=torch.float32).scatter_add_(0, assign, resid_sq)
            hit2 = cnt >= 2
            if bool(hit2.any()):
                v_unbiased = (b_vsum[hit2] / (cnt[hit2] - 1.0)).clamp_min(1e-10)
                alpha_v = (0.10 * (cnt[hit2] - 1.0) / (cnt[hit2] + 4.0)).clamp(0.02, 0.20)
                self.proto_var[hit2] = (1.0 - alpha_v) * self.proto_var[hit2] + alpha_v * v_unbiased

        med_v = self.proto_var.median().clamp_min(1e-10)
        v_loc = torch.maximum(self.proto_var[assign], med_v)
        uncov_factor = max(25.0, 16.0 * float(dim))
        uncov_mask = d2_min > (uncov_factor * v_loc)
        if int(uncov_mask.sum().item()) >= 2:
            cand = x[uncov_mask]
            for _ in range(min(8, len(cand))):
                d2_c = torch.cdist(cand, self.proto_c).pow(2).min(dim=1).values
                if int((d2_c > float(uncov_factor * med_v.item())).sum().item()) < 2:
                    break
                best_d2, best_j = d2_c.max(dim=0)
                if float(best_d2.item()) <= float(uncov_factor * med_v.item()):
                    break
                pt = cand[best_j : best_j + 1]
                new_w = max(1.0 / float(max(b, len(self.proto_c) + 1)), 0.005)
                if len(self.proto_c) < self.max_prototypes:
                    self.proto_c = torch.cat([self.proto_c, pt], dim=0)
                    self.proto_var = torch.cat([self.proto_var, med_v.view(1)], dim=0)
                    self.proto_w = torch.cat([self.proto_w * (1.0 - new_w), torch.tensor([new_w], device=x.device)], dim=0)
                    self.proto_hits = torch.cat([self.proto_hits, torch.tensor([2.0], device=x.device)], dim=0)
                    self.psi = torch.cat([self.psi, torch.zeros(1, device=x.device)], dim=0)
                    self.w_fake = torch.cat([self.w_fake * (1.0 - new_w), torch.tensor([new_w], device=x.device)], dim=0)
                    self.v_push = torch.cat([self.v_push, (0.70 * med_v).view(1, 1).expand(1, dim)], dim=0)
                else:
                    slot = int(self.proto_w.argmin().item())
                    self.proto_c[slot] = pt[0]
                    self.proto_var[slot] = med_v
                    self.proto_w[slot] = new_w
                    self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
                    self.proto_hits[slot] = 2.0
                    self.psi[slot] = 0.0
                    self.w_fake[slot] = new_w
                    self.v_push[slot] = (0.70 * med_v).expand(dim)

        if len(self.proto_c) >= 2:
            d2_cc = torch.cdist(self.proto_c, self.proto_c).pow(2)
            d2_cc.fill_diagonal_(float("inf"))
            v_pair = torch.maximum(self.proto_var.unsqueeze(1), self.proto_var.unsqueeze(0))
            ratio_cc = d2_cc / (9.0 * float(dim) * v_pair).clamp_min(1e-12)
            if float(ratio_cc.min().item()) < 1.0:
                flat_idx = int(ratio_cc.argmin().item())
                kc = len(self.proto_c)
                a, b_idx = flat_idx // kc, flat_idx % kc
                if a > b_idx:
                    a, b_idx = b_idx, a
                wa, wb = float(self.proto_w[a].item()), float(self.proto_w[b_idx].item())
                wt = max(wa + wb, 1e-12)
                diff_sq = float((self.proto_c[a] - self.proto_c[b_idx]).pow(2).sum().item()) / float(dim)
                self.proto_c[a] = (wa * self.proto_c[a] + wb * self.proto_c[b_idx]) / wt
                self.proto_var[a] = max(
                    (wa * float(self.proto_var[a].item()) + wb * float(self.proto_var[b_idx].item())) / wt
                    + (wa * wb / (wt * wt)) * diff_sq,
                    1e-10,
                )
                self.proto_w[a] = wt
                self.proto_hits[a] = self.proto_hits[a] + self.proto_hits[b_idx]
                self.psi[a] = (wa * self.psi[a] + wb * self.psi[b_idx]) / wt
                self.w_fake[a] = self.w_fake[a] + self.w_fake[b_idx]
                self.v_push[a] = (wa * self.v_push[a] + wb * self.v_push[b_idx]) / wt
                keep = torch.ones(kc, dtype=torch.bool, device=x.device)
                keep[b_idx] = False
                self._filter_atlas(keep)

        if len(self.proto_c) >= 2:
            prune_thresh = min(0.0012, 0.06 / float(len(self.proto_c)))
            alive = (self.proto_w >= prune_thresh) | (self.proto_hits < 8.0)
            if not bool(alive.all()) and int(alive.sum().item()) >= 1:
                self._filter_atlas(alive)


def _resolve_prior_kernel(prior: nn.Module, latent: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
    base_prior = unwrap_prior(prior)
    if getattr(base_prior, "code_law", "table") != "table" or not hasattr(base_prior, "z"):
        return latent, 0.0, False
    transforms = bool(getattr(prior, "transforms_codes", False))
    if not transforms:
        if isinstance(base_prior, MoGParticlePrior):
            sigma_z = float(base_prior.sigma.detach().item())
            return base_prior.means()[indices], sigma_z, True
        return base_prior.z[indices], 0.0, True
    manips = tuple(getattr(prior, "manipulators", ()))
    if (
        len(manips) == 1
        and type(manips[0]).__name__ == "NormalNoiseManipulator"
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
    ):
        return base_prior.z[indices], float(manips[0].sigma), True
    return latent, 0.0, False


def _forward_with_tangent(
    trainer: Any,
    z_in: torch.Tensor,
    tangent: torch.Tensor,
    *,
    sigma_out: Any,
    noise: torch.Generator,
    rows: torch.Tensor,
    need_tangent: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate (G(z_in), J_G(z_in) tangent) via exact 1st-order fwAD with central-secant fallback."""
    if not need_tangent:
        primal = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, z_in, sigma_out, noise, rows=rows),
        )
        return primal, torch.ones_like(primal)

    out_sigma = float(sigma_out) if isinstance(sigma_out, (int, float)) else 0.0
    can_fwad = not bool(getattr(trainer, "_model_uses_rng", False)) and out_sigma == 0.0
    use_bf16 = bool(getattr(trainer, "_in_training_step", False) and getattr(trainer, "_bf16_generator", False) and z_in.device.type == "cuda")
    if can_fwad:
        try:
            with fwAD.dual_level():
                z_dual = fwAD.make_dual(z_in, tangent)
                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        raw_dual = trainer.G(z_dual)
                    primal_raw, jvp_raw = fwAD.unpack_dual(raw_dual)
                    dual_fp32 = fwAD.make_dual(
                        primal_raw.float(),
                        torch.zeros_like(primal_raw, dtype=torch.float32) if jvp_raw is None else jvp_raw.float(),
                    )
                    post_dual = trainer._postprocess(trainer.prior, dual_fp32)
                else:
                    raw_dual = trainer.G(z_dual)
                    post_dual = trainer._postprocess(trainer.prior, raw_dual)
                primal, jvp = fwAD.unpack_dual(post_dual)
            if jvp is None:
                jvp = torch.zeros_like(primal)
            return primal, jvp
        except (RuntimeError, NotImplementedError, TypeError):
            pass

    primal = trainer._postprocess(
        trainer.prior,
        trainer._generate(trainer.G, z_in, sigma_out, noise, rows=rows),
    )
    with torch.no_grad():
        delta = 1e-3
        t_norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        u_dir = tangent / t_norm
        p_plus = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_in.detach() + delta * u_dir, 0.0, noise))
        p_minus = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_in.detach() - delta * u_dir, 0.0, noise))
        jvp_fd = ((p_plus - p_minus) / (2.0 * delta)) * t_norm.view(len(z_in), *([1] * (primal.ndim - 1)))
    return primal, jvp_fd


def t4_arch_step(
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
    """Execute Theory 4B Part A (smooth chart + fwAD saturation pullback) and Part B (concentric condensation)."""
    recipe = trainer.recipe
    reg: T4ArchPriorRegularizer = trainer.prior_regularizer
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)

    real_proj_peek = _project_for_transport(real.detach())
    with torch.no_grad():
        reg.update_atlas(real_proj_peek)

    k_modes = len(reg.proto_c)
    is_point = reg.is_point
    mu_g, sigma_z, can_condense_table = _resolve_prior_kernel(trainer.prior, latent, indices)
    has_trainable_table = hasattr(base_prior, "z") and base_prior.z.requires_grad
    use_part_b = bool(can_condense_table and has_trainable_table and not is_point and k_modes >= 2 and float(recipe.t1_prior_row_lr) > 0.0)

    # When Part B is active, detach prior.z in Part A for clean Division of Labor;
    # on K=1 / point-mass tasks (e.g. rail_escape, gaussian1d), keep prior.z attached!
    if sigma_z > 0.0:
        eps_g = ((latent - mu_g) / sigma_z).detach()
        mu_for_a = mu_g.detach() if use_part_b else mu_g
        z_in = mu_for_a + sigma_z * eps_g
        tangent_in = sigma_z * eps_g
    else:
        eps_g = _deterministic_box_muller(
            indices if indices is not None else torch.arange(b, device=latent.device),
            latent.shape[1],
            latent.dtype,
        ).to(latent.device)
        z_in = latent.detach() if use_part_b else latent
        tangent_in = 0.05 * (eps_g / eps_g.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    need_jvp = not (not is_point and k_modes == 1 and real.ndim == 4 and (real.shape[2] > 16 or real.shape[3] > 16))
    fake_g, dx_jvp_raw = _forward_with_tangent(
        trainer,
        z_in,
        tangent_in,
        sigma_out=sigma_out,
        noise=noise,
        rows=indices,
        need_tangent=need_jvp,
    )
    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(fake_g)

    # Preserve exact RNG / critic call order
    fake_logits = critic(fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:]:
        raise ValueError("generator_real must match the real sample shape")
    if len(real_g) != len(real):
        raise ValueError("RpGAN generator_real must match the real batch size")
    real_logits = critic(real_g)
    loss_gan = trainer.loss.g_loss(fake_logits, real_logits)

    fake_proj = _project_for_transport(fake_g)
    real_proj = _project_for_transport(real_g.detach())
    dx_jvp = _project_for_transport(dx_jvp_raw.detach())
    dim = fake_proj.shape[1]
    t_in_norm = tangent_in.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    unit_jvp = dx_jvp / t_in_norm

    transport = recipe.kinetic_transport_loss(fake_g, real_g) if recipe.kinetic_transport_weight else loss_gan.new_zeros(())
    transport_local = (
        recipe.kinetic_transport_local_loss(fake_g, real_g)
        if recipe.kinetic_transport_local_weight
        else loss_gan.new_zeros(())
    )

    if is_point:
        # Case A: Point-mass / rail-escape target with exact coordinatewise fwAD inverse-attenuation
        c_pt = reg.proto_c[0].to(fake_proj.dtype).unsqueeze(0).expand_as(fake_proj)
        cur_res = float((fake_proj.detach().float() - reg.proto_c[0].unsqueeze(0)).norm(dim=-1).mean().item())
        if cur_res > max(4.0 * reg.res_ema, reg.res_ema + 0.15):
            reg.res_ema = min(cur_res, 10.0)
        else:
            reg.res_ema = 0.90 * reg.res_ema + 0.10 * min(cur_res, 10.0)
        g_scale = max(0.015, min(1.0, reg.res_ema / 0.035))
        wakeup_boost = 1.0
        prior_scale = g_scale

        s_coord = unit_jvp.abs().clamp_min(1e-5)
        s_ref = torch.maximum(
            torch.quantile(s_coord, 0.5, dim=0, keepdim=True),
            torch.full((1, dim), max(0.15 * reg.res_ema, 1e-4), device=fake_proj.device, dtype=s_coord.dtype),
        )
        w_sat_pt = (s_ref / s_coord).clamp(1.0, 12.0).to(fake_proj.dtype)
        loss_ot = (torch.sqrt(1e-6 + w_sat_pt * (fake_proj - c_pt).pow(2)) - 1e-3).mean()
        if has_trainable_table:
            loss_ot = loss_ot + 0.004 * base_prior.z.square().mean()
        loss_iso = fake_proj.new_zeros(())
        loss_env = fake_proj.new_zeros(())
        gan_blend = 0.02 * min(1.0, reg.res_ema / 0.25)
    elif k_modes == 1:
        # Case B: Connected single-mode continuous target (1D/2D Gaussians, DCGAN image distributions)
        y_feydy, cur_res = _feydy_sinkhorn_barycenter(
            fake_proj,
            real_proj,
            iters=6,
            ref_mean=reg.proto_c[0],
            ref_std=reg.proto_var[0].clamp_min(1e-10).sqrt(),
        )
        if cur_res > max(4.0 * reg.res_ema, reg.res_ema + 0.15):
            reg.res_ema = min(cur_res, 10.0)
        else:
            reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(cur_res, 10.0)
        r_std = float(real_proj.float().std(dim=0, unbiased=False).mean().clamp_min(1e-5).item())
        f_std = float(fake_proj.detach().float().std(dim=0, unbiased=False).mean().clamp_min(1e-12).item())
        scale_ratio = f_std / r_std
        wakeup_boost = 5.0 if scale_ratio < 0.5 else (1.5 if scale_ratio < 0.85 else 1.0)
        g_scale = 1.0 if scale_ratio < 0.6 else max(0.015, min(1.0, reg.res_ema / 0.18))
        prior_scale = g_scale

        s_vec = unit_jvp.norm(dim=-1).clamp_min(1e-8)
        s_med = s_vec.median().clamp_min(1e-8)
        w_sat_vec = (s_med / s_vec).clamp(1.0, 6.0).to(fake_proj.dtype) if float(s_med.item()) > 1e-5 else torch.ones(b, device=fake_proj.device, dtype=fake_proj.dtype)
        sc_ref = torch.tensor(0.25 * r_std, device=fake_proj.device, dtype=fake_proj.dtype).clamp_min(1e-4)
        sq_feydy = w_sat_vec * (fake_proj - y_feydy).pow(2).mean(dim=-1)
        loss_ot = (torch.sqrt(sc_ref.pow(2) + sq_feydy) - sc_ref).mean()
        loss_iso = fake_proj.new_zeros(())
        r_env = real_proj.abs().amax(dim=0, keepdim=True) + 3.5 * r_std
        loss_env = F.relu(fake_proj.abs() - r_env).pow(2).mean()
        gan_blend = 0.05 if dim == 1 else 0.15
    else:
        # Case C: Disconnected multi-modal target (K >= 2)
        with torch.no_grad():
            yf = real_proj.float()
            xff = fake_proj.detach().float()
            r_macro = yf.std(dim=0, unbiased=False).mean().clamp_min(1e-4)
            f_std = xff.std(dim=0, unbiased=False).mean().clamp_min(1e-12)
            scale_ratio = float((f_std / r_macro).item())
            sigma_cell_all = reg.proto_var.clamp_min(1e-10).sqrt()
            sigma_cell_med = sigma_cell_all.median().clamp_min(1e-6)
            if sigma_z > 0.0:
                iso_t = max(1.0, min(4.5, float((sigma_z * r_macro / sigma_cell_med).item())))
                if not reg.iso_initialized:
                    reg.iso_ema = iso_t
                    reg.iso_initialized = True
                else:
                    reg.iso_ema = 0.95 * reg.iso_ema + 0.05 * iso_t

            if scale_ratio < 0.35:
                fw = (xff - xff.mean(0, keepdim=True)) / xff.std(0, keepdim=True).clamp_min(1e-12)
                cw = (reg.proto_c - reg.proto_c.mean(0, keepdim=True)) / reg.proto_c.std(0, keepdim=True).clamp_min(1e-6)
                d_xp = torch.cdist(fw, cw, p=2.0)
                eps_s = 0.06
            else:
                d_xp = torch.cdist(xff, reg.proto_c, p=2.0)
                c_scale = d_xp.mean().clamp_min(1e-5)
                d_xp = d_xp / c_scale
                eps_s = max(0.012, min(0.06, 0.25 * reg.res_ema))

            L_xp = -d_xp / eps_s
            log_a = -math.log(float(b))
            log_b = torch.log(reg.proto_w.clamp_min(1e-12))
            u_s = torch.zeros(b, device=xff.device, dtype=torch.float32)
            v_s = torch.zeros(k_modes, device=xff.device, dtype=torch.float32)
            for _ in range(10):
                v_s = log_b - torch.logsumexp(L_xp + u_s.unsqueeze(1), dim=0)
                u_s = log_a - torch.logsumexp(L_xp + v_s.unsqueeze(0), dim=1)
            P_xp = torch.exp(L_xp + u_s.unsqueeze(1) + v_s.unsqueeze(0))
            y_ot_soft = ((float(b) * P_xp) @ reg.proto_c).to(fake_proj.dtype)

            d_raw = torch.cdist(xff, reg.proto_c, p=2.0)
            d_near, near_k_full = d_raw.min(dim=1)
            w_p2f = (reg.proto_w * d_raw.min(dim=0).values).sum()
            f2p = d_near.median()
            cell_ref = 1.5 * math.sqrt(float(dim)) * sigma_cell_med
            cur_res = float(((torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro).item())
            if cur_res > max(4.0 * reg.res_ema, reg.res_ema + 0.20):
                reg.res_ema = min(cur_res, 10.0)
            else:
                reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(cur_res, 10.0)

            if scale_ratio >= 0.65 and indices is not None:
                halo_r = 3.5 * math.sqrt(float(dim)) * sigma_cell_all[near_k_full]
                halo_blend = ((d_near / halo_r.clamp_min(1e-6)).clamp(0.0, 1.0) ** 2).unsqueeze(1).to(fake_proj.dtype)
                v_mu_i = (
                    (reg.proto_var.unsqueeze(1) - reg.v_push).clamp_min(0.0)[near_k_full]
                    if sigma_z > 0.0
                    else reg.proto_var[near_k_full].unsqueeze(1).expand(-1, dim)
                )
                xi_i = _deterministic_box_muller(indices, dim, torch.float32).to(xff.device)
                cnt_f = torch.bincount(near_k_full, minlength=k_modes).float()
                xi_f_sum = torch.zeros((k_modes, dim), device=xff.device, dtype=torch.float32).index_add_(0, near_k_full, xi_i)
                xi_f_mean = torch.zeros((k_modes, dim), device=xff.device, dtype=torch.float32)
                cf2 = cnt_f >= 2
                xi_f_mean[cf2] = xi_f_sum[cf2] / cnt_f[cf2].unsqueeze(1)
                innov = dx_jvp.to(fake_proj.dtype) if sigma_z > 0.0 else 0.0
                c_exact_i = (reg.proto_c[near_k_full] + v_mu_i.sqrt() * (xi_i - xi_f_mean[near_k_full])).to(fake_proj.dtype) + innov
                y_ot = halo_blend * y_ot_soft + (1.0 - halo_blend) * c_exact_i
            else:
                y_ot = y_ot_soft

            s_vec = unit_jvp.norm(dim=-1).clamp_min(1e-8)
            s_med = s_vec.median().clamp_min(1e-8)
            w_sat_vec = (
                (s_med / s_vec).clamp(1.0, 8.0).to(fake_proj.dtype)
                if float(s_med.item()) > 1e-5
                else torch.ones(b, device=xff.device, dtype=fake_proj.dtype)
            )

        sc_med = sigma_cell_med.to(fake_proj.dtype)
        sq_ot = w_sat_vec * (fake_proj - y_ot).pow(2).sum(dim=-1)
        loss_ot = (torch.sqrt(sc_med.pow(2) + sq_ot) - sc_med).mean()

        if sigma_z > 0.0 and float(recipe.t1_iso_weight) > 0.0:
            d_z = mu_for_a.shape[1]
            e1 = eps_g / eps_g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if dim == 2 and d_z == 2:
                m_iso = min(b, 256)
                ms = mu_for_a[:m_iso].detach()
                e1s = e1[:m_iso]
                e2s = torch.stack((-e1s[:, 1], e1s[:, 0]), dim=1)
                z_cat = torch.cat([ms, ms + sigma_z * e1s, ms + sigma_z * e2s], dim=0)
                x_cat = _project_for_transport(trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_cat, 0.0, noise)))
                x_c, x_1, x_2 = x_cat.chunk(3, dim=0)
                dx1, dx2 = x_1 - x_c, x_2 - x_c
                near_sub = torch.cdist(x_c.detach().float(), reg.proto_c, p=2.0).argmin(dim=1)
                v_mode_sub = reg.proto_var[near_sub].to(fake_proj.dtype)
                inv_w_sub = (1.0 / (float(k_modes) * reg.proto_w[near_sub]).clamp_min(1e-4)).sqrt().clamp(0.5, 4.0).to(fake_proj.dtype)
                Sigma_push = dx1.unsqueeze(2) * dx1.unsqueeze(1) + dx2.unsqueeze(2) * dx2.unsqueeze(1)
                eye2 = torch.eye(2, device=xff.device, dtype=fake_proj.dtype).unsqueeze(0)
                t_vp = 0.72 * v_mode_sub
                with torch.no_grad():
                    vp_diag = dx1.detach().float().pow(2) + dx2.detach().float().pow(2)
                    cnt_vp = torch.bincount(near_sub, minlength=k_modes).float()
                    sum_vp = torch.zeros((k_modes, dim), device=xff.device).index_add_(0, near_sub, vp_diag)
                    h_vp = cnt_vp > 0
                    if bool(h_vp.any()):
                        reg.v_push[h_vp] = 0.88 * reg.v_push[h_vp] + 0.12 * (sum_vp[h_vp] / cnt_vp[h_vp].unsqueeze(1))
                    iso_rel = ((Sigma_push.detach().float() / t_vp.detach().float().view(m_iso, 1, 1).clamp_min(1e-10)) - eye2.float()).pow(2).sum(dim=(1, 2)).sqrt().mean()
                    reg.iso_err_ema = 0.92 * reg.iso_err_ema + 0.08 * min(float(iso_rel.item()), 5.0)
                denom_c = (t_vp.sqrt() * sc_med).clamp_min(1e-8).view(m_iso, 1, 1)
                loss_iso = 0.5 * (inv_w_sub * ((Sigma_push - t_vp.view(m_iso, 1, 1) * eye2) / denom_c).pow(2).sum(dim=(1, 2))).mean()
            elif dim == 2 and d_z == 4:
                m_iso = min(b, 96)
                ms = mu_for_a[:m_iso].detach()
                e1s = e1[:m_iso]
                a_q, b_q, c_q, d_q = e1s[:, 0], e1s[:, 1], e1s[:, 2], e1s[:, 3]
                e2s = torch.stack((-b_q, a_q, -d_q, c_q), dim=1)
                e3s = torch.stack((-c_q, d_q, a_q, -b_q), dim=1)
                e4s = torch.stack((-d_q, -c_q, b_q, a_q), dim=1)
                z_cat = torch.cat([ms, ms + sigma_z * e1s, ms + sigma_z * e2s, ms + sigma_z * e3s, ms + sigma_z * e4s], dim=0)
                x_cat = _project_for_transport(trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_cat, 0.0, noise)))
                x_c, x_1, x_2, x_3, x_4 = x_cat.chunk(5, dim=0)
                dxs = [x_1 - x_c, x_2 - x_c, x_3 - x_c, x_4 - x_c]
                M_dx = torch.stack(dxs, dim=-1)
                Sigma_push = M_dx @ M_dx.transpose(1, 2)
                near_sub = torch.cdist(x_c.detach().float(), reg.proto_c, p=2.0).argmin(dim=1)
                v_mode_sub = reg.proto_var[near_sub].to(fake_proj.dtype)
                inv_w_sub = (1.0 / (float(k_modes) * reg.proto_w[near_sub]).clamp_min(1e-4)).sqrt().clamp(0.5, 4.0).to(fake_proj.dtype)
                eye2 = torch.eye(2, device=xff.device, dtype=fake_proj.dtype).unsqueeze(0)
                t_vp = 0.72 * v_mode_sub
                with torch.no_grad():
                    vp_diag = torch.diagonal(Sigma_push.detach().float(), dim1=1, dim2=2)
                    cnt_vp = torch.bincount(near_sub, minlength=k_modes).float()
                    sum_vp = torch.zeros((k_modes, dim), device=xff.device).index_add_(0, near_sub, vp_diag)
                    h_vp = cnt_vp > 0
                    if bool(h_vp.any()):
                        reg.v_push[h_vp] = 0.88 * reg.v_push[h_vp] + 0.12 * (sum_vp[h_vp] / cnt_vp[h_vp].unsqueeze(1))
                    iso_rel = ((Sigma_push.detach().float() / t_vp.detach().float().view(m_iso, 1, 1).clamp_min(1e-10)) - eye2.float()).pow(2).sum(dim=(1, 2)).sqrt().mean()
                    reg.iso_err_ema = 0.92 * reg.iso_err_ema + 0.08 * min(float(iso_rel.item()), 5.0)
                denom_c = (t_vp.sqrt() * sc_med).clamp_min(1e-8).view(m_iso, 1, 1)
                loss_iso = 0.5 * (inv_w_sub * ((Sigma_push - t_vp.view(m_iso, 1, 1) * eye2) / denom_c).pow(2).sum(dim=(1, 2))).mean()
            else:
                m_iso = min(b, 64)
                ms = mu_for_a[:m_iso].detach()
                e1s = e1[:m_iso]
                z_cat = torch.cat([ms, ms + sigma_z * e1s], dim=0)
                x_cat = _project_for_transport(trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_cat, 0.0, noise)))
                x_c, x_1 = x_cat.chunk(2, dim=0)
                dx1 = x_1 - x_c
                near_sub = torch.cdist(x_c.detach().float(), reg.proto_c, p=2.0).argmin(dim=1)
                v_mode_sub = reg.proto_var[near_sub].to(fake_proj.dtype)
                vp_diag = dx1.detach().float().pow(2) * float(d_z)
                vp_sub = dx1.pow(2).mean(dim=-1) * float(d_z)
                with torch.no_grad():
                    cnt_vp = torch.bincount(near_sub, minlength=k_modes).float()
                    sum_vp = torch.zeros((k_modes, dim), device=xff.device).index_add_(0, near_sub, vp_diag)
                    h_vp = cnt_vp > 0
                    if bool(h_vp.any()):
                        reg.v_push[h_vp] = 0.88 * reg.v_push[h_vp] + 0.12 * (sum_vp[h_vp] / cnt_vp[h_vp].unsqueeze(1))
                t_vp = 0.72 * v_mode_sub
                loss_iso = ((vp_sub.clamp_min(1e-12).sqrt() - t_vp.sqrt()) / sc_med.clamp_min(1e-6)).pow(2).mean()
        else:
            loss_iso = fake_proj.new_zeros(())

        r_env = real_proj.abs().amax(dim=0, keepdim=True) + 3.5 * sc_med
        loss_env = F.relu(fake_proj.abs() - r_env).pow(2).mean()
        wakeup_boost = 5.0 if scale_ratio < 0.5 else (1.5 if scale_ratio < 0.85 else 1.0)
        iso_gate = max(0.0, min(1.0, (reg.iso_err_ema - 0.22) / 0.45)) if (sigma_z > 0.0 and dim == 2 and mu_for_a.shape[1] in (2, 4)) else 0.0
        g_scale = 1.0 if scale_ratio < 0.6 else max(0.012, min(1.0, max(reg.res_ema / 0.10, 0.28 * iso_gate)))
        prior_scale = 1.0
        gan_blend = 0.15

    pb_w = float(recipe.t1_pullback_weight)
    loss_g = (
        gan_blend * loss_gan
        + pb_w * loss_ot
        + float(recipe.t1_iso_weight) * loss_iso
        + (2.0 * pb_w) * loss_env
    )
    if not use_part_b and has_trainable_table:
        if not is_point and recipe.prior_reg > 0:
            loss_g = loss_g + recipe.prior_regularization(
                latent,
                regularizer=trainer.prior_regularizer,
                prior=trainer.prior,
            )
        elif recipe.prior_l2 > 0:
            loss_g = loss_g + recipe.prior_l2 * base_prior.z.square().mean()

    prior_manip_loss = None
    if trainer._prior_penalty:
        raw_rows = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        prior_manip_loss = trainer.prior.regularize(
            raw_rows,
            sampled_codes=latent,
            indices=indices,
            generated=fake_g,
            logits=fake_logits,
        )
        loss_g = loss_g + prior_manip_loss
    if recipe.kinetic_transport_weight:
        loss_g = loss_g + transport
    if recipe.kinetic_transport_local_weight:
        loss_g = loss_g + transport_local

    trainer.opt_g.zero_grad()
    loss_g.backward()

    # --- PART B: Kantorovich-Laguerre + Concentric Total-Variance Condensation + Dead-Jacobian Recovery ---
    prior_reg_diag = loss_gan.new_zeros(())
    u_rows = torch.unique(indices) if indices is not None else None
    if use_part_b and u_rows is not None and len(u_rows) > 0:
        g_flags = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)

            n_tot = len(base_prior.z)
            if sigma_z == 0.0 and n_tot <= 256 and dim <= 256:
                with torch.no_grad():
                    c_all = base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z
                    x_all = _project_for_transport(
                        trainer._postprocess(trainer.prior, trainer._generate(trainer.G, c_all, 0.0, noise))
                    ).float()
                    k_all = (torch.cdist(x_all, reg.proto_c, p=2.0).pow(2) - reg.psi.unsqueeze(0)).argmin(dim=1)
                    w_hard_full = torch.bincount(k_all, minlength=k_modes).float() / float(n_tot)
            else:
                w_hard_full = None

            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[u_rows]
            x_u = _project_for_transport(
                trainer._postprocess(trainer.prior, trainer._generate(trainer.G, c_u, 0.0, noise))
            )

            with torch.no_grad():
                xu_det = x_u.detach().float()
                d2_up = torch.cdist(xu_det, reg.proto_c, p=2.0).pow(2)
                k_star = (d2_up - reg.psi.unsqueeze(0)).argmin(dim=1)

                if w_hard_full is None:
                    w_hard_u = torch.bincount(k_star, minlength=k_modes).float() / float(len(u_rows))
                    reg.w_fake = 0.88 * reg.w_fake + 0.12 * w_hard_u
                    w_fb = 0.35 * w_hard_u + 0.65 * reg.w_fake
                else:
                    reg.w_fake = 0.75 * reg.w_fake + 0.25 * w_hard_full
                    w_fb = 0.50 * w_hard_full + 0.50 * reg.w_fake

                k_float = float(k_modes)
                n_eff = float(n_tot) if w_hard_full is not None else float(len(u_rows))
                tv_now = 0.5 * float((reg.proto_w - reg.w_fake).abs().sum().item())
                tv_floor = 0.28 * math.sqrt(k_float / max(n_eff * 8.0, 1.0))
                psi_gate = max(0.0, min(1.0, (tv_now - tv_floor) / max(0.025, tv_floor)))

                r_mac_sq = float(real_proj.float().std(dim=0, unbiased=False).mean().clamp_min(1e-4).pow(2).item())
                eta_scale = min(3.0, max(0.45, float(recipe.t1_laguerre_lr) * k_float))
                eta_psi = psi_gate * eta_scale * float(dim) * r_mac_sq
                reg.psi.mul_(1.0 - 0.008 * psi_gate)
                reg.psi.add_(reg.proto_w - w_fb, alpha=eta_psi)
                reg.psi.sub_(reg.psi.mean())
                psi_cap = 2.5 * float(dim) * r_mac_sq + 0.5
                reg.psi.clamp_(-psi_cap, psi_cap)

                v_mu_star_k = (
                    (reg.proto_var.unsqueeze(1) - reg.v_push).clamp_min(0.0)
                    if sigma_z > 0.0
                    else reg.proto_var.unsqueeze(1).expand(-1, dim)
                )
                xi_raw = _deterministic_box_muller(u_rows, dim, torch.float32).to(xu_det.device)
                cnt_k = torch.bincount(k_star, minlength=k_modes).float()
                xi_sum = torch.zeros((k_modes, dim), device=xu_det.device, dtype=torch.float32).index_add_(0, k_star, xi_raw)
                xi_mean = torch.zeros((k_modes, dim), device=xu_det.device, dtype=torch.float32)
                cnt2 = cnt_k >= 2
                xi_mean[cnt2] = xi_sum[cnt2] / cnt_k[cnt2].unsqueeze(1)
                xi_centered = xi_raw - xi_mean[k_star]

                c_targ = (reg.proto_c[k_star] + v_mu_star_k[k_star].sqrt() * xi_centered).to(x_u.dtype)
                sc_all = reg.proto_var.clamp_min(1e-10).sqrt()
                sc_u = sc_all[k_star]
                dist_u = (xu_det - c_targ.float()).norm(dim=-1)
                part_scale = (dist_u / (1.8 * math.sqrt(float(dim)) * sc_u)).clamp(0.015, 1.0)
                prior_scale = max(0.20, min(1.0, max(reg.res_ema / 0.10, psi_gate)))

            sq_u = (x_u - c_targ).pow(2).sum(dim=-1)
            sc_u_dt = (0.40 * math.sqrt(float(dim)) * sc_u).to(x_u.dtype)
            loss_mu_cond = (torch.sqrt(sc_u_dt.pow(2) + sq_u) - sc_u_dt).mean()
            if sigma_z > 0.0 and len(u_rows) >= 2:
                mu_raw_u = base_prior.z[u_rows]
                mu_std = (mu_raw_u - mu_raw_u.mean(dim=0, keepdim=True)).pow(2).mean(dim=0).add(1e-6).sqrt()
                loss_mu_iso = F.relu(reg.iso_ema - mu_std).pow(2).mean()
            else:
                mu_raw_u = base_prior.z[u_rows]
                loss_mu_iso = x_u.new_zeros(())

            loss_mu_total = loss_mu_cond + 0.12 * loss_mu_iso
            prior_reg_diag = loss_mu_total.detach()
            if base_prior.z.grad is not None:
                base_prior.z.grad.zero_()
            (loss_mu_total * float(len(u_rows))).backward()

            # Coordinatewise Dead-Jacobian latent recovery for particles trapped on flat clamp/saturation walls
            with torch.no_grad():
                if base_prior.z.grad is not None:
                    gu = base_prior.z.grad[u_rows]
                    far_mask_1d = dist_u > (2.8 * math.sqrt(float(dim)) * sc_u)
                    d_z_u = c_u.shape[1]
                    if d_z_u <= 8 and bool(far_mask_1d.any()):
                        far_idx = torch.nonzero(far_mask_1d, as_tuple=False).squeeze(-1)
                        in_idx = torch.nonzero(~far_mask_1d, as_tuple=False).squeeze(-1)[:64]
                        if in_idx.numel() < 4:
                            in_idx = torch.arange(min(64, len(c_u)), device=c_u.device, dtype=torch.long)
                        probe_idx = torch.cat([far_idx, in_idx], dim=0)
                        c_probe = c_u[probe_idx].detach()
                        x_probe = xu_det[probe_idx]
                        out_sgn = torch.where(c_probe >= 0.0, 1.0, -1.0)
                        eye_z = torch.eye(d_z_u, device=c_u.device, dtype=c_u.dtype)
                        c_pert = (c_probe.unsqueeze(1) + (1e-3 * out_sgn).unsqueeze(2) * eye_z.unsqueeze(0)).reshape(-1, d_z_u)
                        x_pert = _project_for_transport(
                            trainer._postprocess(trainer.prior, trainer._generate(trainer.G, c_pert, 0.0, noise))
                        ).float().reshape(len(probe_idx), d_z_u, dim)
                        col_sens = (x_pert - x_probe.unsqueeze(1)).norm(dim=-1) / 1e-3
                        s_far = col_sens[: len(far_idx)]
                        s_in = col_sens[len(far_idx) :]
                        s_ref_col = torch.quantile(s_in, 0.5, dim=0, keepdim=True).clamp_min(1e-6)
                        dead_far = s_far < (0.04 * s_ref_col)
                        if bool(dead_far.any()):
                            mu_ref = c_u[in_idx].detach().mean(dim=0, keepdim=True)
                            z_far = mu_raw_u[far_idx].detach()
                            gu_far = gu[far_idx]
                            g_mag = torch.maximum(gu_far.float().norm(dim=-1, keepdim=True), torch.ones_like(z_far[:, :1], dtype=torch.float32))
                            inward_grad = (torch.sign(z_far - mu_ref).float() * g_mag).to(gu.dtype)
                            gu_far_new = torch.where(dead_far, inward_grad, gu_far)
                            gu_new = gu.clone()
                            gu_new[far_idx] = gu_far_new
                            base_prior.z.grad[u_rows] = gu_new
                            any_dead_far = dead_far.any(dim=1)
                            ps_new = part_scale.clone()
                            ps_new[far_idx] = torch.where(any_dead_far, torch.ones_like(ps_new[far_idx]), ps_new[far_idx])
                            part_scale = ps_new
                    else:
                        far_mask = far_mask_1d.unsqueeze(1)
                        dead_coord = far_mask & (gu.float().abs() <= 1e-9)
                        if bool(dead_coord.any()):
                            live_coord = ~dead_coord
                            live_cnt = live_coord.float().sum(dim=0, keepdim=True).clamp_min(1.0)
                            mu_ref = (mu_raw_u.detach() * live_coord.to(mu_raw_u.dtype)).sum(dim=0, keepdim=True) / live_cnt.to(mu_raw_u.dtype)
                            inward_grad = 0.5 * (mu_raw_u.detach() - mu_ref)
                            base_prior.z.grad[u_rows] = torch.where(dead_coord, inward_grad, gu)
                            any_dead_row = dead_coord.any(dim=1)
                            part_scale = torch.where(any_dead_row, torch.ones_like(part_scale), part_scale)

            if hasattr(trainer.opt_g, "set_sampled_rows"):
                trainer.opt_g.set_sampled_rows(base_prior.z, u_rows)
            if hasattr(trainer.opt_g, "set_row_scales"):
                trainer.opt_g.set_row_scales(base_prior.z, part_scale)
        finally:
            for p, flag in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(flag)
    elif has_trainable_table:
        if base_prior.z.grad is None:
            base_prior.z.grad = torch.zeros_like(base_prior.z)
        if u_rows is not None and hasattr(trainer.opt_g, "set_sampled_rows"):
            trainer.opt_g.set_sampled_rows(base_prior.z, u_rows)

    trainer.policy.after_generator_backward(
        loss_gan=loss_gan.detach(),
        loss_critic=(loss_d - penalty).detach(),
    )

    if hasattr(trainer.opt_g, "bind_t4_modulation"):
        trainer.opt_g.bind_t4_modulation(
            g_scale=g_scale,
            wakeup_boost=wakeup_boost,
            prior_scale=prior_scale,
        )

    if trainer._gradient_response is not None:
        params = list(trainer.G.parameters())
        grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
        gain = trainer._gradient_response.observe(grads)
        for p, m in zip(params, trainer._gradient_response.packet["means"]):
            p.grad = (m / trainer._gradient_response.packet["weight"]).to(p).clone()
        saved_lrs = [grp["lr"] for grp in trainer.opt_g.param_groups]
        try:
            for grp in trainer.opt_g.param_groups:
                if grp.get("role") == "generator":
                    grp["lr"] = grp["lr"] * gain
            trainer.opt_g.step()
            trainer._gradient_response.response["applied_steps"] += 1
        finally:
            for grp, lr0 in zip(trainer.opt_g.param_groups, saved_lrs):
                grp["lr"] = lr0
    else:
        trainer.opt_g.step()

    trainer.policy.after_generator_step()
    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


__all__ = ["T4ArchPriorRegularizer", "t4_arch_step"]
