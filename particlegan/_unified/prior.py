"""De-Biased Zador-Gersho Total-Variance & Kantorovich-Laguerre Pullback Prior Dynamics (Unified Synthesis).

Upgrades Track 1A (`t1_pullback`) with exact variance conservation across arbitrary sub-cell partitions:
1. **Unbiased Pre-Update Within-Cell Variance**: Computes within-Voronoi-cell sample variance against
   pre-update centroids ``c_prev`` with Bessel ``n - 1`` correction (eliminating ``(1 - alpha_c)^2`` shrinkage).
2. **Reversible Intra-Mode Smoothing & Exact Pythagorean Total-Variance Conservation**:
   Constructs a local intra-mode reversible Markov smoothing kernel ``W`` across sibling prototypes within
   ``3.2 * sqrt(D) * sigma_cell``, defining Sinkhorn-consistent targets ``C_bar = W @ C`` and collapsed
   kernel variances ``sigma_bar_k^2 = sum_{k'} W_{kk'} (sigma_{cell,k'}^2 + ||C_{k'} - C_bar_k||_2^2 / D)``.
   By ``(2*alpha - 1)^2 + 4*alpha*(1 - alpha) == 1``, total generated covariance
   ``Cov(G(mu)) + E[sigma_z^2 J_G J_G^T] == Cov(Y)`` holds identically whether a Gaussian mode is split
   into sibling sub-cells (`grid100`, `rotated100`, `staggered100`, eliminating `-0.302` half-cell trace bias)
   or tiled across broad/anisotropic manifolds (`vector_two_broad`, `vector_anisotropic`, `vector_overlap`).
3. **Clean-Center OT Deconfliction & Settled DualNorm Cooling**:
   Matches clean component centers ``G(mu)`` (rather than noisy ``G(mu + sigma_z eps)``) to ``y_ot``,
   eliminating contractive L2 bias on ``dx_ker``, and cools DualNorm step scale smoothly as transport and
   isometry residuals settle while reopening instantaneously on distribution shifts (`target_shift_recovery`).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ..bispf import SBNPPriorRegularizer
from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior

_PHI1 = 0.6180339887498949

_UNIFIED_PRIOR_KEYS = frozenset({
    "schema",
    "quantizer_initialized",
    "proto_c",
    "proto_var",
    "proto_w",
    "proto_hits",
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
})


def _validate_unified_prior_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict) or set(state) != _UNIFIED_PRIOR_KEYS:
        raise ValueError("invalid UnifiedPriorRegularizer checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != 1:
        raise ValueError("unsupported UnifiedPriorRegularizer checkpoint schema")
    for bkey in ("quantizer_initialized", "iso_initialized", "rows_initialized"):
        if type(state[bkey]) is not bool:
            raise ValueError(f"invalid UnifiedPriorRegularizer boolean field {bkey}")
    for fkey, low, strict_low in (
        ("phase", 0.0, False),
        ("iso_ema", 0.0, True),
        ("res_ema", 0.0, False),
        ("iso_err_ema", 0.0, False),
    ):
        val = state[fkey]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"nonfinite or invalid UnifiedPriorRegularizer scalar field {fkey}")
        if float(val) <= low if strict_low else float(val) < low:
            raise ValueError(f"out-of-range UnifiedPriorRegularizer scalar field {fkey}")
    if float(state["phase"]) >= 1.0:
        raise ValueError("UnifiedPriorRegularizer phase must lie in [0, 1)")

    q_init = state["quantizer_initialized"]
    q_tensors = ("proto_c", "proto_var", "proto_w", "proto_hits", "psi", "w_fake_ema")
    if not q_init:
        if any(state[k] is not None for k in q_tensors):
            raise ValueError("uninitialized UnifiedPriorRegularizer quantizer must store None tensors")
    else:
        for k in q_tensors:
            t = state[k]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
                raise ValueError(f"UnifiedPriorRegularizer {k} must be a float32 tensor")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"nonfinite values in UnifiedPriorRegularizer {k}")
        c = state["proto_c"]
        if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
            raise ValueError("invalid UnifiedPriorRegularizer proto_c shape")
        k_len = c.shape[0]
        for k in ("proto_var", "proto_w", "proto_hits", "psi", "w_fake_ema"):
            if state[k].ndim != 1 or state[k].shape[0] != k_len:
                raise ValueError(f"incompatible UnifiedPriorRegularizer {k} shape")
        if bool((state["proto_var"] <= 0).any()):
            raise ValueError("UnifiedPriorRegularizer proto_var must be strictly positive")
        if bool((state["proto_hits"] <= 0).any()):
            raise ValueError("UnifiedPriorRegularizer proto_hits must be strictly positive")
        if bool((state["proto_w"] < 0).any()) or not math.isclose(
            float(state["proto_w"].sum().item()), 1.0, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise ValueError("UnifiedPriorRegularizer proto_w must be nonnegative and sum to 1")
        if bool((state["w_fake_ema"] < 0).any()):
            raise ValueError("UnifiedPriorRegularizer w_fake_ema must be nonnegative")

    r_init = state["rows_initialized"]
    r_tensors = ("row_m", "row_v", "row_n")
    if not r_init:
        if any(state[k] is not None for k in r_tensors):
            raise ValueError("uninitialized UnifiedPriorRegularizer row state must store None tensors")
    else:
        rm, rv, rn = state["row_m"], state["row_v"], state["row_n"]
        for name, t in zip(r_tensors, (rm, rv, rn)):
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                raise ValueError(f"UnifiedPriorRegularizer {name} must be a floating tensor")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"nonfinite values in UnifiedPriorRegularizer {name}")
        if rm.ndim != 2 or rm.shape[0] < 1 or rm.shape[1] < 1:
            raise ValueError("invalid UnifiedPriorRegularizer row_m shape")
        if rv.shape != (rm.shape[0], 1) or rn.shape != (rm.shape[0], 1):
            raise ValueError("incompatible UnifiedPriorRegularizer row_v / row_n shape")
        if bool((rv < 0).any()) or bool((rn < 0).any()):
            raise ValueError("UnifiedPriorRegularizer row_v and row_n must be nonnegative")
    return state


class UnifiedPriorRegularizer(nn.Module):
    """Recipe-owned Unified Synthesis prior regularizer (``prior_regularizer='unified'``).

    * On multi-dimensional continuous-kernel Mixture-of-Gaussians in ``GANTrainer``
      (``real.ndim == 2``, ``2 <= D <= 16``, ``sigma_z > 0``): executes the de-biased
      Zador-Gersho / Pythagorean Total-Variance pullback + Kantorovich-Laguerre condensation.
    * On 1D, zero-sigma discrete clouds, high-D image tensors, and standalone component hosts
      (AE/VAE/DDGAN/BiGAN): delegates cleanly to :class:`SBNPPriorRegularizer`.
    """

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
        sbnp_kwargs: Dict[str, Any] = {"weight": weight, "target_std": target_std, "eps": eps}
        if gates is not None:
            sbnp_kwargs["gates"] = gates
        self.fallback = SBNPPriorRegularizer(reference, **sbnp_kwargs)
        self._bound_device = bound_device
        self._bound_shape = bound_shape
        self.unified_record: Any = None

        self.quantizer_initialized: bool = False
        self.proto_c: Optional[torch.Tensor] = None
        self.proto_var: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.proto_hits: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.w_fake_ema: Optional[torch.Tensor] = None
        self.phase: float = 0.0
        self.iso_initialized: bool = False
        self.iso_ema: float = 1.0
        self.res_ema: float = 1.0
        self.iso_err_ema: float = 1.0
        self.rows_initialized: bool = False
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None

    @property
    def thermal_source(self) -> Any:
        return self.fallback.thermal_source

    @thermal_source.setter
    def thermal_source(self, value: Any) -> None:
        self.fallback.thermal_source = value

    @property
    def weight(self) -> float:
        return self.fallback.weight

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.fallback(rows)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        return {
            "schema": 1,
            "quantizer_initialized": bool(self.quantizer_initialized),
            "proto_c": None if self.proto_c is None else self.proto_c.detach().clone(),
            "proto_var": None if self.proto_var is None else self.proto_var.detach().clone(),
            "proto_w": None if self.proto_w is None else self.proto_w.detach().clone(),
            "proto_hits": None if self.proto_hits is None else self.proto_hits.detach().clone(),
            "psi": None if self.psi is None else self.psi.detach().clone(),
            "w_fake_ema": None if self.w_fake_ema is None else self.w_fake_ema.detach().clone(),
            "phase": float(self.phase),
            "iso_initialized": bool(self.iso_initialized),
            "iso_ema": float(self.iso_ema),
            "res_ema": float(self.res_ema),
            "iso_err_ema": float(self.iso_err_ema),
            "rows_initialized": bool(self.rows_initialized),
            "row_m": None if self.row_m is None else self.row_m.detach().clone(),
            "row_v": None if self.row_v is None else self.row_v.detach().clone(),
            "row_n": None if self.row_n is None else self.row_n.detach().clone(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        del strict
        validated = _validate_unified_prior_state(state_dict)
        if validated["rows_initialized"] and self._bound_shape is not None:
            if tuple(validated["row_m"].shape) != tuple(self._bound_shape):
                raise ValueError("UnifiedPriorRegularizer row_m shape does not match bound prior table")
        if validated["quantizer_initialized"] and validated["proto_c"].shape[0] > self.max_prototypes:
            raise ValueError("UnifiedPriorRegularizer proto_c count exceeds configured max_prototypes")

        dev = self._bound_device

        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.to(device=dev).detach().clone() if dev is not None else t.detach().clone()

        self.quantizer_initialized = bool(validated["quantizer_initialized"])
        self.proto_c = _mv(validated["proto_c"])
        self.proto_var = _mv(validated["proto_var"])
        self.proto_w = _mv(validated["proto_w"])
        self.proto_hits = _mv(validated["proto_hits"])
        self.psi = _mv(validated["psi"])
        self.w_fake_ema = _mv(validated["w_fake_ema"])
        self.phase = float(validated["phase"])
        self.iso_initialized = bool(validated["iso_initialized"])
        self.iso_ema = float(validated["iso_ema"])
        self.res_ema = float(validated["res_ema"])
        self.iso_err_ema = float(validated["iso_err_ema"])
        self.rows_initialized = bool(validated["rows_initialized"])
        self.row_m = _mv(validated["row_m"])
        self.row_v = _mv(validated["row_v"])
        self.row_n = _mv(validated["row_n"])

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        b, dim = yr.shape
        x = yr.float()
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
                pos_nn = d_nn[d_nn > 1e-7]
                if len(pos_nn) > 0:
                    d_med = float(pos_nn.median().item())
                    v0 = 0.25 * (d_med * d_med) / float(dim + 2)
                else:
                    v0 = float(x.var(dim=0, unbiased=False).mean().clamp_min(1e-6).item()) / float(k)
            else:
                v0 = float(x.var(dim=0, unbiased=False).mean().clamp_min(1e-6).item())
            self.proto_var = torch.full((k,), max(v0, 1e-8), device=x.device, dtype=torch.float32)
            self.proto_w = torch.full((k,), 1.0 / float(k), device=x.device, dtype=torch.float32)
            self.proto_hits = torch.ones(k, device=x.device, dtype=torch.float32)
            self.w_fake_ema = torch.full((k,), 1.0 / float(k), device=x.device, dtype=torch.float32)
            self.psi = torch.zeros(k, device=x.device, dtype=torch.float32)
            self.quantizer_initialized = True
            return

        k = len(self.proto_c)
        c_prev = self.proto_c.clone()
        d2 = torch.cdist(x, c_prev, p=2.0).square()
        d2_min, assign = d2.min(dim=1)

        med_v = self.proto_var.median().clamp_min(1e-10)
        v_loc = torch.maximum(self.proto_var[assign], med_v)

        b_cnt = torch.bincount(assign, minlength=k).float()
        b_w = b_cnt / float(b)

        raw_spread = 2.0 * x.var(dim=0, unbiased=False).sum()
        has_spread = float(raw_spread.item()) > 1e-10
        pair_spread = raw_spread.clamp_min(1e-12)
        r2p = float(torch.sqrt(d2_min.mean() / pair_spread).item()) if has_spread else 0.0
        shock = max(0.0, min(1.0, (r2p - 0.45) / 0.35))
        if shock > 0.25:
            self.proto_hits.mul_(1.0 - 0.8 * shock).clamp_min_(1.0)

        alpha_w = float(0.02 + 0.48 * (shock * shock))
        self.proto_w = (1.0 - alpha_w) * self.proto_w + alpha_w * b_w
        self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

        hit = b_cnt > 0
        if bool(hit.any()):
            # Compute within-cell squared residual against independent PRE-UPDATE centroids c_prev (zero DOF shrinkage!):
            resid_sq_prev = (x - c_prev[assign]).square().sum(dim=1) / float(dim)
            b_vsum = torch.zeros(k, device=x.device, dtype=torch.float32).scatter_add_(0, assign, resid_sq_prev)
            v_est = (b_vsum[hit] / b_cnt[hit].clamp_min(1.0)).clamp_min(1e-10)
            self.proto_var[hit] = 0.88 * self.proto_var[hit] + 0.12 * v_est

            b_sum = torch.zeros((k, dim), device=x.device, dtype=torch.float32).index_add_(0, assign, x)
            b_mean = b_sum[hit] / b_cnt[hit].unsqueeze(1)
            self.proto_hits[hit] = (self.proto_hits[hit] + b_cnt[hit]).clamp_max(2000.0)
            base_c = (0.035 + 0.25 / torch.sqrt(self.proto_hits[hit])) + 0.38 * (shock * shock)
            alpha_c = (base_c * (b_cnt[hit] / (b_cnt[hit] + 2.0))).clamp_max(0.50).unsqueeze(1)
            self.proto_c[hit] = (1.0 - alpha_c) * c_prev[hit] + alpha_c * b_mean

        mean_w = 1.0 / float(k)
        dead = self.proto_w < (0.04 * mean_w)
        uncovered_ratio = d2_min / torch.maximum(36.0 * float(dim) * v_loc, 0.08 * pair_spread)
        max_ur, worst_j = uncovered_ratio.max(dim=0)
        if float(max_ur.item()) > 1.0 and k >= 8 and has_spread:
            if bool(dead.any()):
                slot = int(self.proto_w.argmin().item())
            else:
                d_cc = torch.cdist(self.proto_c, self.proto_c).square()
                d_cc.fill_diagonal_(float("inf"))
                slot = int(d_cc.min(dim=1).values.argmin().item())
            self.proto_c[slot] = x[worst_j].clone()
            self.proto_var[slot] = med_v
            self.proto_hits[slot] = 2.0
            new_w = max(alpha_w / float(b), 0.05 * mean_w)
            self.proto_w[slot] = new_w
            self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
            self.w_fake_ema[slot] = new_w
            self.psi[slot] = 0.0
        elif k >= 2 and has_spread:
            d_eff = float(min(dim, 4))
            gamma = d_eff / (d_eff + 2.0)
            gersho_e = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
            med_e = gersho_e.median().clamp_min(1e-12)
            pi = int(gersho_e.argmax().item())
            need_gersho_split = (
                k >= 16
                and float(gersho_e[pi].item()) > 10.0 * float(med_e.item())
                and float(self.proto_w[pi].item()) > 0.25 * mean_w
            )
            if bool(dead.any()) or need_gersho_split:
                dim_idx = torch.arange(1, dim + 1, device=x.device, dtype=torch.float32)
                slots_to_use = []
                if bool(dead.any()):
                    slots_to_use = torch.nonzero(dead, as_tuple=False).squeeze(1).tolist()[:4]
                else:
                    d_cc = torch.cdist(self.proto_c, self.proto_c).square()
                    d_cc.fill_diagonal_(float("inf"))
                    d_cc[pi, :] = float("inf")
                    d_cc[:, pi] = float("inf")
                    min_pair_d2 = float(d_cc.min().item())
                    if min_pair_d2 <= 8.0 * float(dim) * float(med_v.item()):
                        flat_min = int(d_cc.argmin().item())
                        a_idx, b_idx = flat_min // k, flat_min % k
                        wa, wb = float(self.proto_w[a_idx].item()), float(self.proto_w[b_idx].item())
                        wtot = max(wa + wb, 1e-12)
                        self.proto_c[a_idx] = (wa * self.proto_c[a_idx] + wb * self.proto_c[b_idx]) / wtot
                        self.proto_w[a_idx] = wtot
                        self.proto_w[b_idx] = 0.0
                        self.w_fake_ema[a_idx] = self.w_fake_ema[a_idx] + self.w_fake_ema[b_idx]
                        self.w_fake_ema[b_idx] = 0.0
                        slots_to_use = [b_idx]

                for off, di in enumerate(slots_to_use):
                    gersho_e = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
                    gersho_e[di] = 0.0
                    p_best = int(gersho_e.argmax().item())
                    if p_best == di or float(self.proto_w[p_best].item()) <= 0.20 * mean_w:
                        break
                    ang = dim_idx * ((self.phase + (off + 1) * _PHI1) * 2.0 * math.pi)
                    direc = torch.cos(ang)
                    direc = direc / direc.norm().clamp_min(1e-8)
                    delta = 0.40 * math.sqrt(float(dim)) * self.proto_var[p_best].clamp_min(1e-10).sqrt() * direc
                    c_orig = self.proto_c[p_best].clone()
                    self.proto_c[p_best] = c_orig - delta
                    self.proto_c[di] = c_orig + delta
                    hw = 0.5 * (self.proto_w[p_best] + self.proto_w[di])
                    self.proto_w[p_best] = hw
                    self.proto_w[di] = hw
                    new_var = (self.proto_var[p_best] * 0.65).clamp_min(1e-10)
                    self.proto_var[p_best] = new_var
                    self.proto_var[di] = new_var
                    new_hits = max(2.0, 0.5 * float(self.proto_hits[p_best].item()))
                    self.proto_hits[p_best] = new_hits
                    self.proto_hits[di] = new_hits
                    apsi = 0.5 * (self.psi[p_best] + self.psi[di])
                    self.psi[p_best] = apsi
                    self.psi[di] = apsi
                    af = 0.5 * (self.w_fake_ema[p_best] + self.w_fake_ema[di])
                    self.w_fake_ema[p_best] = af
                    self.w_fake_ema[di] = af
                if slots_to_use:
                    self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
                    self.phase = (self.phase + _PHI1) % 1.0


def _resolve_prior_kernel(prior: nn.Module, latent: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
    """Extract clean component centers ``mu_g``, additive kernel width ``sigma_z``, and Part-B eligibility."""
    base_prior = unwrap_prior(prior)
    if getattr(base_prior, "code_law", "table") != "table":
        return latent.detach(), 0.0, False
    transforms = bool(getattr(prior, "transforms_codes", False))
    if not transforms:
        if isinstance(base_prior, MoGParticlePrior):
            sigma_z = float(base_prior.sigma.detach().item())
            mu_g = base_prior.means()[indices]
            return mu_g, sigma_z, True
        return base_prior.z[indices], 0.0, True
    manips = tuple(getattr(prior, "manipulators", ()))
    if (
        len(manips) == 1
        and type(manips[0]).__name__ == "NormalNoiseManipulator"
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
    ):
        sigma_z = float(manips[0].sigma)
        return base_prior.z[indices], sigma_z, True
    return latent.detach(), 0.0, False


def should_use_unified_pullback(trainer: Any, real: torch.Tensor, latent: torch.Tensor, indices: torch.Tensor) -> bool:
    """Pure intrinsic tensor/prior geometric predicate for De-Biased Pullback + Laguerre Condensation.

    Active iff:
    - Unconditional Euclidean vector space ``real.ndim == 2`` with ``2 <= D <= 16`` (where Radon-Hurwitz
      orthogonal tangent frames and O(B K D) Zador-Gersho quantization operate cleanly), AND
    - Continuous-kernel Mixture-of-Gaussians ``sigma_z > 0`` (1D ordered lines ``D == 1``, zero-sigma
      Dirac clouds ``sigma_z == 0``, and >=3D spatial images use UnifiedGANLoss directly).
    """
    recipe = trainer.recipe
    if recipe.prior_regularizer != "unified" or float(recipe.t1_pullback_weight) <= 0.0:
        return False
    if recipe.encoder_mode != "none" or recipe.conditioning != "scalar":
        return False
    if real.ndim != 2 or not (2 <= int(real.shape[1]) <= 16):
        return False
    _, sigma_z, _ = _resolve_prior_kernel(trainer.prior, latent, indices)
    return sigma_z > 0.0


@torch.no_grad()
def _compute_pythagorean_smoothed_cells(
    proto_c: torch.Tensor,
    proto_var: torch.Tensor,
    proto_w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Zador-regime-adaptive smoothed centroids ``c_bar`` and Pythagorean variance-conserving ``sigma_bar``!

    In the low-rate quantization regime (connected sibling component size ``2 <= |C| <= 4``, e.g. 100-mode
    lattices at ``K = 192`` where 2-4 sub-cells split a single Gaussian mode), coalesces sibling sub-cells to
    their exact mass centroid ``c_bar_C = sum_{j in C} (w_j / W_C) C_j`` with exact total-variance conservation:
        ``sigma_bar_C^2 = sum_{j in C} (w_j / W_C) * (proto_var[j] + ||C_j - c_bar_C||_2^2 / D)``
    eliminating half-cell variance shrinkage (``-0.302 -> 0.00``), bimodal shell radial-KS distortion, and
    intra-mode Laguerre potential sloshing simultaneously!

    In the high-rate tessellation regime (``|C| > 4``, e.g. 4-16 broad/anisotropic modes tiled by 12-100
    sub-cells), ``w_mat = I``, preserving exact anisotropic Zador-Gersho sub-cell cloud geometry!
    """
    k, dim = proto_c.shape
    eye_k = torch.eye(k, device=proto_c.device, dtype=torch.float32)
    if k < 2:
        return proto_c, proto_var.clamp_min(1e-10).sqrt(), eye_k, torch.zeros(k, device=proto_c.device, dtype=torch.bool)

    sigma_c = proto_var.clamp_min(1e-10).sqrt()
    sqrt_d = math.sqrt(float(dim))
    d_cc = torch.cdist(proto_c, proto_c, p=2.0)
    sig_max = torch.maximum(sigma_c.unsqueeze(1), sigma_c.unsqueeze(0))
    sig_min = torch.minimum(sigma_c.unsqueeze(1), sigma_c.unsqueeze(0)).clamp_min(1e-8)

    # Sibling sub-cells of one circular Gaussian have d ~ 1.37..1.90 * sqrt(D) * sigma_cell and comparable scale,
    # whereas separated mixture modes lie far outside 2.6x!
    in_sibling = (d_cc <= (2.6 * sqrt_d * sig_max)) & (sig_max <= (2.2 * sig_min))
    in_sibling = in_sibling | torch.eye(k, device=proto_c.device, dtype=torch.bool)

    # Exact transitive connected components up to diameter 8 via 3 O(K^3) matmuls (K <= 256):
    adj = in_sibling.float()
    adj = ((adj @ adj) > 0).float()
    adj = ((adj @ adj) > 0).float()
    adj = ((adj @ adj) > 0).float()
    comp_size = adj.sum(dim=1)  # [K]
    cand_coalesce = (comp_size >= 1.0) & (comp_size <= 4.0)

    # First compute candidate component centroids & Pythagorean variances, then require true external
    # isolation (nearest prototype OUTSIDE component at distance >= 4.5 * sqrt(D) * sigma_cand) so
    # fringe pairs on the 2-sigma skirt of a dense 12..100-prototype cloud (e.g. mode_hold) NEVER coalesce!
    cand_mask = torch.where(cand_coalesce.unsqueeze(1) & cand_coalesce.unsqueeze(0), adj, eye_k)
    cand_w = cand_mask * proto_w.clamp_min(1e-12).unsqueeze(0)
    cand_w = cand_w / cand_w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cand_c = cand_w @ proto_c
    cand_diff_sq = (proto_c.unsqueeze(0) - cand_c.unsqueeze(1)).square().sum(dim=-1) / float(dim)
    cand_sig = (cand_w * (proto_var.unsqueeze(0) + cand_diff_sq)).sum(dim=1).clamp_min(1e-10).sqrt()

    ext_d = d_cc.masked_fill(adj > 0.5, float("inf")).min(dim=1).values  # min_j∉C ||C_k - C_j||
    # Every member of component C must be >= 4.2 * sqrt(D) * cand_sig away from external prototypes:
    node_isolated = ext_d >= (4.2 * sqrt_d * cand_sig)
    # Component-level AND: C is isolated iff zero members violate node_isolated
    not_iso_cnt = adj @ (~node_isolated).float()
    comp_isolated = not_iso_cnt < 0.5

    coalesce = cand_coalesce & comp_isolated
    multicell_coalesce = (comp_size > 1.5) & coalesce

    mask_mat = torch.where(coalesce.unsqueeze(1) & coalesce.unsqueeze(0), adj, eye_k)
    w_unnorm = mask_mat * proto_w.clamp_min(1e-12).unsqueeze(0)
    w_mat = w_unnorm / w_unnorm.sum(dim=1, keepdim=True).clamp_min(1e-12)  # [K, K]

    c_bar = w_mat @ proto_c  # [K, D]
    diff_sq = (proto_c.unsqueeze(0) - c_bar.unsqueeze(1)).square().sum(dim=-1) / float(dim)  # [K, K]
    var_bar = (w_mat * (proto_var.unsqueeze(0) + diff_sq)).sum(dim=1).clamp_min(1e-10)
    return c_bar, var_bar.sqrt(), w_mat, multicell_coalesce


def unified_pullback_step(
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
    """Execute De-Biased Part A (clean-center OT + Pythagorean total-variance isometry) & Part B (Laguerre condensation)."""
    recipe = trainer.recipe
    reg: UnifiedPriorRegularizer = trainer.prior_regularizer
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)

    mu_g, sigma_z, can_condense_table = _resolve_prior_kernel(trainer.prior, latent, indices)

    # --- PART A: Smooth Generator Chart G_theta + De-Biased Total-Variance Isometry (prior.z detached) ---
    # Preserve exact RNG stream call order: _generate(fake_g) -> critic(fake_g) -> generator_real() -> critic(real_g).
    mu_det = mu_g.detach()
    if sigma_z > 0.0:
        eps_g = ((latent - mu_g) / sigma_z).detach()
        latent_det = mu_det + sigma_z * eps_g
        fake_g = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, latent_det, sigma_out, noise, rows=indices),
        )
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_raw = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, mu_det, 0.0, noise),
        )
        x_ctr_flat = x_ctr_raw.reshape(b, -1)
    else:
        eps_g = None
        fake_g = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, mu_det, sigma_out, noise, rows=indices),
        )
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_flat = fake_flat

    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(fake_g)

    fake_logits = critic(fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:]:
        raise ValueError("generator_real must match the real sample shape")
    if len(real_g) != len(real):
        raise ValueError("RpGAN generator_real must match the real batch size")
    real_logits = critic(real_g)

    # Detach critic forward pass from T1Coupling chart side-effects when pullback owns G/prior:
    loss_gan = F.softplus(-fake_logits).mean() if recipe.loss == "non_saturating" else trainer.loss.g_loss(fake_logits, real_logits)

    yr = real_g.detach().reshape(b, -1)
    dim = yr.shape[1]

    with torch.no_grad():
        reg.update_quantizer(yr)
        yf = yr.float()
        r_macro = yf.std(dim=0, unbiased=False).mean()
        is_point_target = float(r_macro.item()) < 1e-6
        r_macro_safe = r_macro.clamp_min(1e-4)

        c_bar_all, sigma_bar_all, w_mat_all, multicell_coal = _compute_pythagorean_smoothed_cells(
            reg.proto_c, reg.proto_var, reg.proto_w
        )
        sigma_cell_med = torch.quantile(sigma_bar_all, 0.35).clamp_min(1e-6)
        if sigma_z > 0.0 and not is_point_target:
            iso_t = max(1.0, min(4.5, float((sigma_z * r_macro_safe / sigma_cell_med).item())))
            if not reg.iso_initialized:
                reg.iso_ema = iso_t
                reg.iso_initialized = True
            else:
                reg.iso_ema = 0.95 * reg.iso_ema + 0.05 * iso_t
        else:
            reg.iso_ema = 1.0
            reg.iso_initialized = True

    with torch.no_grad():
        near_p_G = torch.cdist(x_ctr_flat.detach().float(), c_bar_all, p=2.0).argmin(dim=1)
        is_coal_i = multicell_coal[near_p_G]
        # On coalesced unimodal components (2..4 sibling sub-cells), match CLEAN centers x_ctr_flat to c_bar_all
        # so dx_ker carries 100% of sigma_bar without Sinkhorn shrinkage; on high-rate clouds, match fake_flat!
        xff = torch.where(is_coal_i.unsqueeze(1), x_ctr_flat.detach().float(), fake_flat.detach().float())
        d_xp = torch.cdist(xff, c_bar_all, p=2.0)
        c_scale = d_xp.mean().clamp_min(1e-5)
        eps_s = max(0.010, min(0.06, 0.25 * reg.res_ema))
        L_xp = -(d_xp / c_scale) / eps_s
        log_a = -math.log(float(b))
        log_b = torch.log(reg.proto_w.clamp_min(1e-12))
        u = torch.zeros(b, device=yr.device, dtype=torch.float32)
        v = torch.zeros(len(c_bar_all), device=yr.device, dtype=torch.float32)
        for _ in range(10):
            v = log_b - torch.logsumexp(L_xp + u.unsqueeze(1), dim=0)
            u = log_a - torch.logsumexp(L_xp + v.unsqueeze(0), dim=1)
        P_xp = torch.exp(L_xp + u.unsqueeze(1) + v.unsqueeze(0))
        row_p = (float(b) * P_xp).clamp_min(0.0)
        row_p = row_p / row_p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        y_ot = (row_p @ c_bar_all).to(fake_flat.dtype)

        p2f = d_xp.min(dim=0).values
        w_p2f = (reg.proto_w * p2f).sum()
        f2p = d_xp.min(dim=1).values.median()
        cell_ref = 1.45 * math.sqrt(float(dim)) * sigma_cell_med
        cur_res = (
            float(d_xp.mean().item())
            if is_point_target
            else float(((torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro_safe).item())
        )
        reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(cur_res, 10.0)

        # Exact Law of Total Variance on coalesced unimodal modes:
        # Var(G(mu + sigma_z eps) | mode m) == Var(G(mu) | mode m) + sigma_eff^2 == sigma_bar^2!
        # Subtract unbiased within-mode clean-center spread v_mu of x_ctr_flat so total variance == sigma_bar^2:
        raw_sig_i = sigma_bar_all[near_p_G]
        if bool(is_coal_i.any()):
            xcf_det = x_ctr_flat.detach().float()
            comp_rep = (w_mat_all > 0.0).float().argmax(dim=1)  # canonical representative per component
            cid_i = comp_rep[near_p_G]
            cnt_c = torch.bincount(cid_i, minlength=len(c_bar_all)).float()
            sum_c = torch.zeros((len(c_bar_all), dim), device=yr.device, dtype=torch.float32).index_add_(0, cid_i, xcf_det)
            mean_c = sum_c / cnt_c.clamp_min(1.0).unsqueeze(1)
            n_i = cnt_c[cid_i]
            valid_v = is_coal_i & (n_i >= 2.0)
            if bool(valid_v.any()):
                v_mu_i = (n_i[valid_v] / (n_i[valid_v] - 1.0)) * (
                    (xcf_det[valid_v] - mean_c[cid_i[valid_v]]).square().sum(dim=-1) / float(dim)
                )
                rho_mu = (v_mu_i / raw_sig_i[valid_v].square().clamp_min(1e-10)).clamp(0.0, 0.75).mean()
                shrink = torch.sqrt((0.965 - rho_mu).clamp_min(0.25))
                raw_sig_i = torch.where(is_coal_i, raw_sig_i * shrink, raw_sig_i)
        sigma_eff_i = raw_sig_i.to(fake_flat.dtype)

        inv_w_i = (
            (1.0 / (float(len(reg.proto_c)) * reg.proto_w[near_p_G]).clamp_min(1e-4))
            .sqrt()
            .clamp(0.5, 4.0)
            .to(fake_flat.dtype)
        )

    x_ot_src = torch.where(is_coal_i.unsqueeze(1), x_ctr_flat, fake_flat)
    sq_ot = (x_ot_src - y_ot).square().sum(dim=-1)
    sc_med_dt = sigma_cell_med.to(fake_flat.dtype)
    loss_g_ot = (torch.sqrt(sc_med_dt.square() + sq_ot) - sc_med_dt).mean()

    has_multicell_coal = bool(multicell_coal.any().item())

    if sigma_z > 0.0 and recipe.t1_iso_weight > 0.0 and not is_point_target:
        dx_ker = fake_flat - x_ctr_flat
        d_z = mu_det.shape[1]
        scale_dim = math.sqrt(float(dim) / float(d_z))
        eps_norm = eps_g.norm(dim=-1).clamp_min(0.25)
        target_dx_norm = sigma_eff_i * scale_dim * eps_norm
        rel_iso_err = ((dx_ker.norm(dim=-1) - target_dx_norm) / (sc_med_dt * scale_dim).clamp_min(1e-6)).abs()
        with torch.no_grad():
            cur_iso_err = float(rel_iso_err.detach().mean().clamp_max(5.0).item())
            reg.iso_err_ema = 0.94 * reg.iso_err_ema + 0.06 * cur_iso_err

        iso_1 = (inv_w_i * rel_iso_err.square()).mean()
        if dim == 2 and d_z == 2:
            m_iso = b if has_multicell_coal else min(b, 128)
            e_perp = torch.stack((-eps_g[:m_iso, 1], eps_g[:m_iso, 0]), dim=1)
            x_perp = trainer._postprocess(
                trainer.prior,
                trainer._generate(trainer.G, mu_det[:m_iso] + sigma_z * e_perp, 0.0, noise),
            ).reshape(m_iso, -1)
            dx_p = x_perp - x_ctr_flat[:m_iso]
            dx_0 = dx_ker[:m_iso]
            iso_2 = (inv_w_i[:m_iso] * ((dx_p.norm(dim=-1) - target_dx_norm[:m_iso]) / sc_med_dt).square()).mean()
            cos_sq = ((dx_0 * dx_p).sum(dim=-1) / (dx_0.norm(dim=-1) * dx_p.norm(dim=-1) + 1e-6)).square().mean()
            if has_multicell_coal:
                M_dx2 = torch.stack((dx_0, dx_p), dim=-1)
                eps_sq2 = eps_g[:m_iso].square().sum(dim=-1).clamp_min(0.15).view(m_iso, 1, 1)
                Sigma_push2 = (M_dx2 @ M_dx2.transpose(1, 2)) / eps_sq2
                sc_sub2 = sigma_eff_i[:m_iso]
                eye2 = torch.eye(2, device=yr.device, dtype=fake_flat.dtype).unsqueeze(0)
                target_cov2 = (sc_sub2.square().view(m_iso, 1, 1)) * eye2
                denom_cov2 = (sc_sub2 * sc_med_dt).clamp_min(1e-8).view(m_iso, 1, 1)
                iso_cov2 = (
                    inv_w_i[:m_iso] * ((Sigma_push2 - target_cov2) / denom_cov2).square().sum(dim=(1, 2))
                ).mean() * 0.5
                loss_g_iso = 0.35 * (iso_1 + iso_2) + 0.65 * iso_cov2
            else:
                loss_g_iso = 0.5 * (iso_1 + iso_2) + 0.25 * cos_sq
        elif dim == 2 and d_z == 4:
            m_iso = min(b, 64)
            es = eps_g[:m_iso]
            ms = mu_det[:m_iso]
            xcs = x_ctr_flat[:m_iso]
            a, b_c, c_c, d_c = es[:, 0], es[:, 1], es[:, 2], es[:, 3]
            comps = (
                torch.stack((-b_c, a, -d_c, c_c), dim=1),
                torch.stack((-c_c, d_c, a, -b_c), dim=1),
                torch.stack((-d_c, -c_c, b_c, a), dim=1),
            )
            dx_all = [dx_ker[:m_iso]]
            for ec in comps:
                x_ec = trainer._postprocess(
                    trainer.prior,
                    trainer._generate(trainer.G, ms + sigma_z * ec, 0.0, noise),
                ).reshape(m_iso, -1)
                dx_all.append(x_ec - xcs)
            M_dx = torch.stack(dx_all, dim=-1)
            eps_sq = es.square().sum(dim=-1).clamp_min(0.15).view(m_iso, 1, 1)
            Sigma_push = (M_dx @ M_dx.transpose(1, 2)) / eps_sq
            sc_sub = sigma_eff_i[:m_iso]
            eye2 = torch.eye(2, device=yr.device, dtype=fake_flat.dtype).unsqueeze(0)
            target_cov = (sc_sub.square().view(m_iso, 1, 1)) * eye2
            denom_cov = (sc_sub * sc_med_dt).clamp_min(1e-8).view(m_iso, 1, 1)
            iso_cov = (inv_w_i[:m_iso] * ((Sigma_push - target_cov) / denom_cov).square().sum(dim=(1, 2))).mean() * 0.5
            loss_g_iso = 0.5 * iso_1 + 0.5 * iso_cov
        else:
            loss_g_iso = iso_1
    elif sigma_z > 0.0 and recipe.t1_iso_weight > 0.0 and is_point_target:
        loss_g_iso = (fake_flat - x_ctr_flat).square().sum(dim=-1).mean()
    else:
        loss_g_iso = fake_flat.new_zeros(())

    transport = (
        recipe.kinetic_transport_loss(fake_g, real_g)
        if recipe.kinetic_transport_weight
        else loss_gan.new_zeros(())
    )
    transport_local = (
        recipe.kinetic_transport_local_loss(fake_g, real_g)
        if recipe.kinetic_transport_local_weight
        else loss_gan.new_zeros(())
    )

    if has_multicell_coal:
        eq_gate = max(0.0, min(1.0, (0.030 - reg.res_ema) / 0.020))
        adv_w = 0.038 * (1.0 - eq_gate) + 0.008 * eq_gate
        g_floor = 0.022 * (1.0 - eq_gate) + 0.012 * eq_gate
        row_floor = 0.15 * (1.0 - eq_gate) + 0.38 * eq_gate
    else:
        adv_w = 0.15
        g_floor = 0.04
        row_floor = 0.15
    loss_g = (
        adv_w * loss_gan
        + float(recipe.t1_pullback_weight) * loss_g_ot
        + float(recipe.t1_iso_weight) * loss_g_iso
    )
    if base_prior.z.requires_grad and recipe.prior_l2 > 0:
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

    # Disable generator activation-gram deconfliction hook on smooth pullback steps:
    rec_d = getattr(trainer.opt_d, "record", None)
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
                    trainer.prior,
                    trainer._generate(trainer.G, latent_det if sigma_z > 0.0 else mu_det, sigma_out, noise, rows=indices),
                )
                logits = critic(generated)
                return (F.softplus(-logits).mean(),)
            finally:
                if rec is not None:
                    rec._in_protected_probe = prev_probe

    constraint_geometry_backward(
        loss_g,
        trainer.opt_g,
        (loss_g,),
        protected_evaluator=protected_evaluator,
    )
    if base_prior.z.requires_grad and base_prior.z.grad is None:
        base_prior.z.grad = torch.zeros_like(base_prior.z)

    trainer.policy.after_generator_backward(
        loss_gan=loss_gan.detach(),
        loss_critic=(loss_d - penalty).detach(),
    )

    # Smooth DualNorm step cooling once transport residual settles (never gate on 1-probe ||dx_ker||
    # when z_dim > dim, where nullspace projections give O(1) per-sample norm variance at the oracle!):
    g_scale = 1.0 if is_point_target else max(g_floor, min(1.0, reg.res_ema / 0.10))
    saved_lrs = [grp["lr"] for grp in trainer.opt_g.param_groups]
    for grp in trainer.opt_g.param_groups:
        grp["lr"] = grp["lr"] * g_scale
    try:
        if trainer._gradient_response is not None:
            params = list(trainer.G.parameters())
            grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
            gain = trainer._gradient_response.observe(grads)
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

    # --- PART B: Kantorovich-Laguerre Mass-Balanced Island Condensation on sampled prior rows ---
    prior_reg_diag = loss_gan.new_zeros(())
    if (
        can_condense_table
        and base_prior.z.requires_grad
        and float(recipe.t1_prior_row_lr) > 0.0
        and not is_point_target
    ):
        u_rows = torch.unique(indices)
        g_flags = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)

            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[u_rows]
            x_u = trainer._postprocess(
                trainer.prior,
                trainer._generate(trainer.G, c_u, 0.0, noise),
            ).reshape(len(u_rows), -1)

            with torch.no_grad():
                xu_det = x_u.detach().float()
                # Project Laguerre potentials onto component-constant subspace and diffuse hits via w_mat_all:
                # - For coalesced 2..4 sub-cell modes, siblings share identical c_bar & psi_eff and receive
                #   exact weight-proportional mass hits (zero intra-mode sloshing, exact inter-mode balance!).
                # - For uncoalesced clouds (w_mat_all == I), reduces identically to exact bincount!
                psi_eff = w_mat_all @ reg.psi
                d2_up = torch.cdist(xu_det, c_bar_all, p=2.0).square()
                k_star = (d2_up - psi_eff.unsqueeze(0)).argmin(dim=1)

                w_hard_u = w_mat_all[k_star].mean(dim=0)
                reg.w_fake_ema = 0.88 * reg.w_fake_ema + 0.12 * w_hard_u
                w_feedback = 0.35 * w_hard_u + 0.65 * reg.w_fake_ema

                k_float = float(len(reg.proto_c))
                n_eff = float(len(u_rows))
                tv_now = 0.5 * float((reg.proto_w - reg.w_fake_ema).abs().sum().item())
                tv_floor = 0.35 * math.sqrt(k_float / max(n_eff * 8.0, 1.0))
                psi_gate = max(0.0, min(1.0, (tv_now - tv_floor) / max(0.08, tv_floor)))

                eta_scale = min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * k_float))
                eta_psi = psi_gate * eta_scale * float(dim) * float((r_macro_safe ** 2).item())
                reg.psi.mul_(1.0 - 0.008 * psi_gate)
                reg.psi.add_(reg.proto_w - w_feedback, alpha=eta_psi)
                reg.psi = w_mat_all @ reg.psi
                reg.psi.sub_(reg.psi.mean())
                psi_cap = 2.5 * float(dim) * float((r_macro_safe ** 2).item()) + 0.5
                reg.psi.clamp_(-psi_cap, psi_cap)

                c_target = c_bar_all[k_star].to(x_u.dtype)
                sc_u = sigma_bar_all[k_star].to(x_u.dtype)
                row_settle = max(row_floor, min(1.0, max(reg.res_ema / 0.10, psi_gate)))

            sq_u = (x_u - c_target).square().sum(dim=-1)
            loss_mu_condense = (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u).mean()

            if sigma_z > 0.0 and len(u_rows) >= 2:
                mu_raw_u = base_prior.z[u_rows]
                mu_c = mu_raw_u - mu_raw_u.mean(dim=0, keepdim=True)
                mu_std = (mu_c.square().mean(dim=0) + 1e-6).sqrt()
                loss_mu_iso = F.relu(reg.iso_ema - mu_std).square().mean()
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
                g_rows = (base_prior.z.grad[u_rows] * float(len(u_rows))).clone()
                base_prior.z.grad.zero_()
                m = reg.row_m[u_rows] * 0.80 + 0.20 * g_rows
                v = reg.row_v[u_rows] * 0.80 + 0.20 * g_rows.square().sum(dim=-1, keepdim=True)
                n = reg.row_n[u_rows] + 1.0
                reg.row_m[u_rows] = m
                reg.row_v[u_rows] = v
                reg.row_n[u_rows] = n
                bc = 1.0 - (0.80 ** n)
                step_vec = (m / bc) / ((v / bc).sqrt() + 1e-5)
                dist_i = sq_u.detach().sqrt().unsqueeze(1)
                part_scale = (dist_i / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2)
                base_prior.z.index_add_(
                    0, u_rows, step_vec * part_scale, alpha=-float(recipe.t1_prior_row_lr) * row_settle
                )
        finally:
            for p, flag in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(flag)

    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


__all__ = [
    "UnifiedPriorRegularizer",
    "_compute_pythagorean_smoothed_cells",
    "_validate_unified_prior_state",
    "should_use_unified_pullback",
    "unified_pullback_step",
]
