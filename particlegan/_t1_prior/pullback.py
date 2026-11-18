"""Gersho Total-Variance & Kantorovich-Laguerre Pullback Prior Dynamics (Theory 1A).

Solves the continuous-generator / discontinuous-data obstruction (Theory 1) from
first principles via a strict Division of Labor between the continuous generator
chart ``G_theta`` and the recipe-owned latent location table ``prior.z``:

1. **Streaming Zador-Gersho Vector Quantizer** ``(C_k, w_k, sigma_{cell,k}^2)``:
   Maintains ``K = min(max_prototypes, B)`` online Voronoi cells in ``O(B K D)``
   with slow empirical probability mass EMA ``w_k`` (eliminating single-batch
   multinomial shot noise), unbiased within-cell variance ``sigma_{cell,k}^2``,
   uncovered-mode teleportation, and Zador-Gersho optimal distortion allocation
   ``E_k = w_k^{d/(d+2)} sigma_{cell,k}^2`` (conserving exact probability mass
   ``w_k`` while giving rare modes non-collinear multi-axis resolution).

2. **Part A — Smooth Low-Energy Generator Chart & Gersho Pullback Isometry**
   (with ``prior.z`` detached):
   * Mass-weighted entropic Sinkhorn barycentric projection onto ``(C_k, w_k)``
     with transport-residual DualNorm step cooling ``g_scale``.
   * Local Gersho pullback kernel law matching ``||G(mu + sigma_z eps) - G(mu)||``
     (plus exact deterministic 2D orthogonal companion probe when ``d_z == D == 2``)
     to local cell width ``sigma_{cell,k(i)}`` weighted by inverse cell density.
     By the Law of Total Variance:
     ``Cov(G(z)) = Cov_{k ~ w}(C_k) + sum_k w_k sigma_{cell,k}^2 I == Cov(Y)``.

3. **Part B — Kantorovich-Laguerre Mass-Balanced Island Condensation**
   (on sampled ``prior.z[u_rows]`` with ``G_theta`` frozen):
   * Persistent Kantorovich dual potentials ``psi_k`` updated by subgradient
     ascent on empirical Laguerre cell shares ``k*(i) = argmin_k (||G(mu_i) - C_k||^2 - psi_k)``,
     enforcing exact mode masses without cross-valley midpoint blur.
   * Pullback condensation of ``G(mu_i)`` onto ``C_{k*(i)}`` plus one-sided
     isometric latent cloud envelope ``relu(sigma_mu* - std(mu))^2``, stepped by
     bias-corrected per-row coherent drift SNR ``m_i / sqrt(v_i)``.

Strictly clock-free (zero reads of ``completed_steps``, ``total_steps``, or
``max_steps``; zero extra RNG draws) and completely checkpointable.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .._empf.prior import EMPFPriorRegularizer
from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior

_PHI1 = 0.6180339887498949

_T1_STATE_KEYS = frozenset({
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
    "rows_initialized",
    "row_m",
    "row_v",
    "row_n",
})


def _validate_t1_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict) or set(state) != _T1_STATE_KEYS:
        raise ValueError("invalid T1PullbackPriorRegularizer checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != 1:
        raise ValueError("unsupported T1PullbackPriorRegularizer checkpoint schema")
    for bkey in ("quantizer_initialized", "iso_initialized", "rows_initialized"):
        if type(state[bkey]) is not bool:
            raise ValueError(f"invalid T1 boolean field {bkey}")
    for fkey, low, strict_low in (
        ("phase", 0.0, False),
        ("iso_ema", 0.0, True),
        ("res_ema", 0.0, False),
    ):
        val = state[fkey]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"nonfinite or invalid T1 scalar field {fkey}")
        if float(val) <= low if strict_low else float(val) < low:
            raise ValueError(f"out-of-range T1 scalar field {fkey}")
    if float(state["phase"]) >= 1.0:
        raise ValueError("T1 phase must lie in [0, 1)")

    q_init = state["quantizer_initialized"]
    q_tensors = ("proto_c", "proto_var", "proto_w", "proto_hits", "psi", "w_fake_ema")
    if not q_init:
        if any(state[k] is not None for k in q_tensors):
            raise ValueError("uninitialized T1 quantizer must store None tensors")
    else:
        for k in q_tensors:
            t = state[k]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
                raise ValueError(f"T1 {k} must be a float32 tensor")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"nonfinite values in T1 {k}")
        c = state["proto_c"]
        if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
            raise ValueError("invalid T1 proto_c shape")
        k_len = c.shape[0]
        for k in ("proto_var", "proto_w", "proto_hits", "psi", "w_fake_ema"):
            if state[k].ndim != 1 or state[k].shape[0] != k_len:
                raise ValueError(f"incompatible T1 {k} shape")
        if bool((state["proto_var"] <= 0).any()):
            raise ValueError("T1 proto_var must be strictly positive")
        if bool((state["proto_hits"] <= 0).any()):
            raise ValueError("T1 proto_hits must be strictly positive")
        if bool((state["proto_w"] < 0).any()) or not math.isclose(float(state["proto_w"].sum().item()), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("T1 proto_w must be nonnegative and sum to 1")
        if bool((state["w_fake_ema"] < 0).any()):
            raise ValueError("T1 w_fake_ema must be nonnegative")

    r_init = state["rows_initialized"]
    r_tensors = ("row_m", "row_v", "row_n")
    if not r_init:
        if any(state[k] is not None for k in r_tensors):
            raise ValueError("uninitialized T1 row state must store None tensors")
    else:
        rm, rv, rn = state["row_m"], state["row_v"], state["row_n"]
        for name, t in zip(r_tensors, (rm, rv, rn)):
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                raise ValueError(f"T1 {name} must be a floating tensor")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"nonfinite values in T1 {name}")
        if rm.ndim != 2 or rm.shape[0] < 1 or rm.shape[1] < 1:
            raise ValueError("invalid T1 row_m shape")
        if rv.shape != (rm.shape[0], 1) or rn.shape != (rm.shape[0], 1):
            raise ValueError("incompatible T1 row_v / row_n shape")
        if bool((rv < 0).any()) or bool((rn < 0).any()):
            raise ValueError("T1 row_v and row_n must be nonnegative")
    return state


class T1PullbackPriorRegularizer(nn.Module):
    """Recipe-owned Theory 1A Gersho-Laguerre pullback prior module (``prior_regularizer='t1_pullback'``).

    * In ``GANTrainer`` (when ``Recipe.t1_pullback_weight > 0``): executes the
      coupled Division-of-Labor chart + Kantorovich-Laguerre pullback step and
      checkpoints all streaming quantizer, dual-potential, and per-row drift
      state via ``state_dict`` / ``load_state_dict``.
    * In standalone component hosts (AE/VAE/DDGAN/BiGAN loops calling
      ``prior_regularizer(rows)`` directly): delegates cleanly to
      :class:`EMPFPriorRegularizer` so non-``GANTrainer`` hosts retain
      degree-of-freedom-aware latent regularization.
    """

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
        bound_shape: Optional[Tuple[int, int]] = None,
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
        self._bound_shape = bound_shape

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
        self.rows_initialized: bool = False
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None

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
            "rows_initialized": bool(self.rows_initialized),
            "row_m": None if self.row_m is None else self.row_m.detach().clone(),
            "row_v": None if self.row_v is None else self.row_v.detach().clone(),
            "row_n": None if self.row_n is None else self.row_n.detach().clone(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        del strict
        validated = _validate_t1_state(state_dict)
        if validated["rows_initialized"] and self._bound_shape is not None:
            if tuple(validated["row_m"].shape) != tuple(self._bound_shape):
                raise ValueError("T1 row_m shape does not match bound prior table")
        if validated["quantizer_initialized"] and validated["proto_c"].shape[0] > self.max_prototypes:
            raise ValueError("T1 proto_c count exceeds configured max_prototypes")

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
        d2 = torch.cdist(x, self.proto_c, p=2.0).square()
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
            b_sum = torch.zeros((k, dim), device=x.device, dtype=torch.float32).index_add_(0, assign, x)
            b_mean = b_sum[hit] / b_cnt[hit].unsqueeze(1)
            self.proto_hits[hit] = (self.proto_hits[hit] + b_cnt[hit]).clamp_max(2000.0)
            base_c = (0.035 + 0.25 / torch.sqrt(self.proto_hits[hit])) + 0.38 * (shock * shock)
            alpha_c = (base_c * (b_cnt[hit] / (b_cnt[hit] + 2.0))).clamp_max(0.50).unsqueeze(1)
            self.proto_c[hit] = (1.0 - alpha_c) * self.proto_c[hit] + alpha_c * b_mean

            resid_sq = (x - self.proto_c[assign]).square().sum(dim=1) / float(dim)
            b_vsum = torch.zeros(k, device=x.device, dtype=torch.float32).scatter_add_(0, assign, resid_sq)
            hit2 = b_cnt >= 2
            if bool(hit2.any()):
                v_est = (b_vsum[hit2] / (b_cnt[hit2] - 0.75).clamp_min(1.0)).clamp_min(1e-10)
                self.proto_var[hit2] = 0.88 * self.proto_var[hit2] + 0.12 * v_est

        mean_w = 1.0 / float(k)
        dead = self.proto_w < (0.04 * mean_w)
        # Compare each point against its OWN nearest prototype's local variance v_loc AND macroscopic scale
        # so continuous/wide Gaussian 3-sigma tails never trigger false uncovered-mode teleportation!
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
                    # Merge close intra-mode duplicate pairs (d^2 <= 8 * dim * med_v), never distinct separated modes!
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


def t1_pullback_step(
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
    """Execute Part A (smooth chart + Gersho isometry) and Part B (Kantorovich-Laguerre condensation)."""
    recipe = trainer.recipe
    reg: T1PullbackPriorRegularizer = trainer.prior_regularizer
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)

    mu_g, sigma_z, can_condense_table = _resolve_prior_kernel(trainer.prior, latent, indices)

    # --- PART A: Smooth Generator Chart G_theta + Local Gersho Isometry (prior.z detached) ---
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
    loss_gan = trainer.loss.g_loss(fake_logits, real_logits)

    yr = real_g.detach().reshape(b, -1)
    dim = yr.shape[1]

    with torch.no_grad():
        reg.update_quantizer(yr)
        yf = yr.float()
        r_macro = yf.std(dim=0, unbiased=False).mean()
        is_point_target = float(r_macro.item()) < 1e-6
        r_macro_safe = r_macro.clamp_min(1e-4)
        sigma_cell_all = reg.proto_var.clamp_min(1e-10).sqrt()
        sigma_cell_med = torch.quantile(sigma_cell_all, 0.35).clamp_min(1e-6)
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
        xff = fake_flat.detach().float()
        d_xp = torch.cdist(xff, reg.proto_c, p=2.0)
        c_scale = d_xp.mean().clamp_min(1e-5)
        eps_s = max(0.012, min(0.06, 0.25 * reg.res_ema))
        L_xp = -(d_xp / c_scale) / eps_s
        log_a = -math.log(float(b))
        log_b = torch.log(reg.proto_w.clamp_min(1e-12))
        u = torch.zeros(b, device=yr.device, dtype=torch.float32)
        v = torch.zeros(len(reg.proto_c), device=yr.device, dtype=torch.float32)
        for _ in range(10):
            v = log_b - torch.logsumexp(L_xp + u.unsqueeze(1), dim=0)
            u = log_a - torch.logsumexp(L_xp + v.unsqueeze(0), dim=1)
        P_xp = torch.exp(L_xp + u.unsqueeze(1) + v.unsqueeze(0))
        y_ot = ((float(b) * P_xp) @ reg.proto_c).to(fake_flat.dtype)

        p2f = d_xp.min(dim=0).values
        w_p2f = (reg.proto_w * p2f).sum()
        f2p = d_xp.min(dim=1).values.median()
        cell_ref = 1.5 * math.sqrt(float(dim)) * sigma_cell_med
        cur_res = (
            float(d_xp.mean().item())
            if is_point_target
            else float(((torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro_safe).item())
        )
        reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(cur_res, 10.0)

        near_p_G = torch.cdist(x_ctr_flat.detach().float(), reg.proto_c, p=2.0).argmin(dim=1)
        sigma_cell_i = sigma_cell_all[near_p_G].to(fake_flat.dtype)
        inv_w_i = (
            (1.0 / (float(len(reg.proto_c)) * reg.proto_w[near_p_G]).clamp_min(1e-4))
            .sqrt()
            .clamp(0.5, 4.0)
            .to(fake_flat.dtype)
        )

    sq_ot = (fake_flat - y_ot).square().sum(dim=-1)
    sc_med_dt = sigma_cell_med.to(fake_flat.dtype)
    loss_g_ot = (torch.sqrt(sc_med_dt.square() + sq_ot) - sc_med_dt).mean()

    if sigma_z > 0.0 and recipe.t1_iso_weight > 0.0 and not is_point_target:
        dx_ker = fake_flat - x_ctr_flat
        d_z = mu_det.shape[1]
        scale_dim = math.sqrt(float(dim) / float(d_z))
        eps_norm = eps_g.norm(dim=-1).clamp_min(0.25)
        target_dx_norm = sigma_cell_i * scale_dim * eps_norm
        iso_1 = (inv_w_i * ((dx_ker.norm(dim=-1) - target_dx_norm) / (sc_med_dt * scale_dim)).square()).mean()
        if dim == 2 and d_z == 2:
            m_iso = min(b, 128)
            e_perp = torch.stack((-eps_g[:m_iso, 1], eps_g[:m_iso, 0]), dim=1)
            x_perp = trainer._postprocess(
                trainer.prior,
                trainer._generate(trainer.G, mu_det[:m_iso] + sigma_z * e_perp, 0.0, noise),
            ).reshape(m_iso, -1)
            dx_p = x_perp - x_ctr_flat[:m_iso]
            dx_0 = dx_ker[:m_iso]
            iso_2 = (inv_w_i[:m_iso] * ((dx_p.norm(dim=-1) - target_dx_norm[:m_iso]) / sc_med_dt).square()).mean()
            cos_sq = ((dx_0 * dx_p).sum(dim=-1) / (dx_0.norm(dim=-1) * dx_p.norm(dim=-1) + 1e-6)).square().mean()
            loss_g_iso = 0.5 * (iso_1 + iso_2) + 0.25 * cos_sq
        elif dim == 2 and d_z == 4:
            # Exact Radon-Hurwitz quaternion pushforward covariance Sigma_push = sigma_z^2 J_G J_G^T in R^{2x2}!
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
            M_dx = torch.stack(dx_all, dim=-1)  # [m_iso, 2, 4]
            eps_sq = es.square().sum(dim=-1).clamp_min(0.15).view(m_iso, 1, 1)
            Sigma_push = (M_dx @ M_dx.transpose(1, 2)) / eps_sq  # [m_iso, 2, 2]
            sc_sub = sigma_cell_i[:m_iso]
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

    loss_g = (
        0.15 * loss_gan
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

    trainer.opt_g.zero_grad()
    protected_evaluator = None
    if recipe.constraint_geometry_mode == "strict_progress":
        def protected_evaluator():
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
                return (trainer.loss.g_loss(logits, real_logits),)
            finally:
                if rec is not None:
                    rec._in_protected_probe = prev_probe

    constraint_geometry_backward(
        loss_g,
        trainer.opt_g,
        (loss_g,),
        protected_evaluator=protected_evaluator,
    )
    # Ensure DualNorm / Adam optimizer step counter for prior.z advances in lockstep with G and D.
    if base_prior.z.requires_grad and base_prior.z.grad is None:
        base_prior.z.grad = torch.zeros_like(base_prior.z)

    trainer.policy.after_generator_backward(
        loss_gan=loss_gan.detach(),
        loss_critic=(loss_d - penalty).detach(),
    )

    g_scale = 1.0 if is_point_target else max(0.04, min(1.0, (reg.res_ema / 0.10)))
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

            n_tot = len(base_prior.z)
            if sigma_z == 0.0 and n_tot <= 256 and dim <= 256:
                with torch.no_grad():
                    c_all = base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z
                    x_all = trainer._postprocess(
                        trainer.prior,
                        trainer._generate(trainer.G, c_all, 0.0, noise),
                    ).reshape(n_tot, -1).float()
                    d2_all = torch.cdist(x_all, reg.proto_c, p=2.0).square()
                    k_all = (d2_all - reg.psi.unsqueeze(0)).argmin(dim=1)
                    w_hard = torch.bincount(k_all, minlength=len(reg.proto_c)).float() / float(n_tot)
            else:
                w_hard = None

            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[u_rows]
            x_u = trainer._postprocess(
                trainer.prior,
                trainer._generate(trainer.G, c_u, 0.0, noise),
            ).reshape(len(u_rows), -1)

            with torch.no_grad():
                xu_det = x_u.detach().float()
                d2_up = torch.cdist(xu_det, reg.proto_c, p=2.0).square()
                k_star = (d2_up - reg.psi.unsqueeze(0)).argmin(dim=1)

                if w_hard is None:
                    w_hard_u = torch.bincount(k_star, minlength=len(reg.proto_c)).float() / float(len(u_rows))
                    reg.w_fake_ema = 0.88 * reg.w_fake_ema + 0.12 * w_hard_u
                    w_feedback = 0.35 * w_hard_u + 0.65 * reg.w_fake_ema
                else:
                    reg.w_fake_ema = 0.75 * reg.w_fake_ema + 0.25 * w_hard
                    w_feedback = 0.50 * w_hard + 0.50 * reg.w_fake_ema

                k_float = float(len(reg.proto_c))
                n_eff = float(n_tot) if w_hard is not None else float(len(u_rows))
                tv_now = 0.5 * float((reg.proto_w - reg.w_fake_ema).abs().sum().item())
                tv_floor = 0.35 * math.sqrt(k_float / max(n_eff * 8.0, 1.0))
                psi_gate = max(0.0, min(1.0, (tv_now - tv_floor) / max(0.08, tv_floor)))

                eta_scale = min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * k_float))
                eta_psi = psi_gate * eta_scale * float(dim) * float((r_macro_safe ** 2).item())
                # Gate-coupled Moreau-Yosida Tikhonov damping: damps active (psi, mu) sloshing
                # while freezing equilibrium potentials psi* cleanly when psi_gate == 0!
                reg.psi.mul_(1.0 - 0.008 * psi_gate)
                reg.psi.add_(reg.proto_w - w_feedback, alpha=eta_psi)
                reg.psi.sub_(reg.psi.mean())
                psi_cap = 2.5 * float(dim) * float((r_macro_safe ** 2).item()) + 0.5
                reg.psi.clamp_(-psi_cap, psi_cap)

                c_target = reg.proto_c[k_star].to(x_u.dtype)
                sc_u = sigma_cell_all[k_star].to(x_u.dtype)
                row_settle = max(0.15, min(1.0, max(reg.res_ema / 0.10, psi_gate)))

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
                part_scale = (dist_i / (1.5 * sc_u.unsqueeze(1))).clamp(0.05, 1.2)
                base_prior.z.index_add_(0, u_rows, step_vec * part_scale, alpha=-float(recipe.t1_prior_row_lr) * row_settle)
        finally:
            for p, flag in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(flag)

    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local
