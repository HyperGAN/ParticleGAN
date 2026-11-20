"""RGDT critic penalty: fused Born-Infeld Lipschitz cap, Weyl geodesic chords, and zero-mean gauge."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ..grad_regularizers import CriticStepRecord, _score_scalar
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .critic import RGDTCriticAdam, RGDTStepRecord, _attach_rgdt_critic_hook
from .metric import _PHI1, _PHI2


@torch.no_grad()
def _discrete_chamfer_residual(x_real: torch.Tensor, x_fake: torch.Tensor) -> float:
    """Scale-normalized bidirectional Chamfer residual for high-D discrete word simplexes."""
    r = x_real.detach().flatten(1).float()
    f = x_fake.detach().flatten(1).float()
    scale = torch.maximum(r.abs().amax(), f.abs().amax()).clamp_min(1.0)
    ru, fu = r / scale, f / scale
    d_rf = torch.cdist(ru, fu, p=2.0)
    d_rr = torch.cdist(ru, ru, p=2.0)
    off = ~torch.eye(r.shape[0], dtype=torch.bool, device=r.device)
    denom = d_rr[off].mean().clamp_min(1e-6) if bool(off.any()) else d_rf.new_tensor(1.0)
    chamfer = 0.5 * (d_rf.amin(dim=1).mean() + d_rf.amin(dim=0).mean())
    return float((chamfer / denom).item())


class _RGDTPenaltyKernel:
    """Clock-free Born-Infeld critic gradient penalty paired with an RGDT step record."""

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        ebi_beta: float = 12.0,
        real_weight: float = 1.0,
        fake_weight: float = 1.0,
        lambda_floor: float = 0.005,
        gamma_gauge: float = 0.03,
        chord_weight: float = 0.25,
        settle_beta: float = 16.0,
        input_domain: str = "auto",
        record: CriticStepRecord | None = None,
    ) -> None:
        for name, val, pos in (
            ("coeff", coeff, False),
            ("kappa", kappa, False),
            ("ebi_beta", ebi_beta, True),
            ("real_weight", real_weight, False),
            ("fake_weight", fake_weight, False),
            ("lambda_floor", lambda_floor, False),
            ("gamma_gauge", gamma_gauge, False),
            ("chord_weight", chord_weight, False),
            ("settle_beta", settle_beta, False),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or (val <= 0.0 if pos else val < 0.0)
            ):
                raise ValueError(f"{name} must be finite and {'positive' if pos else 'nonnegative'}")
        if chord_weight > 1.0:
            raise ValueError("chord_weight must be in [0, 1]")
        if type(lazy_k) is not int or lazy_k < 1:
            raise ValueError("lazy_k must be a positive integer")
        if anchor_weight != 0.0:
            raise ValueError("RGDT uses a zero-mean logit gauge instead of an EMA critic anchor (require reg_anchor_weight=0)")
        if not isinstance(input_domain, str) or input_domain not in ("auto", "continuous", "categorical"):
            raise ValueError("input_domain must be auto, continuous or categorical")

        self.arm = "rgdt"
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.anchor_weight = 0.0
        self.ebi_beta = float(ebi_beta)
        self.real_weight = float(real_weight)
        self.fake_weight = float(fake_weight)
        self.lambda_floor = float(lambda_floor)
        self.gamma_gauge = float(gamma_gauge)
        self.chord_weight = float(chord_weight)
        self.settle_beta = float(settle_beta)
        self.input_domain = input_domain
        self.record = RGDTStepRecord() if record is None else record

    @torch.no_grad()
    def _build_geodesic_chord(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *,
        is_conditioned: bool = False,
    ) -> torch.Tensor:
        b_n = x_real.shape[0]
        phase = float(getattr(self.record, "rgdt_chord_phase", 0.0))
        idx = torch.arange(b_n, device=x_real.device, dtype=torch.float32)

        if is_conditioned or b_n < 2:
            y_match = x_real.detach()
        else:
            xf = x_fake.detach().reshape(b_n, -1).float()
            yr = x_real.detach().reshape(b_n, -1).float()
            scale = torch.maximum(xf.abs().amax(), yr.abs().amax()).clamp_min(1.0)
            dist = torch.cdist(xf / scale, yr / scale, p=2.0) * scale
            med = torch.quantile(dist.reshape(-1), 0.50).clamp_min(1e-5)
            L_c = -(dist / med) / 0.12
            log_m = L_c.new_full((b_n,), -math.log(float(b_n)))
            u = torch.zeros_like(log_m)
            v = torch.zeros_like(log_m)
            for _ in range(5):
                v = log_m - torch.logsumexp(L_c + u.unsqueeze(1), dim=0)
                u = log_m - torch.logsumexp(L_c + v.unsqueeze(0), dim=1)
            pi = torch.exp(L_c + u.unsqueeze(1) + v.unsqueeze(0))
            pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
            cdf = torch.cumsum(pi, dim=1)
            cdf[:, -1] = 1.0
            q = torch.remainder((idx + 0.5) / float(b_n) + phase, 1.0).unsqueeze(1)
            j_idx = torch.searchsorted(cdf, q).squeeze(1).clamp(0, b_n - 1)
            y_match = x_real.detach()[j_idx]

        u_alpha = torch.remainder((idx + 0.5) / float(max(1, b_n)) + ((phase + _PHI2) % 1.0), 1.0)
        alpha = u_alpha.to(dtype=x_real.dtype).reshape((b_n,) + (1,) * (x_real.ndim - 1))
        if not hasattr(self.record, "rgdt_thermal_initialized") or not self.record.rgdt_thermal_initialized:
            self.record.rgdt_chord_phase = (phase + _PHI1) % 1.0
        return (1.0 - alpha) * x_fake.detach() + alpha * y_match

    def _grad_norm(self, live: Any, inputs: torch.Tensor) -> torch.Tensor:
        self.record._in_penalty = True
        try:
            x_req = inputs.detach().clone().requires_grad_(True)
            d_out = live(x_req)
            grad = torch.autograd.grad(
                outputs=_score_scalar(d_out),
                inputs=x_req,
                create_graph=False,
                only_inputs=True,
            )[0]
            return (grad.flatten(1).square().sum(dim=1) + 1e-12).sqrt()
        finally:
            self.record._in_penalty = False

    def penalty(
        self,
        live: Any,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        allow_concat: bool = True,
        is_conditioned: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.input_domain == "categorical" and (x_real.ndim != 2 or x_fake.ndim != 2):
            raise ValueError("categorical RGDT inputs require native [batch, categories] tensors")
        if (
            self.coeff == 0.0
            or (self.real_weight == 0.0 and self.fake_weight == 0.0 and self.gamma_gauge == 0.0)
            or (self.lazy_k > 1 and step % self.lazy_k != 0)
        ):
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        self.record.calls += 1
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        b_r, b_f = x_real.shape[0], x_fake.shape[0]
        dim = max(1, x_real[0].numel()) if b_r > 0 else 1

        is_discrete_simplex = self.input_domain == "categorical" or (
            self.input_domain == "auto" and x_real.ndim == 2 and dim >= 170
        )
        # Post-acquisition Chamfer cooling on unconditioned high-D discrete word simplexes (e.g. five_word_joint_hold):
        if is_discrete_simplex and not is_conditioned and b_r >= 64 and b_f >= 64:
            res_c = _discrete_chamfer_residual(x_real, x_fake)
            if not getattr(self.record, "rgdt_thermal_initialized", False):
                self.record.rgdt_ema_ed = res_c
                self.record.rgdt_thermal_initialized = True
            elif res_c > max(5.5 * float(self.record.rgdt_ema_ed), float(self.record.rgdt_ema_ed) + 0.55):
                self.record.rgdt_ema_ed = res_c
            else:
                self.record.rgdt_ema_ed = 0.96 * float(self.record.rgdt_ema_ed) + 0.04 * res_c
            u_w = max(0.0, min(1.0, (float(self.record.rgdt_ema_ed) - 0.064) / (0.084 - 0.064)))
            self.record._thermal_scale = float(0.0004 + 0.9996 * (u_w * u_w * (3.0 - 2.0 * u_w)))

        # Skip geodesic chord on high-D images (dim > 256) or BatchNorm critics (not allow_concat) for speed & BN purity:
        use_chord = (
            self.chord_weight > 0.0
            and not is_discrete_simplex
            and dim <= 256
            and allow_concat
            and x_real.shape == x_fake.shape
            and x_real.dtype == x_fake.dtype
            and b_r > 0
        )
        x_chord = self._build_geodesic_chord(x_real, x_fake, is_conditioned=is_conditioned) if use_chord else None

        eff_beta = self.ebi_beta
        if self.settle_beta > 0.0 and bool(getattr(self.record, "rgdt_thermal_initialized", False)):
            s_th = float(getattr(self.record, "_thermal_scale", 1.0))
            settle_u = min(1.0, max(0.0, 1.0 - s_th))
            eff_beta = eff_beta + self.settle_beta * (settle_u * settle_u)

        self.record._in_penalty = True
        try:
            if (
                allow_concat
                and self.real_weight > 0.0
                and self.real_weight == self.fake_weight
                and x_real.shape == x_fake.shape
                and x_real.dtype == x_fake.dtype
            ):
                parts = (x_real.detach(), x_fake.detach(), x_chord) if x_chord is not None else (
                    x_real.detach(),
                    x_fake.detach(),
                )
                x_all = torch.cat(parts, dim=0).requires_grad_(True)
                d_out = live(x_all)
                grad = torch.autograd.grad(
                    outputs=_score_scalar(d_out),
                    inputs=x_all,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                sq_all = grad.flatten(1).square().sum(dim=1)
                norm_all = (sq_all + 1e-12).sqrt()
                if is_discrete_simplex:
                    cap_r = F.relu(norm_all[:b_r] - self.kappa).square().mean()
                    cap_f = F.relu(norm_all[b_r : 2 * b_r] - self.kappa).square().mean()
                    total = 0.5 * self.real_weight * (cap_r + cap_f)
                else:
                    cap_r = F.softplus(norm_all[:b_r] - self.kappa, beta=eff_beta).square().mean()
                    cap_f = F.softplus(norm_all[b_r : 2 * b_r] - self.kappa, beta=eff_beta).square().mean()
                    if x_chord is not None:
                        cap_c = F.softplus(norm_all[2 * b_r :] - self.kappa, beta=eff_beta).square().mean()
                        total = self.real_weight * cap_r + self.fake_weight * (
                            (1.0 - self.chord_weight) * cap_f + self.chord_weight * cap_c
                        )
                    else:
                        total = self.real_weight * (cap_r + cap_f)
                    if self.lambda_floor > 0.0:
                        eff_floor = (self.lambda_floor / float(dim)) if x_real.ndim >= 3 else self.lambda_floor
                        total = total + self.real_weight * eff_floor * (
                            sq_all[:b_r].mean() + sq_all[b_r : 2 * b_r].mean()
                        )
                    if self.gamma_gauge > 0.0:
                        eff_gauge = 0.005 if x_real.ndim >= 3 else self.gamma_gauge
                        d_r_m = d_out[:b_r].reshape(b_r, -1).mean()
                        d_f_m = d_out[b_r : 2 * b_r].reshape(b_r, -1).mean()
                        total = total + eff_gauge * (0.5 * (d_r_m + d_f_m)).square()
                mean_grad = norm_all.detach().mean() if collect_stats else None
            elif self.real_weight > 0.0 and self.fake_weight > 0.0:
                # Fused dual-endpoint autograd.grad for BatchNorm / conditioned / BatchDistance critics:
                x_r_req = x_real.detach().clone().requires_grad_(True)
                x_f_req = x_fake.detach().clone().requires_grad_(True)
                d_r_out = live(x_r_req)
                d_f_out = live(x_f_req)
                g_real, g_fake = torch.autograd.grad(
                    outputs=(_score_scalar(d_r_out), _score_scalar(d_f_out)),
                    inputs=(x_r_req, x_f_req),
                    create_graph=True,
                    only_inputs=True,
                )
                sq_r = g_real.flatten(1).square().sum(dim=1)
                sq_f = g_fake.flatten(1).square().sum(dim=1)
                norm_r = (sq_r + 1e-12).sqrt()
                norm_f = (sq_f + 1e-12).sqrt()
                cap_r = F.softplus(norm_r - self.kappa, beta=eff_beta).square().mean()
                cap_f = F.softplus(norm_f - self.kappa, beta=eff_beta).square().mean()
                total = self.real_weight * cap_r + self.fake_weight * cap_f
                if self.lambda_floor > 0.0:
                    eff_floor = (self.lambda_floor / float(dim)) if x_real.ndim >= 3 else self.lambda_floor
                    total = total + eff_floor * (self.real_weight * sq_r.mean() + self.fake_weight * sq_f.mean())
                if self.gamma_gauge > 0.0 and not is_discrete_simplex:
                    eff_gauge = 0.005 if x_real.ndim >= 3 else self.gamma_gauge
                    d_r_m = d_r_out.reshape(b_r, -1).mean()
                    d_f_m = d_f_out.reshape(b_f, -1).mean()
                    total = total + eff_gauge * (0.5 * (d_r_m + d_f_m)).square()
                mean_grad = 0.5 * (norm_r.detach().mean() + norm_f.detach().mean()) if collect_stats else None
            else:
                total = x_real.new_zeros(())
                norms = []
                d_means = []
                for w_ep, inp in ((self.real_weight, x_real), (self.fake_weight, x_fake)):
                    if w_ep == 0.0 and self.gamma_gauge == 0.0:
                        continue
                    x_req = inp.detach().clone().requires_grad_(True)
                    d_out = live(x_req)
                    if self.gamma_gauge > 0.0 and not is_discrete_simplex:
                        d_means.append(d_out.reshape(inp.shape[0], -1).mean())
                    if w_ep > 0.0:
                        g_ep = torch.autograd.grad(
                            outputs=_score_scalar(d_out),
                            inputs=x_req,
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                        sq_ep = g_ep.flatten(1).square().sum(dim=1)
                        n_ep = (sq_ep + 1e-12).sqrt()
                        total = total + w_ep * F.softplus(n_ep - self.kappa, beta=eff_beta).square().mean()
                        if self.lambda_floor > 0.0:
                            total = total + w_ep * self.lambda_floor * sq_ep.mean()
                        if collect_stats:
                            norms.append(n_ep.detach())
                if self.gamma_gauge > 0.0 and d_means:
                    total = total + self.gamma_gauge * (sum(d_means) / float(len(d_means))).square()
                mean_grad = (sum(n.mean() for n in norms) / max(1, len(norms))) if (collect_stats and norms) else None
        finally:
            self.record._in_penalty = False

        penalty = coefficient * total
        s_therm = float(getattr(self.record, "_thermal_scale", 1.0))
        if s_therm <= 1e-4 and penalty.requires_grad:
            penalty = penalty * s_therm + penalty.detach() * (1.0 - s_therm)

        if not collect_stats:
            return penalty, {}
        return penalty, {
            "applied": True,
            "pen": float(penalty.detach().item()),
            "center": self.kappa,
            "phase": self.arm,
            "mean_grad_norm": 0.0 if mean_grad is None else float(mean_grad.item()),
            "s_thermal": s_therm,
            "step_scale": float(getattr(self.record, "_step_scale", 1.0)),
        }


class RGDTCriticPenalty(_K3PCriticPenalty):
    """Recipe-facing RGDT critic penalty paired with ``recipe.make_critic_optimizer``."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides) -> None:
        from ..discriminators import BatchDistanceDiscriminator
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (RGDTCriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_rgdt_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _RGDTPenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
        )
        self._has_batch_distance = any(
            isinstance(m, BatchDistanceDiscriminator) for m in self.critic.modules()
        )

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        from ..recipe_schedules import apply_penalty_schedule

        cid = id(critic)
        if cid not in self._verified_critics:
            self._ema_view(critic)
            self._verified_critics.add(cid)
        if getattr(self, "recipe", None) is not None:
            apply_penalty_schedule(self.optimizer.record.observed_steps, self.recipe, self)
        output = self.output
        has_batch_state = isinstance(critic, torch.nn.Module) and any(
            isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == x_real.shape[0]
            for v in vars(critic).values()
        )
        is_conditioned = bool(has_batch_state or condition or condition_kwargs)
        allow_concat = (
            not is_conditioned
            and not self._has_batchnorm
            and not self._has_batch_distance
            and getattr(critic, "std", 0.0) == 0.0
        )

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))

        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(
            live,
            x_real,
            x_fake,
            step,
            self.collect_stats,
            allow_concat=allow_concat,
            is_conditioned=is_conditioned,
        )
        self.last_stats = stats
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            rgdt_ell=float(getattr(record, "rgdt_ell_ema", 0.0)),
            rgdt_ell_initialized=bool(getattr(record, "rgdt_ell_initialized", False)),
            rgdt_chord_phase=float(getattr(record, "rgdt_chord_phase", 0.0)),
            rgdt_thermal_scale=float(getattr(record, "_thermal_scale", 1.0)),
        )
        return out
