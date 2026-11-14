"""RHBI critic penalty: smooth Born-Infeld endpoint Lipschitz cap."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ..grad_regularizers import CriticStepRecord, GradientPenalty, _score_scalar
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .critic import RHBICriticAdam, RHBIStepRecord, _attach_rhbi_critic_hook
from .primitive import RHBILoss

_GOLDEN_CONJ = 0.6180339887498949
_PLASTIC_CONJ = 0.7548776662466927


class _RHBIPenaltyKernel:
    """Smooth C^infinity Born-Infeld endpoint and geodesic chord gradient-norm cap.

    Enforces ``||grad_x D(x)||_2 <= kappa`` via smooth squared softplus on active
    real and fake endpoints (plus optional balanced-Sinkhorn geodesic transport
    chords when ``chord_weight > 0``) so the relativistic slack tunneling gain
    ``0.5 * tau`` matches the unit linear-tail slope of the physical Born-Infeld
    transport cost without spurious Fourier dead-zone traps.
    """

    _grad_norm = staticmethod(GradientPenalty._grad_norm)

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        ebi_beta: float = 30.0,
        real_weight: float = 1.0,
        fake_weight: float = 1.0,
        lambda_floor: float = 0.0,
        gamma_gauge: float = 0.0,
        chord_weight: float = 0.0,
        akt_horizon: bool = False,
        akt_mode: str = "sqrt",
        settle_beta: float = 0.0,
        settle_mode: str = "residual",
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
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or (
                val <= 0.0 if pos else val < 0.0
            ):
                raise ValueError(f"{name} must be finite and {'positive' if pos else 'nonnegative'}")
        if float(chord_weight) > 1.0:
            raise ValueError("chord_weight must be in [0, 1]")
        if type(lazy_k) is not int or lazy_k < 1:
            raise ValueError("lazy_k must be a positive integer")
        if type(akt_horizon) is not bool:
            raise ValueError("akt_horizon must be a boolean")
        if type(akt_mode) is not str or akt_mode not in ("sqrt", "weibull"):
            raise ValueError("akt_mode must be 'sqrt' or 'weibull'")
        if type(settle_mode) is not str or settle_mode not in ("residual", "thermal"):
            raise ValueError("settle_mode must be 'residual' or 'thermal'")
        if anchor_weight != 0.0:
            raise ValueError("RHBI does not use an EMA critic anchor (require reg_anchor_weight=0)")
        self.arm = "rhbi"
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
        self.akt_horizon = akt_horizon
        self.akt_mode = akt_mode
        self.settle_beta = float(settle_beta)
        self.settle_mode = settle_mode
        self.record = RHBIStepRecord() if record is None else record

    @torch.no_grad()
    def _build_geodesic_chord(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *,
        is_conditioned: bool = False,
    ) -> torch.Tensor:
        b_n = x_real.shape[0]
        phi = float(getattr(self.record, "rhbi_chord_phase", 0.0))
        next_phi = (phi + _GOLDEN_CONJ) % 1.0
        if hasattr(self.record, "rhbi_chord_phase"):
            self.record.rhbi_chord_phase = next_phi
        idx_f = (torch.arange(b_n, device=x_real.device, dtype=torch.float32) + 0.5) / float(b_n)
        alpha_1d = ((idx_f + phi) % 1.0).to(dtype=x_real.dtype)
        alpha = alpha_1d.reshape(b_n, *([1] * (x_real.ndim - 1)))
        if is_conditioned or b_n <= 1:
            y_pair = x_real.detach()
        else:
            ell_init = bool(getattr(self.record, "rhbi_ell_initialized", False))
            ell_s = float(getattr(self.record, "rhbi_ell_ema", 0.0)) if ell_init else 0.0
            if ell_s <= 1e-6:
                ell_t = RHBILoss._batch_ell_quantile_tensor(
                    x_real, akt_horizon=self.akt_horizon, akt_mode=self.akt_mode
                )
            else:
                ell_t = torch.tensor(ell_s, dtype=torch.float32, device=x_real.device)
            c_hat = RHBILoss._born_infeld_c_hat_detached(x_fake, x_real, ell_t)
            L_bal = -c_hat.float() / 0.25
            u_s = L_bal.new_zeros(b_n)
            v_s = L_bal.new_zeros(b_n)
            for _ in range(6):
                v_s = -torch.logsumexp(L_bal + u_s.unsqueeze(1), dim=0)
                u_s = -torch.logsumexp(L_bal + v_s.unsqueeze(0), dim=1)
            probs = torch.exp(L_bal + u_s.unsqueeze(1) + v_s.unsqueeze(0))
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            cdf = probs.cumsum(dim=1)
            u_qmc = (idx_f + ((phi * _PLASTIC_CONJ) % 1.0)) % 1.0
            j_idx = (cdf < u_qmc.unsqueeze(1)).sum(dim=1).clamp(0, b_n - 1)
            y_pair = x_real.detach()[j_idx]
        return (1.0 - alpha) * x_fake.detach() + alpha * y_pair

    def penalty(
        self,
        live,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        allow_concat: bool = False,
        is_conditioned: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        self.record.calls += 1
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        use_chord = (
            self.chord_weight > 0.0
            and x_real.shape == x_fake.shape
            and x_real.dtype == x_fake.dtype
            and x_real.shape[0] > 0
        )
        x_chord = (
            self._build_geodesic_chord(x_real, x_fake, is_conditioned=is_conditioned)
            if use_chord
            else None
        )

        eff_beta = self.ebi_beta
        if self.settle_beta > 0.0 and bool(getattr(self.record, "rhbi_thermal_initialized", False)):
            if self.settle_mode == "thermal":
                s_th = float(getattr(self.record, "_thermal_scale", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - s_th))
            else:
                res = float(getattr(self.record, "rhbi_ema_residual", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - res / 0.22))
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
                x_both = torch.cat(parts, dim=0).requires_grad_(True)
                d_out = live(x_both)
                grad = torch.autograd.grad(
                    outputs=_score_scalar(d_out),
                    inputs=x_both,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                sq_all = grad.flatten(1).square().sum(dim=1)
                norm_all = (sq_all + 1e-12).sqrt()
                b_n = x_real.shape[0]
                cap_r = F.softplus(norm_all[:b_n] - self.kappa, beta=eff_beta).square().mean()
                cap_f = F.softplus(norm_all[b_n : 2 * b_n] - self.kappa, beta=eff_beta).square().mean()
                if x_chord is not None:
                    cap_c = F.softplus(norm_all[2 * b_n :] - self.kappa, beta=eff_beta).square().mean()
                    total = self.real_weight * cap_r + self.fake_weight * (
                        (1.0 - self.chord_weight) * cap_f + self.chord_weight * cap_c
                    )
                else:
                    total = self.real_weight * (cap_r + cap_f)
                if self.lambda_floor > 0.0:
                    total = total + self.real_weight * self.lambda_floor * (
                        sq_all[:b_n].mean() + sq_all[b_n : 2 * b_n].mean()
                    )
                if self.gamma_gauge > 0.0:
                    d_r_mean = d_out[:b_n].reshape(b_n, -1).mean()
                    d_f_mean = d_out[b_n : 2 * b_n].reshape(b_n, -1).mean()
                    total = total + self.gamma_gauge * (0.5 * (d_r_mean + d_f_mean)).square()
                norms = [norm_all.detach()] if collect_stats else []
            else:
                total = x_real.new_zeros(())
                norms = []
                d_means = []
                fake_ep_w = self.fake_weight * (1.0 - self.chord_weight) if x_chord is not None else self.fake_weight
                chord_ep_w = self.fake_weight * self.chord_weight if x_chord is not None else 0.0
                for cap_w, floor_w, inputs, is_endpoint in (
                    (self.real_weight, self.real_weight, x_real, True),
                    (fake_ep_w, self.fake_weight, x_fake, True),
                    *((chord_ep_w, 0.0, x_chord, False) for _ in (() if x_chord is None else (1,))),
                ):
                    if cap_w == 0.0 and floor_w == 0.0 and (not is_endpoint or self.gamma_gauge == 0.0):
                        continue
                    x_req = inputs.detach().clone().requires_grad_(True)
                    d_out = live(x_req)
                    if is_endpoint and self.gamma_gauge > 0.0:
                        d_means.append(d_out.reshape(inputs.shape[0], -1).mean())
                    if cap_w > 0.0 or (is_endpoint and floor_w > 0.0 and self.lambda_floor > 0.0):
                        grad = torch.autograd.grad(
                            outputs=_score_scalar(d_out),
                            inputs=x_req,
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                        sq = grad.flatten(1).square().sum(dim=1)
                        norm = (sq + 1e-12).sqrt()
                        if cap_w > 0.0:
                            cap = F.softplus(norm - self.kappa, beta=eff_beta).square().mean()
                            total = total + cap_w * cap
                        if is_endpoint and floor_w > 0.0 and self.lambda_floor > 0.0:
                            total = total + floor_w * self.lambda_floor * sq.mean()
                        if collect_stats:
                            norms.append(norm.detach())
                if self.gamma_gauge > 0.0 and d_means:
                    d_bar = sum(d_means) / float(len(d_means))
                    total = total + self.gamma_gauge * d_bar.square()
        finally:
            self.record._in_penalty = False

        pen = 0.5 * coefficient * total
        s_therm = float(getattr(self.record, "_thermal_scale", 1.0))
        if s_therm < 1.0 and pen.requires_grad:
            pen = pen * s_therm + pen.detach() * (1.0 - s_therm)
        if not collect_stats:
            return pen, {}
        ell_t = getattr(self.record, "_rhbi_ell_t", None)
        norm_mean_t = (norms[0].mean() if len(norms) == 1 else torch.cat(norms).mean()) if norms else pen.new_zeros(())
        if ell_t is not None and ell_t.device == pen.device:
            pen_v, mean_norm, ell_v = torch.stack(
                (pen.detach().float(), norm_mean_t.float(), ell_t.detach().float())
            ).tolist()
        else:
            pen_v, mean_norm = torch.stack((pen.detach().float(), norm_mean_t.float())).tolist()
            ell_v = float(getattr(self.record, "rhbi_ell_ema", 0.0))
        return pen, {
            "applied": True,
            "pen": float(pen_v),
            "center": self.kappa,
            "phase": self.arm,
            "mean_grad_norm": float(mean_norm),
            "rhbi_ell": float(ell_v),
        }


class RHBICriticPenalty(_K3PCriticPenalty):
    """Recipe-facing RHBI critic penalty paired with ``recipe.make_critic_optimizer``."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides) -> None:
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (RHBICriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")
        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_rhbi_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _RHBIPenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
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
        if getattr(getattr(self, "recipe", None), "critic_step_mode", "none") == "finite_cap":
            real, fake = x_real.detach().clone(), x_fake.detach().clone()

            def finite_cap():
                return torch.maximum(
                    self.regularizer._grad_norm(live, real).max(),
                    self.regularizer._grad_norm(live, fake).max(),
                )

            self.optimizer.bind_cap_panel(finite_cap, self.regularizer.kappa)
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            rhbi_ell=float(getattr(record, "rhbi_ell_ema", 0.0)),
            rhbi_ell_initialized=bool(getattr(record, "rhbi_ell_initialized", False)),
            rhbi_chord_phase=float(getattr(record, "rhbi_chord_phase", 0.0)),
            rhbi_thermal_scale=float(getattr(record, "_thermal_scale", 1.0)),
        )
        return out
