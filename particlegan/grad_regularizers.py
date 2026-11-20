"""Historical K3P penalty and step record, also used as KA2 base classes.

The active public ``recipe.make_critic_penalty(opt_d)`` factory constructs
the KA2 subclass in ``particlegan.ka2``. The original classes here preserve
K3P research replay and provide shared gradient/lazy-application machinery.
"""

from typing import Any, Callable, Dict, Optional, Tuple
import math

import torch
import torch.nn.functional as F


def _score_scalar(logits):
    """Sum of per-image logit means.

    One logit per image matches ``logits.sum()``: other batch rows do not
    change an image's gradient. Extra logits are averaged inside the image so
    a spatial map does not multiply that gradient by its number of locations.
    """
    if logits.ndim < 2:
        return logits.sum()
    return logits.flatten(1).mean(dim=1).sum()


class CriticStepRecord:
    """Per-critic step state that K3P's penalty reads: LR record, anchor start, counters.

    One record belongs to one critic optimizer. ``record_step`` (after every
    critic optimizer step) advances the anchor EMA once started, then records
    the applied LR. The penalty starts the anchor and counts its calls.
    """

    KEYS = ("lr_max", "lr_last", "anchor_started", "calls", "observed_steps")

    def __init__(self, anchor: Optional[Any] = None) -> None:
        self.anchor = anchor
        self.lr_max = 0.0
        self.lr_last: Optional[float] = None
        self.anchor_started = False
        self.calls = 0
        self.observed_steps = 0

    def record_step(self, optimizer_or_lr) -> None:
        if self.anchor_started and self.anchor is not None:
            self.anchor.update_()
        if isinstance(optimizer_or_lr, (int, float, torch.Tensor)):
            lr = float(optimizer_or_lr)
        else:
            lr = max(float(g["lr"]) for g in optimizer_or_lr.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"lr_max": self.lr_max, "lr_last": self.lr_last, "anchor_started": self.anchor_started,
                "calls": self.calls, "observed_steps": self.observed_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.KEYS):
            keys = sorted(state) if isinstance(state, dict) else state
            raise ValueError(f"k3p state keys {keys} != expected {sorted(self.KEYS)}")
        values = (float(state["lr_max"]), None if state["lr_last"] is None else float(state["lr_last"]),
                  bool(state["anchor_started"]), int(state["calls"]), int(state["observed_steps"]))
        self.lr_max, self.lr_last, self.anchor_started, self.calls, self.observed_steps = values


class GradientPenalty:
    """K3P or a fixed L2/autograd critic gradient penalty.

    ``arm='k3p'`` keeps the learning-rate-scheduled handover below.
    ``arm='a_r1r2'`` is ``coeff/2 * (E ||g_r||² + E ||g_f||²)``;
    ``arm='b_cap'`` is ``coeff/2 * (E relu(||g_r|| - kappa)² +
    E relu(||g_f|| - kappa)²)``. Fixed arms use the v0.7.0 L2 kernels,
    without RMS rescaling, handover or EMA-anchor evaluation. Laziness
    applies to every arm. Other historical norms/methods are not supported.
    Fixed arms optionally weight the real/fake means independently; the
    default ``(1, 1)`` preserves their original arithmetic and forward calls.

    ``pen = coeff/2 * (s * A + (1 - s) * B)`` with

    * ``A`` (early, full LR): R1 on reals plus a one-sided cap on fakes, in RMS
      units: ``mean ||g_r||^2 / d + mean relu(||g_f|| / sqrt(d) - kappa)^2``;
    * ``B`` (late, annealed LR): one-sided caps on reals and fakes in L2 units
      plus ``anchor_weight * prox``, where ``prox = mean ||g_r - gbar_r||^2 / d``
      ties the critic's input gradient to that of its parameter EMA Dbar;
    * ``s = max(0, min(1, 2r) - 2f) / (1 - 2f)``, ``r`` = last critic LR / max
      critic LR seen, ``f`` = ``lr_floor``. A constant LR keeps ``s == 1``.

    ``g = grad_x D(x)`` and ``d`` is the per-sample input size.

    Args:
        arm: ``k3p`` (default), fixed squared-L2 ``a_r1r2``, or fixed
            one-sided L2 ``b_cap``. Fixed arms ignore the K3P blend/anchor.
        coeff: penalty strength.
        real_weight: multiplier of the fixed arm's real-input term.
        fake_weight: multiplier of the fixed arm's fake-input term. Zero skips
            its critic forward and input autograd; ``(1, 0)`` is R1-only.
        kappa: the cap on the gradient norm.
        lazy_k: apply every k-th step with the coefficient multiplied by k.
        lr_floor: the critic LR floor f as a fraction of the peak LR, so
            ``s == 0`` exactly at the floor.
        anchor_weight: weight of the EMA-anchor term ``prox`` (1 is K3P;
            0 removes it and needs no anchor).
        anchor: a ``particlegan.k3p.CriticAnchor`` (or anything with
            ``start_()``, ``update_()`` and ``__call__``), started at the first
            blended call.
        record: a ``CriticStepRecord`` shared with the critic optimizer that
            records its own steps (the recipe's critic optimizer does). Its
            anchor is used. Default: a private record.
    """

    def __init__(
        self,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        lr_floor: float = 0.01,
        anchor_weight: float = 1.0,
        anchor: Optional[Any] = None,
        record: Optional[CriticStepRecord] = None,
        *,
        arm: str = "k3p",
        real_weight: float = 1.0,
        fake_weight: float = 1.0,
    ) -> None:
        if arm not in ("k3p", "a_r1r2", "b_cap"):
            raise ValueError(f"unknown critic penalty arm: {arm!r}")
        self.arm = arm
        for name, value in (("real_weight", real_weight), ("fake_weight", fake_weight)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.real_weight, self.fake_weight = float(real_weight), float(fake_weight)
        if arm == "k3p" and (self.real_weight, self.fake_weight) != (1., 1.):
            raise ValueError("real/fake penalty weights require a fixed arm")
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.lr_floor = float(lr_floor)
        self.anchor_weight = float(anchor_weight)
        for name in ("coeff", "kappa", "anchor_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lazy_k < 1:
            raise ValueError(f"lazy_k must be >= 1, got {lazy_k}")
        if not math.isfinite(self.lr_floor) or not 0.0 <= self.lr_floor < 0.5:
            raise ValueError(f"lr_floor must satisfy 0 <= lr_floor < 0.5, got {lr_floor}")
        if record is None:
            record = CriticStepRecord(anchor)
        elif anchor is not None and anchor is not record.anchor:
            raise ValueError("pass either anchor= or a record= holding that anchor, not both")
        # Step state (LR record, anchor start, counters); shared with the
        # critic optimizer when the recipe builds the pair.
        self.record = record
        self.anchor = record.anchor
        # Identity of the critic served through the constructor anchor
        # (id only; not checkpointed). Guards against one instance silently
        # anchoring a second critic to the first critic's EMA.
        self._critic_id: Optional[int] = None

    def blend_weight(self) -> float:
        """Handover weight s in [0, 1]; 1.0 before any recorded step."""
        if self.record.lr_last is None or self.record.lr_max <= 0.0:
            return 1.0
        r = self.record.lr_last / self.record.lr_max
        f = self.lr_floor
        return max(0.0, min(1.0, 2.0 * r) - 2.0 * f) / (1.0 - 2.0 * f)

    def after_critic_step(self, optimizer_or_lr) -> None:
        """Record one critic optimizer step (call right after ``optimizer.step()``).

        Accepts the critic optimizer (the max group LR is used) or the applied
        LR as a number. Advances the anchor EMA (once started), then records
        the LR that drives s. The recipe's critic optimizer does this itself.
        """
        self.record.record_step(optimizer_or_lr)

    def state_dict(self) -> Dict[str, Any]:
        """Scalar step state. The anchor's EMA critic is saved separately."""
        return self.record.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.record.load_state_dict(state)

    def penalty(
        self,
        D: torch.nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int = 1,
        collect_stats: bool = True,
        *,
        ema_critic: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Return ``(penalty, stats)`` for one critic step.

        ``penalty`` is a scalar attached to D's graph, or a detached zero when
        the lazy schedule skips ``step``. ``stats`` is empty unless
        ``collect_stats`` (which costs a host sync); then it holds
        ``applied``, ``pen``, ``center``, ``s``, ``prox`` and ``phase``
        (``'a'``, ``'blend'`` or ``'b'`` for K3P). Fixed arms report their
        arm name as ``phase`` and omit ``s``/``prox``. ``ema_critic`` evaluates Dbar for
        this call (default: the anchor). Several calls per critic step (e.g.
        one per role of a shared module) are allowed.
        """
        if self.lazy_k > 1 and step % self.lazy_k != 0:
            zero = torch.zeros((), device=x_real.device, dtype=x_real.dtype)
            return zero, ({"applied": False, "pen": 0.0} if collect_stats else {})
        # Lazy regularization: fewer applications, proportionally bigger hits,
        # so the time-averaged pressure on D is unchanged.
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        if self.arm != "k3p":
            return self._fixed_penalty(D, x_real, x_fake, coefficient, collect_stats)
        return self._k3p_penalty(D, x_real, x_fake, step, coefficient, collect_stats, ema_critic)

    def __call__(self, D, x_real, x_fake, step=1, *, ema_critic=None):
        """Return only the penalty tensor, without collecting synchronized stats."""
        return self.penalty(D, x_real, x_fake, step, collect_stats=False, ema_critic=ema_critic)[0]

    def _fixed_penalty(self, D, x_real, x_fake, coefficient, collect_stats):
        """Released fixed formulas, with opt-in weights on their two means."""
        squared = self.arm == "a_r1r2"
        center = 0.0 if squared else self.kappa
        if (self.real_weight, self.fake_weight) == (1., 1.):
            # Keep the original graph, forward order and arithmetic exactly.
            real_norm = self._grad_norm(D, x_real, squared=squared)
            fake_norm = self._grad_norm(D, x_fake, squared=squared)
            if squared:
                real_term, fake_term = real_norm, fake_norm
            else:
                real_term = F.relu(real_norm - self.kappa).pow(2)
                fake_term = F.relu(fake_norm - self.kappa).pow(2)
            total = real_term.mean() + fake_term.mean()
        else:
            total = x_real.new_zeros(())
            for weight, inputs in ((self.real_weight, x_real), (self.fake_weight, x_fake)):
                if weight == 0:
                    # An inactive term performs no critic forward, autograd or
                    # random draw. Each caller retains its named RNG streams.
                    continue
                norm = self._grad_norm(D, inputs, squared=squared)
                term = norm if squared else F.relu(norm - self.kappa).pow(2)
                total = total + weight * term.mean()
        pen = (coefficient / 2.0) * total
        self.record.calls += 1
        if not collect_stats:
            return pen, {}
        return pen, {"applied": True, "pen": float(pen.detach()), "center": center,
                     "phase": self.arm}

    def _k3p_penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic):
        """K3P, op for op the frozen reports/toy100/gap-fill-20260925 k3p rule."""
        dimension = x_real[0].numel()
        if dimension != x_fake[0].numel():
            raise ValueError("k3p needs reals and fakes of the same per-sample size")
        if step > self.lazy_k and self.record.observed_steps == 0:
            raise RuntimeError(
                "k3p: after_critic_step(critic_optimizer) was never called; "
                "without it s stays 1 (the early form) forever"
            )
        if ema_critic is None:
            owner = getattr(self.anchor, "critic", None)
            if owner is not None and D is not owner:
                raise ValueError(
                    "k3p: D is not the critic this penalty's anchor tracks; "
                    "use one penalty per critic or pass ema_critic= explicitly"
                )
            if self._critic_id is None:
                self._critic_id = id(D)
            elif self._critic_id != id(D):
                raise ValueError(
                    "k3p: this penalty already serves a different critic; "
                    "use one penalty per critic or pass ema_critic= explicitly"
                )
        s = self.blend_weight()
        self.record.calls += 1
        if s >= 1.0:
            real_squared = self._grad_norm(D, x_real, squared=True) / dimension
            fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
            fake_cap = (fake_norm - self.kappa).relu().square()
            pen = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
            prox, phase = None, "a"
        else:
            use_anchor = self.anchor_weight != 0.0
            anchor = ema_critic if ema_critic is not None else self.anchor
            if use_anchor and anchor is None:
                raise ValueError("k3p blended phase needs CriticAnchor/ema_critic")
            x = x_real.detach().clone().requires_grad_(True)
            g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
            sq_r = g.pow(2).flatten(1).sum(dim=1)
            n_r = torch.sqrt(sq_r + 1e-12)
            n_f = self._grad_norm(D, x_fake, squared=False)
            if not self.record.anchor_started:
                if self.anchor is not None:
                    self.anchor.start_()
                self.record.anchor_started = True
                prox = g.new_zeros(())
            elif not use_anchor:
                prox = g.new_zeros(())
            else:
                xb = x_real.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    gb = torch.autograd.grad(_score_scalar(anchor(xb)), xb)[0].detach()
                prox = (g - gb).pow(2).flatten(1).sum(dim=1).mean() / dimension
                if self.anchor_weight != 1.0:
                    prox = self.anchor_weight * prox
            b_term = F.relu(n_r - self.kappa).pow(2).mean() + F.relu(n_f - self.kappa).pow(2).mean() + prox
            if s > 0.0:
                a_term = (sq_r / dimension).mean() + (n_f / dimension ** 0.5 - self.kappa).relu().square().mean()
                pen = (coefficient / 2.0) * (s * a_term + (1.0 - s) * b_term)
                phase = "blend"
            else:
                pen = (coefficient / 2.0) * b_term
                phase = "b"
        if not collect_stats:
            return pen, {}
        return pen, {"applied": True, "pen": float(pen.detach()), "center": self.kappa, "s": s,
                     "prox": 0.0 if prox is None else float(prox.detach()), "phase": phase}

    @staticmethod
    def _grad_norm(D: torch.nn.Module, x: torch.Tensor, squared: bool = False) -> torch.Tensor:
        """Per-sample ``||grad_x D(x)||`` (or its square) with ``create_graph=True``.

        The differentiated scalar is the sum of per-image logit means. A critic
        that returns one logit per image is unchanged.
        """
        x = x.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
        sq = g.pow(2).flatten(1).sum(dim=1)
        if squared:
            return sq
        # Epsilon keeps the sqrt differentiable at g = 0.
        return torch.sqrt(sq + 1e-12)
