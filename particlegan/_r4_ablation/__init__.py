"""Round-4 Occam's Razor systematic 1-factor knockout ablations on `unified_v4`.

Isolates every mechanism in `particlegan/_unified_v4/` without modifying any pre-existing
package (`_unified_v4/`, `_nash_v3/`, `_cond_fiber/`, `_unified/`, `recipe_compat.py`)
or adding any fields to `Recipe` (`179` fields preserved):

- `optimizer_family="r4_mp_no_ns"` (`R4MPNoNSOptimizer`): replaces 5-step Quintic
  Newton-Schulz polar orthogonalization with pure-elementwise `O(params)` $10^{-38}$-safe
  `peak_rms` + Marchenko-Pastur RMS-to-RMS operator normalization (`0` SVD, `0` NS).
- `optimizer_family="r4_no_secant_band"` (`R4NoSecantBandOptimizer`): disables realized
  function-space pushforward/logit secant trust-band regulation (`alpha_gain == 1.0`).
- `critic_formulation="r4_autograd_critic"` (`R4AutogradCriticPenalty`): replaces 1st-order
  `fwAD`/secant critic penalty with 2nd-order `torch.autograd.grad(..., create_graph=True)`
  Born-Infeld EBI penalty (`_T1CouplingPenaltyKernel`) for matched quality-vs-energy timing.
- `prior_regularizer` ablations (`R4AblationPriorRegularizer` / `r4_ablation_pullback_step`):
  - `"r4_no_pullback_iso"`: disables 2D-complex & Radon-Hurwitz pushforward covariance isometry.
  - `"r4_no_coalesce_ltv"`: disables externally isolated Pythagorean sibling coalescence & LTV.
  - `"r4_no_chart_laguerre"`: disables Laguerre dual-potential `psi` updates (paired with
    `t1c_chart_weight=0.0` for complete streaming $H_0$-screened Laguerre chart knockout).
  - `"r4_no_sat_probe"`: disables `fwAD` saturation inverse-attenuation & outward dead-clamp rescue.
"""
from __future__ import annotations

from .optimizer import R4MPNoNSOptimizer, R4NoSecantBandOptimizer
from .penalty import R4AutogradCriticPenalty
from .prior import (
    R4_ABLATION_PRIOR_REGULARIZERS,
    R4AblationPriorRegularizer,
    r4_ablation_pullback_step,
    should_use_r4_ablation_pullback,
)

__all__ = [
    "R4_ABLATION_PRIOR_REGULARIZERS",
    "R4AblationPriorRegularizer",
    "R4AutogradCriticPenalty",
    "R4MPNoNSOptimizer",
    "R4NoSecantBandOptimizer",
    "r4_ablation_pullback_step",
    "should_use_r4_ablation_pullback",
]
