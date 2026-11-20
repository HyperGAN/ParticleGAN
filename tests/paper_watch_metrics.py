"""Paper-watch evaluation metrics (kept out of lib/ on purpose).

Every ``lib/*.py`` file is part of the Forge catalog source digest, so adding
study-only metrics there would invalidate all current Forge measurements.
These metrics are used only by the paper-watch metric studies
(tests/test_gini_distance_metric.py, tests/test_cadma_metrics.py) and live
beside them.

  * `gini_distance_correlation` -- the EmbeddGAN (arXiv:2609.22508) salvage:
    whether a dependence-based discrepancy sees a difference at all;
  * `naeem_coverage` -- the reference k-NN ball-coverage fraction.
"""
from typing import Optional

import torch


# =========================
# Gini distance correlation + Naeem coverage
# =========================

def gini_distance_correlation(
    real: torch.Tensor,
    fake: torch.Tensor,
    max_n: int = 1024,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    Empirical Gini distance correlation rho_g between two sample sets.

    This is the evaluation-metric salvage of EmbeddGAN (arXiv:2609.22508),
    which trains against rho_g in a learned embedding space. Used here on
    raw samples it is a plain two-sample statistic: for equal-size sets
    A (real) and B (fake) of size m,

        gCov(A, B) = (1/m^2) * sum_ij ||a_i - b_j||
                     - (1/(2m(m-1))) * sum_{i != j}
                           [||a_i - a_j|| + ||b_i - b_j||]
        Delta(X)   = (1/(2m)^2) * sum over all pairs in X = A u B
        rho_g      = gCov / Delta

    The statistic is zero iff the two distributions agree and grows with
    their discrepancy (toward 2 in the disjoint point-mass limit under this
    normalization); the finite-sample estimate can be marginally negative
    for near-identical distributions, and is returned unclamped.
    Both sets are subsampled (without replacement, via `generator`) to
    m = min(len(real), len(fake), max_n) so the O(m^2) pairwise cost is
    bounded.
    """
    with torch.no_grad():
        a = torch.as_tensor(real).detach().to(torch.float64)
        b = torch.as_tensor(fake).detach().to(torch.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("expected two (N, D) sample sets of equal width")
        m = int(min(a.shape[0], b.shape[0], max_n))
        if m < 2:
            raise ValueError("need at least 2 samples per set")

        def _sub(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] == m:
                return x
            idx = torch.randperm(x.shape[0], generator=generator)[:m]
            return x[idx.to(x.device)]

        a, b = _sub(a), _sub(b)
        d_ab = torch.cdist(a, b)
        d_aa = torch.cdist(a, a)
        d_bb = torch.cdist(b, b)
        off = ~torch.eye(m, dtype=torch.bool, device=a.device)
        cross = d_ab.mean()
        within = 0.5 * (d_aa[off].mean() + d_bb[off].mean())
        gcov = cross - within
        x = torch.cat([a, b], dim=0)
        delta = torch.cdist(x, x).mean()
        return float(gcov / delta)


def naeem_coverage(
    real: torch.Tensor,
    fake: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Naeem et al. (2020) coverage: the fraction of real samples whose
    k-nearest-real-neighbour ball contains at least one fake sample.

    Unlike precision/recall this is computed per real sample against the
    real manifold only, so a generator that drops a mode loses exactly
    that mode's share of the score.
    """
    with torch.no_grad():
        a = torch.as_tensor(real).detach().to(torch.float64)
        b = torch.as_tensor(fake).detach().to(torch.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("expected two (N, D) sample sets of equal width")
        if not 1 <= k < a.shape[0]:
            raise ValueError("k must be in [1, len(real))")
        d_rr = torch.cdist(a, a)
        radius = d_rr.kthvalue(k + 1, dim=1).values  # +1 skips the self-distance
        nearest_fake = torch.cdist(a, b).min(dim=1).values
        return float((nearest_fake <= radius).to(torch.float64).mean())
