"""Paper-watch evaluation metrics (kept out of lib/ on purpose).

Every ``lib/*.py`` file is part of the Forge catalog source digest, so adding
study-only metrics there would invalidate all current Forge measurements.
These metrics are used only by the paper-watch metric studies
(tests/test_gini_distance_metric.py, tests/test_cadma_metrics.py) and live
beside them.

  * `gini_distance_correlation` -- the EmbeddGAN (arXiv:2609.22508) salvage:
    whether a dependence-based discrepancy sees a difference at all;
  * `naeem_coverage` -- the reference k-NN ball-coverage fraction;
  * `cadma_scores` -- the CADMA capacity-aware coverage decomposition
    (arXiv:2611.12243): how much coverage is capacity-supported, and
    whether a coverage failure reads as mode dropping or mode collapse.
"""
from typing import Dict, Optional

import numpy as np
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


# =========================
# CADMA capacity-aware coverage (arXiv:2611.12243)
# =========================

def cadma_scores(
    real,
    fake,
    k: int = 5,
    regularize: bool = True,
) -> Dict[str, float]:
    """
    CADMA capacity-aware coverage decomposition (Meng et al.,
    arXiv:2611.12243, Sections 3.1-3.3).

    Real set R (size N), generated set F (size M; the paper assumes
    N = M). For each real sample r_i, rho_i is the distance to its
    k-th nearest neighbour within R, and the neighbourhood radius is
    regularized (Eq. 10) as

        eps_i = min(rho_i, mean(rho))     if `regularize`, else rho_i.

    The admissible matrix A_ij = 1[ ||r_i - f_j|| <= eps_i ] defines a
    bipartite graph; Z* is a maximum-cardinality matching under A
    (Hopcroft-Karp, via scipy's maximum_bipartite_matching), and
    N_mat = |Z*|. The returned scores are

        C_cap       = N_mat / N                          (Eq. 7)
        Coverage    = N_cov / N,  N_cov = # real samples whose
                      neighbourhood holds >= 1 generated sample
                      (Naeem coverage on the *same* eps radii)
        D_drop      = Coverage - C_cap                   (Eq. 8)
        D_collapse  = (M_cov - N_mat) / N,  M_cov = # generated
                      samples inside >= 1 real neighbourhood
                                                         (Eq. 9)

    so Coverage decomposes exactly as C_cap + D_drop. D_drop counts
    geometrically covered real samples left unmatched for lack of a
    distinct generated counterpart (local under-support); D_collapse
    counts covered generated samples left unmatched because generated
    mass locally outnumbers real samples (local over-concentration).
    Fully uncovered real samples contribute to neither diagnostic --
    they lower C_cap directly (paper, Sec. 3.2 / footnote 1).

    The paper uses k=70 at N=25000; k here is a caller choice and
    should scale with N (the study in tests/test_cadma_metrics.py
    uses k=5 at N=1400 and reports a k-sensitivity grid).

    Args:
        real, fake: (N, D) and (M, D) sample sets (tensor or array).
        k: neighbour order for the real-side radii.
        regularize: apply the Eq. 10 mean-radius cap.

    Returns:
        {'c_cap', 'coverage', 'd_drop', 'd_collapse', 'n_matched',
         'n_real_covered', 'n_fake_covered'} -- scores as floats,
        counts as ints.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    a = np.asarray(real, dtype=np.float64)
    b = np.asarray(fake, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected two (N, D) sample sets of equal width")
    n, m = a.shape[0], b.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be in [1, len(real))")
    if n == 0 or m == 0:
        raise ValueError("sample sets must be non-empty")

    # Real-real distances: rho_i = distance to k-th NN (excluding self).
    d_rr = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d_rr, np.inf)
    rho = np.partition(d_rr, k - 1, axis=1)[:, k - 1]
    eps = np.minimum(rho, rho.mean()) if regularize else rho

    # Admissible matrix A (sparse): ||r_i - f_j|| <= eps_i.
    d_rf = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    rows, cols = np.nonzero(d_rf <= eps[:, None])
    adjacency = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, m)
    )

    # Maximum-cardinality bipartite matching (Hopcroft-Karp).
    match = maximum_bipartite_matching(adjacency, perm_type="row")
    n_mat = int((match >= 0).sum())

    # Row counts > 0  <=> real sample covered; column counts likewise.
    row_has = np.diff(adjacency.indptr) > 0
    col_has = np.asarray(adjacency.sum(axis=0)).ravel() > 0
    n_cov = int(row_has.sum())
    m_cov = int(col_has.sum())

    c_cap = n_mat / n
    coverage = n_cov / n
    return {
        "c_cap": float(c_cap),
        "coverage": float(coverage),
        "d_drop": float(coverage - c_cap),
        "d_collapse": float((m_cov - n_mat) / n),
        "n_matched": n_mat,
        "n_real_covered": n_cov,
        "n_fake_covered": m_cov,
    }
