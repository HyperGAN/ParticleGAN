"""
Distributional metrics for the 2D Gaussian-grid benchmark.

`mode_coverage` (lib/toy_models.py) answers "did the generator find the modes",
which is the coarse question. These functions answer the finer ones a controlled
comparison of gradient penalties needs:

  * how far is the model distribution from the data distribution
    (`sliced_w1`, cheap and GPU-resident; `exact_w1_w2`, exact but O(n^2 log n)),
  * how plausible generated samples are under the target (`mixture_nll`,
    which rewards concentration and is not a calibration metric),
  * how balanced is its mass across the 100 modes
    (`mode_recall_and_hists`).

Everything takes samples as (N, 2) tensors on any device. The two torch-only
metrics stay on the input device; the OT one moves to numpy float64.
"""

from typing import Dict, Optional, Tuple

import math

import numpy as np
import torch


# =========================
#  Grid helpers
# =========================

def _grid_centers(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    grid_scale: float = 1.0,
) -> torch.Tensor:
    """The 100 mixture centers as a (100, 2) tensor, in row-major (x, y) order."""
    coords = (torch.arange(10, device=device, dtype=dtype) - 4.5) * grid_scale
    cx, cy = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([cx.flatten(), cy.flatten()], dim=1)


# =========================
#  Sliced Wasserstein-1
# =========================

def sliced_w1(
    x: torch.Tensor,
    y: torch.Tensor,
    n_proj: int = 128,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    Sliced Wasserstein-1 distance between two point clouds.

    Projects both clouds onto `n_proj` random unit directions, computes the 1D
    W1 distance per direction as the mean absolute difference of the sorted
    projections, and averages over directions. Pure torch, so it runs on the
    GPU and never leaves the device until the final scalar.

    The two clouds must have the same number of points (the 1D W1 shortcut
    "sort both, take mean |difference|" is only valid for equal-size uniform
    empirical measures).

    Args:
        x, y: (N, 2) samples.
        n_proj: number of random projection directions.
        seed: if given, projections are drawn from a fresh generator with this
            seed, making the estimate reproducible. Ignored if `generator` is
            passed.
        generator: explicit torch.Generator (must live on `x`'s device).

    Returns:
        The sliced W1 estimate as a python float.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"sliced_w1 needs equal sample counts, got {x.shape[0]} vs {y.shape[0]}"
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"dim mismatch: {x.shape[1]} vs {y.shape[1]}")

    device = x.device
    dim = x.shape[1]

    if generator is None and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    with torch.no_grad():
        if generator is None:
            proj = torch.randn(dim, n_proj, device=device, dtype=x.dtype)
        else:
            proj = torch.randn(
                dim, n_proj, device=device, dtype=x.dtype, generator=generator
            )
        proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-12)

        px, _ = (x @ proj).sort(dim=0)  # (N, n_proj)
        py, _ = (y.to(x.dtype) @ proj).sort(dim=0)
        return float((px - py).abs().mean())


# =========================
#  Exact optimal transport
# =========================

def exact_w1_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: int = 4096,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """
    Exact Wasserstein-1 and Wasserstein-2 via the POT network-simplex solver.

    Both clouds are (uniformly, without replacement) subsampled to at most
    `max_points` points, given uniform weights, and matched exactly:

        w1 = emd2(a, b, C)          with C_ij = ||x_i - y_j||
        w2 = sqrt(emd2(a, b, C^2))  with C_ij = ||x_i - y_j||^2

    O(n^2) memory in the cost matrix and roughly O(n^3 log n) time, hence the
    subsample cap. Everything is converted to numpy float64 before the solve,
    because the simplex is sensitive to float32 cost ties.

    Returns:
        (w1, w2) as python floats.
    """
    import ot  # imported lazily: POT is only needed for the final eval

    rng = np.random.default_rng(seed)

    def _prep(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().to("cpu", torch.float64).numpy()
        if arr.shape[0] > max_points:
            sel = rng.choice(arr.shape[0], size=max_points, replace=False)
            arr = arr[sel]
        return np.ascontiguousarray(arr)

    xa = _prep(x)
    ya = _prep(y)

    a = np.full(xa.shape[0], 1.0 / xa.shape[0], dtype=np.float64)
    b = np.full(ya.shape[0], 1.0 / ya.shape[0], dtype=np.float64)

    m2 = ot.dist(xa, ya, metric="sqeuclidean")  # squared Euclidean cost
    m1 = np.sqrt(m2)

    w1 = float(ot.emd2(a, b, m1, numItermax=1_000_000))
    w2_sq = float(ot.emd2(a, b, m2, numItermax=1_000_000))
    return w1, math.sqrt(max(w2_sq, 0.0))


# =========================
#  Analytic likelihood
# =========================

def mixture_nll(
    x: torch.Tensor,
    std: float = 0.03,
    grid_scale: float = 1.0,
) -> float:
    """
    Mean negative log-likelihood of `x` under the analytic 100-Gaussian mixture.

    The density is an equal-weight mixture of isotropic 2D Gaussians:

        p(x) = sum_k (1/100) * N(x; mu_k, std^2 I)

    so, with d = 2,

        log p(x) = logsumexp_k [ -||x - mu_k||^2 / (2 std^2) ] - log 100
                   - (d/2) log(2 pi std^2)

    This measures sample plausibility, not distributional calibration. Samples
    concentrated at mode centers score lower than samples from the true target.
    Target samples have expected NLL approximately
    d/2 * (1 + log(2 pi std^2)) + log 100 nats; this is not a lower bound.

    Returns:
        The mean NLL in nats, as a python float.
    """
    with torch.no_grad():
        xf = x.detach().to(torch.float64)
        centers = _grid_centers(xf.device, torch.float64, grid_scale)

        d = xf.shape[1]
        var = float(std) ** 2
        sq = torch.cdist(xf, centers).pow(2)  # (N, 100)

        log_comp = -sq / (2.0 * var)
        log_p = (
            torch.logsumexp(log_comp, dim=1)
            - math.log(centers.shape[0])
            - 0.5 * d * math.log(2.0 * math.pi * var)
        )
        return float(-log_p.mean())


# =========================
#  Mode balance
# =========================

def mode_recall_and_hists(
    x: torch.Tensor,
    n_expected_frac: float = 0.5,
    std: float = 0.03,
    grid_scale: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Mode recall and mass-balance divergences from a nearest-center assignment.

    Every sample is assigned to its nearest of the 100 grid centers (no
    distance threshold: this measures *balance*, not sharpness, which
    `mode_coverage` and `mixture_nll` already cover). From the resulting
    100-cell histogram:

      - mode_recall: fraction of the 100 modes whose count is at least
        `n_expected_frac * N / 100`, i.e. that hold at least that share of the
        mass a uniform model would give them. Recall 1.0 means no mode is
        starved.
      - hist_kl: KL(empirical || uniform) in nats. 0 for perfect balance,
        log(100) ~ 4.605 for total collapse onto one mode.
      - hist_js: Jensen-Shannon divergence between the same two histograms, in
        nats. Bounded by log 2, so it stays informative when KL saturates.

    `eps` smooths both histograms before the logs.

    Returns:
        {'mode_recall': float, 'hist_kl': float, 'hist_js': float}
    """
    with torch.no_grad():
        xf = x.detach().to(torch.float64)
        centers = _grid_centers(xf.device, torch.float64, grid_scale)
        k = centers.shape[0]

        nearest = torch.cdist(xf, centers).argmin(dim=1)
        counts = torch.bincount(nearest, minlength=k).to(torch.float64)

        n = float(xf.shape[0])
        threshold = n_expected_frac * n / k
        mode_recall = float((counts >= threshold).to(torch.float64).mean())

        p = counts / counts.sum().clamp_min(eps) + eps
        p = p / p.sum()
        q = torch.full_like(p, 1.0 / k)

        hist_kl = float((p * (p / q).log()).sum())

        m = 0.5 * (p + q)
        hist_js = float(
            0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum()
        )

    return {
        "mode_recall": mode_recall,
        "hist_kl": hist_kl,
        "hist_js": hist_js,
    }


# =========================
#  Per-mode shape audit
# =========================

def per_mode_moments(
    x: torch.Tensor,
    min_count: int = 50,
    std: float = 0.03,
    grid_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Within-mode spread and placement, from a nearest-center assignment.

    `mode_recall_and_hists` asks whether the mass is spread evenly over the 100
    modes; this asks whether each individual blob has the right *shape*. Every
    sample is assigned to its nearest grid center, and each mode holding at
    least `min_count` samples contributes:

      - its std, sqrt(mean over the 2 dims of the within-mode population
        variance), which the data generator sets to `std` (0.03),
      - its center bias, ||sample mean - true center||.

    Modes with fewer than `min_count` samples are dropped: a handful of points
    gives a variance estimate that is mostly noise, and a mode the generator
    has barely found says more about coverage (already measured) than shape.

    Returns:
        {
          'per_mode_std':       mean per-mode std over qualifying modes,
          'per_mode_std_ratio': that std divided by the true `std`
                                (1.0 is correct; > 1 is over-dispersed,
                                < 1 is a too-sharp / partially collapsed blob),
          'center_bias':        mean ||sample mean - true center||,
          'audited_modes':      how many modes qualified,
        }
        The three floats are nan when no mode qualifies.
    """
    with torch.no_grad():
        xf = x.detach().to(torch.float64)
        centers = _grid_centers(xf.device, torch.float64, grid_scale)
        k, d = centers.shape

        nearest = torch.cdist(xf, centers).argmin(dim=1)
        counts = torch.bincount(nearest, minlength=k).to(torch.float64)

        sums = torch.zeros(k, d, dtype=torch.float64, device=xf.device)
        sums.index_add_(0, nearest, xf)
        sq_sums = torch.zeros(k, d, dtype=torch.float64, device=xf.device)
        sq_sums.index_add_(0, nearest, xf.pow(2))

        qualifies = counts >= float(min_count)
        audited = int(qualifies.sum())
        if audited == 0:
            nan = float("nan")
            return {
                "per_mode_std": nan,
                "per_mode_std_ratio": nan,
                "center_bias": nan,
                "audited_modes": 0,
            }

        n = counts[qualifies].unsqueeze(1)
        means = sums[qualifies] / n
        # E[x^2] - E[x]^2, clamped: it is exact in float64 here, but the
        # subtraction is the one place a pathological batch could go negative.
        var = (sq_sums[qualifies] / n - means.pow(2)).clamp_min(0.0)

        per_mode_std = float(var.mean(dim=1).sqrt().mean())
        center_bias = float((means - centers[qualifies]).norm(dim=1).mean())

    return {
        "per_mode_std": per_mode_std,
        "per_mode_std_ratio": per_mode_std / float(std),
        "center_bias": center_bias,
        "audited_modes": audited,
    }


# The far-tail cut, as a multiple of the data's within-mode sigma. 10 sigma_true
# is 0.30 in absolute coordinates: a third of the way to the neighbouring mode,
# so far outside its own blob that no Gaussian tail reaches it (1e-22 of the
# mass), yet still short of the Voronoi boundary at 0.5.
TAIL_SIGMAS = 10.0

# median ||x - mu|| / sigma for a 2-D isotropic Gaussian: sqrt(2 ln 2) = 1.17741.
RAYLEIGH_MEDIAN_FACTOR = math.sqrt(2.0 * math.log(2.0))


def per_mode_core_ratio(
    x: torch.Tensor,
    min_count: int = 50,
    std: float = 0.03,
    grid_scale: float = 1.0,
    center: str = "median",
) -> Dict[str, float]:
    """
    Within-mode *core* width, measured so the far tail cannot inflate it.

    `per_mode_moments` reports a second moment about each mode's sample mean.
    On these models that is not a width: a small population stranded whole grid
    cells away dominates the variance *and* drags the mean the variance is taken
    about, so a blob whose core is tighter than the data's can still report a
    std ratio of 5 (see results/metric_recon.md, which established this and from
    which this estimator is taken verbatim). Two changes make the estimate read
    the core instead:

      * **center** -- the coordinate-wise *median* of the mode's samples, not
        their mean. A 3% tail at distance 2.5 moves the mean by 2.5 sigma_true
        and every distance measured from it; it moves the median by nothing.
      * **spread** -- the *median* radial distance, inverted through the 2-D
        isotropic Gaussian relation median ||x - mu|| = sigma * sqrt(2 ln 2).
        Everything past the median is ignored by construction, so the estimate
        depends only on the half of the mass nearest the center.

    This scalar assumes an isotropic Gaussian core. A matching radial median
    alone cannot establish Gaussian shape or exclude collapse along one axis;
    inspect `per_mode_covariance_ratios` and radial/angular structure as well.

    On a genuine Gaussian this and `per_mode_std` agree, so their ratio reads
    directly as a tail-inflation factor. `tail_frac_10sigma` is the mass that
    buys that inflation: the fraction of *all* samples whose nearest-center
    distance exceeds 10 * `std`.

    Modes are assigned and filtered exactly as in `per_mode_moments`: nearest
    center, and only modes holding at least `min_count` samples contribute.

    Args:
        x: (N, 2) samples; a numpy array is accepted too.
        min_count: qualifying threshold per mode, as in `per_mode_moments`.
        std: the data's true within-mode sigma, the ratio's denominator.
        grid_scale: spacing between neighbouring mode centers.
        center: which center to measure the radial distances from --
            "median" (the default and the robust choice), "mean" (the audit's
            center, kept so the contamination can be quantified), or "true"
            (the grid center, which is what `mode_coverage`'s `hq` measures
            from).

    Returns:
        {
          'per_mode_core_std':   mean per-mode core sigma over qualifying modes,
          'per_mode_core_ratio': that sigma divided by the true `std`
                                 (1.0 is correct; < 1 is a compressed core,
                                 which the raw std ratio can hide entirely),
          'tail_frac_10sigma':   fraction of samples further than 10 * `std`
                                 from their nearest center,
        }
        The two core floats are nan when no mode qualifies.
    """
    if center not in ("median", "mean", "true"):
        raise ValueError(f"center must be median/mean/true, got {center!r}")

    with torch.no_grad():
        xf = torch.as_tensor(x).detach().to(torch.float64)
        centers = _grid_centers(xf.device, torch.float64, grid_scale)
        k = centers.shape[0]

        d = torch.cdist(xf, centers)  # (N, 100)
        nearest = d.argmin(dim=1)
        dist = d.gather(1, nearest.unsqueeze(1)).squeeze(1)
        counts = torch.bincount(nearest, minlength=k)

        tail_frac = float(
            (dist > TAIL_SIGMAS * float(std)).to(torch.float64).mean()
        )

        sigmas: list = []
        for m in torch.nonzero(counts >= int(min_count), as_tuple=False).flatten():
            pts = xf[nearest == m]
            if center == "median":
                ctr = pts.quantile(0.5, dim=0)
            elif center == "mean":
                ctr = pts.mean(dim=0)
            else:
                ctr = centers[m]
            r = (pts - ctr).norm(dim=1)
            sigmas.append(float(r.quantile(0.5)) / RAYLEIGH_MEDIAN_FACTOR)

    core_std = float(np.mean(sigmas)) if sigmas else float("nan")
    return {
        "per_mode_core_std": core_std,
        "per_mode_core_ratio": core_std / float(std),
        "tail_frac_10sigma": tail_frac,
    }


def per_mode_covariance_ratios(
    x: torch.Tensor,
    data_std: float = 0.03,
    grid_scale: float = 1.0,
    min_count: int = 50,
) -> Dict[str, float]:
    """Mean smallest/largest within-mode covariance eigenvalues / true variance.

    These complement the robust radial median: a line or two-point cloud can
    have the correct median radius while its smallest eigenvalue is zero.
    Covariances use population normalization and are tail-sensitive; report
    both these ratios and the core/tail metrics. Values near one are necessary
    second-moment checks, not proof of Gaussian shape. Modes below min_count
    are omitted and the audited count must accompany the ratios.
    """
    if not math.isfinite(data_std) or data_std <= 0:
        raise ValueError("data_std must be positive and finite")
    if min_count < 2:
        raise ValueError("min_count must be at least 2")
    with torch.no_grad():
        xf = torch.as_tensor(x).detach().to(torch.float64)
        if xf.ndim != 2 or xf.shape[1] != 2:
            raise ValueError("expected samples with shape (N, 2)")
        centers = _grid_centers(xf.device, torch.float64, grid_scale)
        nearest = torch.cdist(xf, centers).argmin(dim=1)
        counts = torch.bincount(nearest, minlength=centers.shape[0])
        eigenvalues = []
        for mode in torch.nonzero(counts >= min_count).flatten():
            points = xf[nearest == mode]
            residuals = points - points.mean(dim=0)
            cov = residuals.T @ residuals / points.shape[0]
            eigenvalues.append(torch.linalg.eigvalsh(cov).clamp_min(0) / data_std**2)
        means = torch.stack(eigenvalues).mean(dim=0) if eigenvalues else None
    return {
        "per_mode_cov_eig_min_ratio": float(means[0]) if means is not None else float("nan"),
        "per_mode_cov_eig_max_ratio": float(means[1]) if means is not None else float("nan"),
        "per_mode_cov_audited_modes": len(eigenvalues),
    }
