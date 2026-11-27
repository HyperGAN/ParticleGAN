"""Direct tests for ``particlegan.execution`` and the deterministic ``_qr`` draws.

``r2_points`` underlies the public deterministic initializer's particle
prior, so its shape, range, determinism and low-discrepancy property are
pinned here over a grid of sizes and dimensions.
"""
import threading

import numpy as np
import pytest
import torch

from particlegan import _qr
from particlegan.execution import disable_autograd_multithreading, serial_autograd


# execution -------------------------------------------------------------------

def test_serial_autograd_disables_and_then_restores_the_caller_state():
    prev = torch.autograd.is_multithreading_enabled()
    torch.autograd.set_multithreading_enabled(True)
    try:
        with serial_autograd():
            assert torch.autograd.is_multithreading_enabled() is False
        assert torch.autograd.is_multithreading_enabled() is True
        with pytest.raises(RuntimeError, match="boom"):
            with serial_autograd():
                raise RuntimeError("boom")
        assert torch.autograd.is_multithreading_enabled() is True
    finally:
        torch.autograd.set_multithreading_enabled(prev)


def test_disable_autograd_multithreading_applies_to_the_calling_thread_only():
    prev = torch.autograd.is_multithreading_enabled()
    torch.autograd.set_multithreading_enabled(True)
    seen = {}

    def worker():
        disable_autograd_multithreading()
        seen["worker"] = torch.autograd.is_multithreading_enabled()

    try:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert seen["worker"] is False
        assert torch.autograd.is_multithreading_enabled() is True
    finally:
        torch.autograd.set_multithreading_enabled(prev)


# _qr.r2_points -----------------------------------------------------------------

def _star_discrepancy_bound(points, grid=8):
    """Max |empirical - volume| over anchored boxes [0, t) on a fixed grid."""
    corners = np.stack(np.meshgrid(*[np.arange(1, grid + 1) / grid] * points.shape[1], indexing="ij"), -1)
    corners = corners.reshape(-1, points.shape[1])
    inside = (points[None, :, :] < corners[:, None, :]).all(-1).mean(1)
    return float(np.abs(inside - corners.prod(1)).max())


@pytest.mark.parametrize("n,d", [(1, 1), (7, 2), (64, 2), (257, 3), (1000, 4), (16, 9)])
def test_r2_points_shape_range_dtype_and_determinism(n, d):
    points = _qr.r2_points(n, d)
    assert points.shape == (n, d) and points.dtype == torch.float64
    assert bool(torch.isfinite(points).all())
    assert bool(((points >= 0) & (points < 1)).all())
    assert torch.equal(points, _qr.r2_points(n, d))
    assert torch.equal(_qr.r2_points(n + 5, d)[:n], points)  # an extensible sequence


def test_r2_first_point_and_one_dimensional_golden_ratio_step():
    golden = (1 + 5 ** .5) / 2
    points = _qr.r2_points(4, 1)[:, 0].numpy()
    np.testing.assert_allclose(points, np.mod(.5 + np.arange(1, 5) / golden, 1.), rtol=0, atol=1e-12)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_r2_points_have_low_discrepancy_compared_with_iid_uniform_draws(d):
    n = 1024
    r2 = _qr.r2_points(n, d).numpy()
    r2_error = _star_discrepancy_bound(r2)
    # Low discrepancy: O(log(n)^d / n) rather than the O(n^-1/2) of iid draws.
    assert r2_error < 4 * np.log(n) ** d / n
    iid = np.random.default_rng(0).random((32, n, d))
    iid_errors = [_star_discrepancy_bound(sample) for sample in iid]
    assert r2_error < min(iid_errors)
    # Every coordinate is equidistributed across ten equal bins, far inside
    # the iid binomial spread (one standard deviation is about 9.6 rows).
    counts = np.stack([np.histogram(r2[:, j], bins=10, range=(0, 1))[0] for j in range(d)])
    assert np.abs(counts - n / 10).max() <= 8


def test_r2_points_do_not_read_or_consume_global_rng():
    torch.manual_seed(3)
    state = torch.get_rng_state()
    np_state = np.random.get_state()[1].copy()
    _qr.r2_points(32, 3)
    assert torch.equal(state, torch.get_rng_state())
    assert np.array_equal(np_state, np.random.get_state()[1])


# _qr.semi_orthogonal and pattern ------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(5, 3), (3, 5), (4, 4)])
def test_semi_orthogonal_has_orthonormal_short_side_and_is_keyed(rows, cols):
    matrix = _qr.semi_orthogonal(_qr.key("layer", rows, cols), rows, cols)
    assert matrix.shape == (rows, cols) and matrix.dtype == torch.float64
    gram = matrix.T @ matrix if rows >= cols else matrix @ matrix.T
    torch.testing.assert_close(gram, torch.eye(min(rows, cols), dtype=torch.float64))
    assert torch.equal(matrix, _qr.semi_orthogonal(_qr.key("layer", rows, cols), rows, cols))
    assert not torch.equal(matrix, _qr.semi_orthogonal(_qr.key("other", rows, cols), rows, cols))


def test_pattern_has_the_declared_moments():
    values = _qr.pattern(_qr.key("bias"), (6, 5), .25, 2.)
    assert values.shape == (6, 5)
    assert float(values.mean()) == pytest.approx(.25, abs=1e-12)
    assert float(values.std(unbiased=False)) == pytest.approx(2., abs=1e-12)
    single = _qr.pattern(_qr.key("bias"), (1,), .5, 3.)
    assert single.shape == (1,) and bool(torch.isfinite(single).all())
