"""Independent target draws and grouped-bag scores for the wide conditional toy."""
import numpy as np


WIDTH = 65
BAGS = 2048
BAG_SIZE = 32
SCALES = np.array([[.5, 2.], [2., .5]])


def target_signals(seed):
    shape = (2, BAGS, BAG_SIZE, 2)
    return np.random.default_rng(seed).standard_normal(shape) * SCALES[:, None, None, :]


def bag_summaries(signals):
    return np.stack((signals.mean(axis=2), signals.std(axis=2, ddof=1)), axis=-1)


def intervals(signals, truth):
    low, high = np.quantile(bag_summaries(signals), [.1, .9], axis=1)
    below = low[:, None] - truth
    above = truth - high[:, None]
    width = high - low
    coverage = ((below <= 0) & (above <= 0)).mean(axis=1)
    score = width + 10 * (np.maximum(below, 0) + np.maximum(above, 0)).mean(axis=1)
    return coverage, width, score


def score_samples(samples):
    """Rows are grouped by the condition actually supplied to the generator."""
    if samples.shape != (2, BAGS * BAG_SIZE, WIDTH) or not np.isfinite(samples).all():
        raise ValueError("expected finite raw draws for both groups and all 65 coordinates")
    signals = samples[:, :, :2].reshape(2, BAGS, BAG_SIZE, 2)
    truth = bag_summaries(target_signals(102))
    coverage, width, score = intervals(signals, truth)
    _, _, reference = intervals(target_signals(100), truth)
    means = signals.mean(axis=(1, 2))
    sds = signals.std(axis=(1, 2), ddof=1)
    metrics = dict(
        max_scaled_mean_error=float(np.max(np.abs(means) / SCALES)),
        max_relative_sd_error=float(np.max(np.abs(sds / SCALES - 1))),
        max_coverage_error=float(np.abs(coverage - .8).max()),
        combined_coverage=float(coverage.mean()),
        variability_coverage=float(coverage[:, :, 1].mean()),
        worst_interval_score_ratio=float(np.max(score / reference)),
        max_group_padding_energy=float(np.square(samples[:, :, 2:], dtype=np.float64).sum(2).mean(1).max()),
    )
    for group in range(2):
        for axis in range(2):
            prefix = f"group{group}_axis{axis}"
            metrics[f"{prefix}_mean"] = float(means[group, axis])
            metrics[f"{prefix}_sd"] = float(sds[group, axis])
            for index, name in enumerate(("mean", "sd")):
                for key, values in (("coverage", coverage), ("width", width), ("interval_score", score)):
                    metrics[f"{prefix}_{name}_{key}"] = float(values[group, axis, index])
    return metrics
