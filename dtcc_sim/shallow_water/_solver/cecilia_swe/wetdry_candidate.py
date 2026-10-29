"""Experimental DG1 positivity rescaling and dry-node momentum cutoff."""

import numpy as np


class NegativeCellMean(ValueError):
    pass


def limit(values, *, dry_depth=1e-6, roundoff_tolerance=1e-14):
    """Return limited (cell, triangle node, component) values and diagnostics.

    Rescale around cell means to make linear depth nonnegative. Apply the same
    rescaling to momentum, then zero momentum at dry nodes. Only the final dry
    cutoff intentionally changes momentum means. Tiny negative depth means
    within roundoff_tolerance are zeroed and explicitly measured by the caller.
    """
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not np.isfinite(values).all():
        raise ValueError("Expected finite DG1 triangle states of shape (cells, 3, 3)")
    means = values.mean(axis=1)
    minimum_mean = float(means[:, 0].min())
    if minimum_mean < -roundoff_tolerance:
        raise NegativeCellMean(f"Negative mean depth {minimum_mean:.6e}")
    depth_means = np.maximum(means[:, 0], 0.)
    minimum = values[:, :, 0].min(axis=1)
    theta = np.ones(len(values))
    needs_limit = minimum < 0.
    theta[needs_limit] = depth_means[needs_limit] / (depth_means[needs_limit]-minimum[needs_limit])
    theta[needs_limit] *= 1-8*np.finfo(float).eps
    limited = means[:, None, :] + theta[:, None, None]*(values-means[:, None, :])
    limited[:, :, 0] += (depth_means-means[:, 0])[:, None]
    # Record any machine-roundoff correction in the measured mass budget.
    limited[:, :, 0] = np.maximum(limited[:, :, 0], 0.)
    dry = limited[:, :, 0] <= dry_depth
    momentum_before = limited[:, :, 1:].copy()
    limited[:, :, 1:][dry] = 0.
    return limited, {
        "limited_cells": int(np.count_nonzero(needs_limit)),
        "roundoff_negative_mean_cells": int(np.count_nonzero(means[:, 0] < 0.)),
        "minimum_cell_mean_before": minimum_mean,
        "max_cell_mean_depth_change": float(np.max(np.abs(limited[:, :, 0].mean(axis=1)-means[:, 0]))),
        "dry_nodes": int(np.count_nonzero(dry)),
        "momentum_cutoff_l1_coefficients": float(np.sum(np.abs(limited[:, :, 1:]-momentum_before))),
    }
