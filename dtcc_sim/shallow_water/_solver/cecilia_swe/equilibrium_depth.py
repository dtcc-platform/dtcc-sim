"""Local nodal lake profile with the same DG1 triangle mean depth."""
import numpy as np


def reconstruct(depth, bed):
    depth = np.asarray(depth, dtype=float)
    bed = np.asarray(bed, dtype=float)
    if depth.shape != bed.shape or depth.ndim != 2 or depth.shape[1] != 3:
        raise ValueError('Expected matching (cell,3) nodal depth and bed arrays')
    if not np.isfinite(depth).all() or not np.isfinite(bed).all():
        raise ValueError('Nonfinite local state')
    total = depth.sum(axis=1)
    if np.any(total < -3e-14):
        raise ValueError('Negative mean depth in equilibrium reconstruction')
    total = np.maximum(total, 0.)
    relative = bed - bed.min(axis=1)[:, None]
    ordered = np.sort(relative, axis=1)
    level = total.copy()
    two = level > ordered[:, 1]
    level[two] = (total[two] + ordered[two, 1]) / 2
    three = two & (level > ordered[:, 2])
    level[three] = (total[three] + ordered[three, 1] + ordered[three, 2]) / 3
    return np.maximum(level[:, None] - relative, 0.)
