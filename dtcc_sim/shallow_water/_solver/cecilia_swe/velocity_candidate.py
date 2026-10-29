"""Experimental momentum-conserving velocity bounds from edge neighbors.

Uses a convex blend toward each cell's depth-weighted mean velocity.
This is not the three-vertex selection algorithm of Vater et al.
"""
import numpy as np
from .wetdry_candidate import limit as depth_limit

_neighbors = None


def configure(neighbors):
    global _neighbors
    table = np.asarray(neighbors, dtype=np.int64)
    if table.ndim != 2 or table.shape[1] != 4 or table.min()<0 or table.max()>=len(table):
        raise ValueError('Expected self and three edge neighbors per triangle')
    if not np.array_equal(table[:,0],np.arange(len(table))):
        raise ValueError('First neighbor must be self')
    _neighbors = table


def limit(values, *, dry_depth=1e-6, roundoff_tolerance=1e-14):
    out, stats = depth_limit(values, dry_depth=dry_depth, roundoff_tolerance=roundoff_tolerance)
    if _neighbors is None or len(_neighbors)!=len(out):
        raise ValueError('Configure mesh neighbors before velocity limiting')
    h = out[:,:,0]
    q = out[:,:,1:].copy()
    hmean = h.mean(axis=1)
    qmean = q.mean(axis=1)
    center = np.zeros_like(qmean)
    np.divide(qmean,hmean[:,None],out=center,where=hmean[:,None]>0.)
    lower = center[_neighbors].min(axis=1)
    upper = center[_neighbors].max(axis=1)
    velocity = np.zeros_like(q)
    np.divide(q,h[:,:,None],out=velocity,where=h[:,:,None]>0.)
    delta = velocity-center[:,None,:]
    ratios = np.ones_like(q)
    np.divide((upper-center)[:,None,:],delta,out=ratios,where=delta>0.)
    negative = delta<0.
    numerator = np.broadcast_to((lower-center)[:,None,:],q.shape)
    ratios[negative] = numerator[negative]/delta[negative]
    # Zero-depth nodes carry no momentum and impose no velocity constraint.
    ratios[h==0.] = 1.
    theta = np.clip(ratios.min(axis=1),0.,1.)
    out[:,:,1:] = theta[:,None,:]*q+(1-theta[:,None,:])*h[:,:,None]*center[:,None,:]
    stats['velocity_limited_cells'] = int(np.count_nonzero(np.any(theta<1.,axis=1)))
    stats['velocity_momentum_max_mean_change'] = float(np.max(np.abs(out[:,:,1:].mean(axis=1)-qmean)))
    stats['velocity_momentum_l1_coefficients'] = float(np.abs(out[:,:,1:]-q).sum())
    return out, stats
