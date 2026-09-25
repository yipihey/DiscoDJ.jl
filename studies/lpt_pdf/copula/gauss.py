"""Rank-Gaussianization and masked, weighted two-point functions in Lagrangian space.

Gaussianization (per run, per z, per R, over the mask M only):
  mass weighting   : y = Phi^-1((rank - 1/2) / N)                  (each element counts once)
  volume weighting : y = Phi^-1((C_before + w/2) / W),  w = V_e     (C_before = summed weight of
                     all lower-ranked elements, W = total weight; reduces to the mass formula
                     for w = 1)
Ties are broken by a uniform random jitter drawn from `seed` (lexicographic sort on
(value, jitter)); the number of tied values is returned and logged.

Two-point function on the element lattice (periodic, pairs inside M only):
  xi_w(dq) = sum_pairs w w' y y' / sum_pairs w w'
computed with FFTs of f = M w y and g = M w.  The zero-lag term (dq = 0, which is the
weighted variance and is not a pair of distinct elements) is returned separately and
is excluded from every separation bin, so it is never double counted.
"""
import numpy as np
import numba as nb
from scipy.special import ndtri

from core import rfftn, irfftn


def gaussianize(values, weights=None, seed=0):
    """values: 1-D float64 over the masked elements.  Returns (y, n_tied)."""
    v = np.asarray(values, np.float64)
    n = v.size
    jit = np.random.default_rng(seed).random(n)
    order = np.lexsort((jit, v))
    vs = v[order]
    n_tied = int(np.count_nonzero(vs[1:] == vs[:-1]))
    y = np.empty(n)
    if weights is None:
        y[order] = ndtri((np.arange(n) + 0.5) / n)
    else:
        w = np.asarray(weights, np.float64)[order]
        c = np.cumsum(w)
        W = c[-1]
        y[order] = ndtri((c - 0.5 * w) / W)
    return y, n_tied


def gaussianize_both(values, vol, seed=0):
    """Mass- and volume-weighted y from one sort.  Returns (y_mass, y_vol, n_tied)."""
    v = np.asarray(values, np.float64)
    n = v.size
    jit = np.random.default_rng(seed).random(n)
    order = np.lexsort((jit, v))
    del jit
    vs = v[order]
    n_tied = int(np.count_nonzero(vs[1:] == vs[:-1]))
    del vs
    ym = np.empty(n)
    ym[order] = ndtri((np.arange(n) + 0.5) / n)
    w = np.asarray(vol, np.float64)[order]
    c = np.cumsum(w)
    yv = np.empty(n)
    yv[order] = ndtri((c - 0.5 * w) / c[-1])
    return ym, yv, n_tied


def xi_bins(cfg):
    b = cfg["xi_bins"]
    return np.geomspace(b["rmin"], b["rmax"], b["nbins"] + 1)


@nb.njit(cache=True)
def _radial_bin(cf, cg, L, lrmin, dlog, nbins, num, den, rw):
    n0, n1, n2 = cf.shape
    d = L / n0
    for i in range(n0):
        x = (i if i <= n0 // 2 else i - n0) * d
        for j in range(n1):
            y = (j if j <= n1 // 2 else j - n1) * d
            for k in range(n2):
                z = (k if k <= n2 // 2 else k - n2) * d
                r = np.sqrt(x * x + y * y + z * z)
                if r == 0.0:
                    continue
                b = int(np.floor((np.log(r) - lrmin) / dlog))
                if b < 0 or b >= nbins:
                    continue
                num[b] += cf[i, j, k]
                den[b] += cg[i, j, k]
                rw[b] += cg[i, j, k] * r


def xi_lagrangian(y_grid, mask, w_grid, L, edges):
    """Masked weighted xi(|dq|) on the element lattice (log-spaced `edges`).

    y_grid, w_grid: (n,n,n) arrays (values outside mask ignored); mask: bool (n,n,n).
    Returns dict(r = pair-weighted mean separation per bin, xi, pairs_w, zero_lag)."""
    n = y_grid.shape[0]
    g = np.where(mask, w_grid, 0.0)
    G = rfftn(g)
    g *= np.where(mask, y_grid, 0.0)          # g is now f = M w y
    F = rfftn(g)
    del g
    np.abs(F, out=F); F *= F
    cf = irfftn(F, n); del F
    np.abs(G, out=G); G *= G
    cg = irfftn(G, n); del G
    zero = cf[0, 0, 0] / cg[0, 0, 0]
    nb_ = len(edges) - 1
    num = np.zeros(nb_); den = np.zeros(nb_); rw = np.zeros(nb_)
    lr = np.log(edges)
    _radial_bin(cf, cg, L, lr[0], lr[1] - lr[0], nb_, num, den, rw)
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = num / den
        rc = rw / den
    return dict(r=rc, xi=xi, pairs_w=den, zero_lag=float(zero))
