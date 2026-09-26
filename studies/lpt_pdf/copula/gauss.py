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
def _radial_bin(c, L, lrmin, dlog, nbins, out, rw):
    """out[b] += sum of c over lattice lags in bin b; rw[b] += c * r (if rw is non-empty)."""
    n0, n1, n2 = c.shape
    d = L / n0
    dor = rw.size > 0
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
                out[b] += c[i, j, k]
                if dor:
                    rw[b] += c[i, j, k] * r


def xi_denominator(mask, w_grid, L, edges):
    """Pair-weight counts of (mask, w) per bin; reusable by xi_lagrangian for any y on the same
    mask and weights.  Returns dict(den, rw, zero)."""
    n = mask.shape[0]
    G = rfftn(np.where(mask, w_grid, 0.0))
    np.abs(G, out=G); G *= G
    cg = irfftn(G, n); del G
    nb_ = len(edges) - 1
    den = np.zeros(nb_); rw = np.zeros(nb_)
    lr = np.log(edges)
    _radial_bin(cg, L, lr[0], lr[1] - lr[0], nb_, den, rw)
    return dict(den=den, rw=rw, zero=float(cg[0, 0, 0]))


def xi_lagrangian(y_grid, mask, w_grid, L, edges, den=None):
    """Masked weighted xi(|dq|) on the element lattice (log-spaced `edges`).

    y_grid, w_grid: (n,n,n) arrays (values outside mask ignored); mask: bool (n,n,n).
    den: optional xi_denominator(mask, w_grid, L, edges) to reuse across calls.
    Returns dict(r = pair-weighted mean separation per bin, xi, pairs_w, zero_lag)."""
    n = y_grid.shape[0]
    if den is None:
        den = xi_denominator(mask, w_grid, L, edges)
    g = np.where(mask, w_grid * y_grid, 0.0)  # f = M w y
    F = rfftn(g)
    del g
    np.abs(F, out=F); F *= F
    cf = irfftn(F, n); del F
    nb_ = len(edges) - 1
    num = np.zeros(nb_)
    lr = np.log(edges)
    _radial_bin(cf, L, lr[0], lr[1] - lr[0], nb_, num, np.zeros(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = num / den["den"]
        rc = den["rw"] / den["den"]
    return dict(r=rc, xi=xi, pairs_w=den["den"], zero_lag=float(cf[0, 0, 0] / den["zero"]))
