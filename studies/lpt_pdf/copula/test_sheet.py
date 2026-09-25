"""Unit tests of the analysis machinery (run before validation).

sheet.py is an independent numba implementation of the phase-space sheet, kept as a
reference oracle: `test_julia_crosscheck` requires DiscoDJNative's periodic sheet kernels
(the production path, via fields.Snapshot) to reproduce it on a real 4LPT snapshot.
"""
import numpy as np
from common import CFG
import sheet, gauss

L = 100.0


def test_volume_and_density():
    n, ng = 32, 64
    rng = np.random.default_rng(0)
    q = np.arange(n) * L / n
    # smooth single-stream displacement: sum of a few long waves
    psi = np.zeros((3, n, n, n))
    for i in range(3):
        for _ in range(3):
            kv = rng.integers(-2, 3, 3)
            ph = rng.random() * 2 * np.pi
            arg = 2 * np.pi * (kv[0] * q[:, None, None] + kv[1] * q[None, :, None] + kv[2] * q[None, None, :]) / L + ph
            psi[i] += 0.6 * np.cos(arg)
    V, flip = sheet.elements(psi, L)
    ok1 = abs(V.sum() / L**3 - 1) < 1e-12 and not flip.any()
    rho = sheet.sheet_density(psi, L, ng)
    ok2 = abs(rho.mean() - 1) < 2e-3
    # 1-D Zel'dovich plane wave: element density exact, sheet density at nodes exact
    psi1 = np.zeros((3, n, n, n)); A = 0.8 * L / (2 * np.pi * 2)
    qc = (np.arange(n) + 0.5) * L / n
    psi1[0] = (A * np.sin(2 * np.pi * 2 * q / L))[:, None, None]
    V1, f1 = sheet.elements(psi1, L)
    # exact element Eulerian length: x(q+dq) - x(q)
    xq = q + A * np.sin(2 * np.pi * 2 * q / L)
    lx = np.roll(xq, -1) - xq; lx[-1] += L
    V1_exact = lx[:, None, None] * (L / n)**2 * np.ones((n, n, n))
    ok3 = np.allclose(V1, V1_exact, rtol=1e-12)
    print(f"  sum V_e / L^3 - 1 = {V.sum()/L**3-1:.1e}; flipped {flip.sum()}; "
          f"<sheet rho> - 1 = {rho.mean()-1:.1e}; plane-wave element volumes exact: {ok3}")
    return ok1 and ok2 and ok3


def test_gauss_xi():
    rng = np.random.default_rng(1)
    v = rng.random(10000)
    y, nt = gauss.gaussianize(v, seed=3)
    y2, _ = gauss.gaussianize(np.exp(v), seed=3)
    yw, _ = gauss.gaussianize(v, np.ones_like(v), seed=3)
    ok = np.array_equal(y, y2) and np.allclose(y, yw) and nt == 0
    # xi of white noise on a grid ~ 0, zero lag = 1
    n = 32
    yg = rng.standard_normal((n, n, n))
    m = np.ones((n, n, n), bool)
    r = gauss.xi_lagrangian(yg, m, np.ones((n, n, n)), L, gauss.xi_bins(CFG))
    ok &= abs(r["zero_lag"] - np.mean(yg**2)) < 1e-10 and np.nanmax(np.abs(r["xi"])) < 0.05
    print(f"  gaussianize monotone/weights consistent: {ok}; white-noise xi max {np.nanmax(np.abs(r['xi'])):.3f}")
    return ok


def test_mask_plane_wave():
    """Shell-crossed 1-D plane wave x = q + A sin(kq), A k = 1.6.  Truth: an element is
    multistream if its Eulerian interval meets the set of points with 3 preimages
    (found on a fine q grid).  The footprint mask must drop every such element; the
    number of single-stream elements it also drops (conservativeness) is reported."""
    n, ng = 64, 128
    kw = 2 * np.pi * 2 / L
    A = 1.6 / kw
    q = np.arange(n) * L / n
    psi = np.zeros((3, n, n, n))
    psi[0] = (A * np.sin(kw * q))[:, None, None]
    V, flip = sheet.elements(psi, L)
    ms = sheet.multistream_elements(psi, L, flip, ng)[:, 0, 0]
    qf = np.linspace(0, L, 200001)[:-1]
    xf = qf + A * np.sin(kw * qf)
    X = np.linspace(0, L, 20001)[:-1]
    # preimage count of X: sign changes of x(q) - X (periodic, unwrapped x spans [0, L))
    # count preimages per X: number of fine q intervals whose x-range contains X
    cnt = np.zeros(X.size, int)
    x0 = np.concatenate([xf, [xf[0] + L]])
    lo = np.minimum(x0[:-1], x0[1:]); hi = np.maximum(x0[:-1], x0[1:])
    for sh in (-L, 0.0, L):
        cnt += np.searchsorted(np.sort(lo + sh), X, side="right") - np.searchsorted(np.sort(hi + sh), X, side="left")
    multi_X = cnt >= 3
    xq = q + A * np.sin(kw * q)
    truth = np.zeros(n, bool)
    for i in range(n):
        a, b = xq[i], xq[(i + 1) % n] + (L if i == n - 1 else 0.0)
        lo_, hi_ = min(a, b), max(a, b)
        sel = ((X >= lo_) & (X <= hi_)) | ((X + L >= lo_) & (X + L <= hi_))
        truth[i] = multi_X[sel].any() or flip[i, 0, 0]
    missed = int(np.count_nonzero(truth & ~ms))
    extra = int(np.count_nonzero(~truth & ms))
    print(f"  plane-wave mask: true multistream elements {truth.sum()}/{n}, missed {missed}, "
          f"conservatively dropped single-stream {extra}")
    return missed == 0


def test_xi_bruteforce():
    """FFT masked/weighted xi against a direct pair sum (min-image) on a 12^3 grid."""
    n = 12
    rng = np.random.default_rng(5)
    y = rng.standard_normal((n, n, n)); w = rng.random((n, n, n)) + 0.5
    m = rng.random((n, n, n)) > 0.3
    edges = np.geomspace(L / n * 0.9, L, 9)
    r = gauss.xi_lagrangian(y, m, w, L, edges)
    idx = np.argwhere(m); yy = y[m]; ww = w[m]
    d = idx[:, None, :] - idx[None, :, :]
    d = (d + n // 2) % n - n // 2
    # the FFT treats the lag n/2 as +n/2 (not both signs): same |d|, so identical radius
    rr = np.sqrt((d**2).sum(-1)) * L / n
    ww2 = ww[:, None] * ww[None, :]; f2 = ww2 * yy[:, None] * yy[None, :]
    b = np.digitize(rr, edges) - 1
    num = np.array([f2[(b == i) & (rr > 0)].sum() for i in range(len(edges) - 1)])
    den = np.array([ww2[(b == i) & (rr > 0)].sum() for i in range(len(edges) - 1)])
    with np.errstate(invalid="ignore"):
        xb = num / den
    z0 = (ww**2 * yy**2).sum() / (ww**2).sum()
    ok = np.allclose(np.nan_to_num(xb), np.nan_to_num(r["xi"]), rtol=1e-10, atol=1e-12) and abs(z0 - r["zero_lag"]) < 1e-12
    print(f"  xi FFT vs brute force: max diff {np.nanmax(np.abs(xb - r['xi'])):.1e}, zero-lag diff {abs(z0 - r['zero_lag']):.1e}")
    return ok


def test_julia_crosscheck():
    import os
    from fields import Snapshot, product_path
    import sheet as nsheet
    from fields import L as LB, NG
    s = Snapshot("4lpt", 64, 0.0, 0)
    psi = np.load(__import__("common").snap_path("4lpt", 64, 0.0))
    V, flip = nsheet.elements(psi, LB)
    rho = nsheet.sheet_density(psi, LB, NG)
    import core
    from cosmo import W_TH
    kx, ky, kz = core.kgrid(NG, LB, np.float64)
    f3 = core.irfftn(core.rfftn(rho) * W_TH(np.sqrt(kx**2 + ky**2 + kz**2) * 3.0), NG)
    v3 = nsheet.sample_centroids(psi, LB, f3)
    dV = np.max(np.abs(s.V / V - 1)); dR = np.max(np.abs(s.values(3.0) / v3 - 1))
    same_flip = s.nflip == int(flip.sum())
    odd = np.all(s.nstream % 2 == 1)
    print(f"  Julia vs numba (4LPT N64 z0): max|dV/V| {dV:.1e}, max|d rho_3/rho_3| {dR:.1e}, "
          f"flipped {s.nflip} vs {int(flip.sum())}, all stream counts odd: {odd}")
    return dV < 1e-10 and dR < 1e-6 and same_flip and odd


if __name__ == "__main__":
    a = test_volume_and_density()
    b = test_gauss_xi()
    c = test_mask_plane_wave()
    d = test_xi_bruteforce()
    e = test_julia_crosscheck()
    print("PASS" if a and b and c and d and e else "FAIL")
