"""Steps 2-5 of the copula study: two-point statistics with the one-point PDF removed.

    python analysis.py run N LEVEL Z      # steps 2-4 (+ step-5 ingredients) for both LPT orders
    python analysis.py collect            # convergence bookkeeping, attribution, figures

EVERY statistic here is CONDITIONAL on the single-stream mask (both runs) and is not an
unconditional two-point statistic.

Definitions (fixed before any science result was looked at):
  Lagrangian mask M   elements with no inverted tetrahedron and exact centroid stream count 1
                      in BOTH runs (DiscoDJNative periodic sheet kernels).
  y                   rank-Gaussianization per run / z / R over M (gauss.gaussianize_both;
                      mass: equal element weights, volume: exact element volumes V_e),
                      tie-break jitter seed CFG["tie_seeds"][0].
  Step 2  xi_y(|dq|)  masked weighted pair average; zero lag excluded from all bins.
          P_y(k_q)    V |F(k)|^2 / (N_e * sum(w^2 M)) with F the DFT of w M y: sum_k P/V equals
                      the weighted zero-lag variance, so P_y CONTAINS the zero-lag term (never add
                      xi_y and P_y).  r(k) = Re<F_a F_b*> / sqrt(<|F_a|^2><|F_b|^2>) per shell.
  Step 3  u = Phi(y); Spearman rho_S = weighted Pearson(u_a, u_b) (mass: equal pair weights,
          volume: pair weight (V_a + V_b)/2); eta^2 = correlation ratio of u_NB on 1000 quantile
          bins of u_LPT (variance explained by ANY function), R2_iso = variance explained by the
          best MONOTONE function (isotonic regression of the bin means); non-monotone part
          eta^2 - R2_iso.
  Step 4  Eulerian: point-sampled AHK sheet density on the ng^3 mesh (ng = CFG["eulerian_mesh"],
          256: 1.17 Mpc/h; 512 does not fit in 15 GB), top-hat smoothed at R (R = 1.5 is
          mesh-limited).
          Eulerian mask E = nodes single-stream in both runs AND covered in both runs by
          elements of M.  Volume weighting: equal node weights; mass weighting: node weight =
          unsmoothed sheet density.  xi_y^E with the mask-window correction corr(wEy)/corr(wE).
          Asymmetric remap (LABELLED): LPT field with the N-body marginal, delta_remap =
          Q_NB(F_LPT(delta_LPT)); xi_delta for LPT, remap, N-body.
  Step 5  (see collect) marginal / Lagrangian-copula / displacement-mapping attribution with
          label transport Z_ab(x) = y_a(e_b(x)); both orderings are reported.
"""
import glob, json, os, sys, subprocess, time
import numpy as np
from scipy.special import ndtr

from common import CFG, OUT, SCRATCH, snap_path
from fields import Snapshot, cleanup, JULIA, JENV, HERE
import gauss
import core
from cosmo import W_TH

L = CFG["box_L"]
NG = CFG["eulerian_mesh"]          # Eulerian analysis mesh (step 4-5); see config _comment_eulerian_mesh
SEED = CFG["tie_seeds"][0]
EDGES = gauss.xi_bins(CFG)
STEPS = os.path.join(OUT, "steps")
os.makedirs(STEPS, exist_ok=True)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def eul_product(model, n, z, level):
    out = os.path.join(SCRATCH, f"eul_{model}_N{n}_z{z:g}_l{level}_ng{NG}.h5")
    if os.path.exists(out):
        return out
    cmd = [JULIA, "-t", str(os.cpu_count()), f"--project={HERE}", os.path.join(HERE, "eulerian_products.jl"),
           snap_path(model, n, z), str(level), str(NG), str(L), out + ".tmp"]
    r = subprocess.run(cmd, env=JENV, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"eulerian_products failed:\n{r.stdout}\n{r.stderr}")
    os.replace(out + ".tmp", out)
    return out


def kedges(n):
    return np.geomspace(2 * np.pi / L * 0.999, np.pi * n / L * 1.001, 31)


def _kmag(n):
    kx, ky, kz = core.kgrid(n, L, np.float64)
    return np.sqrt(kx**2 + ky**2 + kz**2), np.where(kz == 0, 1.0, 2.0) * np.ones_like(kx * ky * kz)


def spectra(fa, wa, fb, wb, n):
    """Masked weighted pseudo-spectra of fa, fb (already multiplied by w*M) and their r(k)."""
    Fa = core.rfftn(fa); Fb = core.rfftn(fb)
    k, mult = _kmag(n)
    V = L**3; Ne = n**3
    na = V / (Ne * np.sum(wa**2)); nb = V / (Ne * np.sum(wb**2))
    edges = kedges(n)
    idx = np.digitize(k.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & (k.ravel() > 0)
    m = mult.ravel()[ok]; i = idx[ok]; nb_ = len(edges) - 1
    cnt = np.bincount(i, m, nb_)
    Paa = np.bincount(i, m * (np.abs(Fa.ravel()[ok])**2), nb_)
    Pbb = np.bincount(i, m * (np.abs(Fb.ravel()[ok])**2), nb_)
    Pab = np.bincount(i, m * np.real(Fa.ravel()[ok] * np.conj(Fb.ravel()[ok])), nb_)
    kc = np.bincount(i, m * k.ravel()[ok], nb_)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(k=kc / cnt, Pa=Paa / cnt * na, Pb=Pbb / cnt * nb, r=Pab / np.sqrt(Paa * Pbb), modes=cnt)


def wpearson(a, b, w=None):
    if w is None:
        w = np.ones_like(a)
    w = w / w.sum()
    ma, mb = np.sum(w * a), np.sum(w * b)
    ca, cb = a - ma, b - mb
    return float(np.sum(w * ca * cb) / np.sqrt(np.sum(w * ca**2) * np.sum(w * cb**2)))


def _pava(y, w):
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators."""
    v, ww, cnt = [], [], []
    for yi, wi in zip(y, w):
        v.append(yi); ww.append(wi); cnt.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            tw = ww[-2] + ww[-1]
            v[-2] = (v[-2] * ww[-2] + v[-1] * ww[-1]) / tw; ww[-2] = tw; cnt[-2] += cnt[-1]
            v.pop(); ww.pop(); cnt.pop()
    return np.repeat(v, cnt)


def monotone_stats(ua, ub, w=None, nbins=1000):
    """eta^2 (any function) and R2_iso (monotone) of ub given ua, on quantile bins of ua."""
    if w is None:
        w = np.ones_like(ua)
    order = np.argsort(ua, kind="stable")
    cw = np.cumsum(w[order]); cw /= cw[-1]
    b = np.minimum((cw * nbins).astype(int), nbins - 1)
    bw = np.bincount(b, w[order], nbins)
    bm = np.bincount(b, (w * ub)[order], nbins) / np.maximum(bw, 1e-300)
    mu = np.sum(w * ub) / np.sum(w)
    var = np.sum(w * (ub - mu)**2) / np.sum(w)
    eta2 = float(np.sum(bw * (bm - mu)**2) / np.sum(bw) / var)
    iso = _pava(bm[bw > 0], bw[bw > 0])
    r2iso = float(np.sum(bw[bw > 0] * (iso - mu)**2) / np.sum(bw) / var)
    return eta2, r2iso


def smooth_mesh(rho, R):
    if R == 0:
        return rho
    kx, ky, kz = core.kgrid(rho.shape[0], L, np.float64)
    return core.irfftn(core.rfftn(rho) * W_TH(np.sqrt(kx**2 + ky**2 + kz**2) * R), rho.shape[0])


def wquant_map(src, wsrc, ref, wref):
    """Asymmetric remap: value of `ref`'s weighted quantile function at `src`'s weighted CDF."""
    o = np.argsort(src, kind="stable")
    c = np.cumsum(wsrc[o]); u = (c - 0.5 * wsrc[o]) / c[-1]
    r = np.argsort(ref, kind="stable")
    cr = np.cumsum(wref[r]); ur = (cr - 0.5 * wref[r]) / cr[-1]
    out = np.empty_like(src)
    out[o] = np.interp(u, ur, ref[r])
    return out


def standardize(z, w):
    mu = np.sum(w * z) / np.sum(w)
    sd = np.sqrt(np.sum(w * (z - mu)**2) / np.sum(w))
    return (z - mu) / sd


def xi_grid(vals, mask, w, n, den=None):
    g = np.zeros(mask.shape); g[mask] = vals
    ww = np.where(mask, w, 0.0)
    return gauss.xi_lagrangian(g, mask, ww, L, EDGES, den=den)


# ---------------------------------------------------------------------------------------------
# one (N, z, level): steps 2-4 and the step-5 ingredients for both LPT orders.
# Two phases to bound memory at 512^3 elements / 512^3 mesh: the Lagrangian phase stores the
# masks and scores (float32) in scratch and frees everything; the Eulerian phase then loads one
# mesh at a time and keeps only mask-restricted vectors.
# ---------------------------------------------------------------------------------------------
def _tmp(tag):
    return os.path.join(SCRATCH, f"ana_{tag}.npy")


def _lagrangian_phase(n, level, z, lpt, res_pair):
    import gc
    sn = Snapshot("nbody", n, z, level); sl = Snapshot(lpt, n, z, level)
    M = ~sn.ms & ~sl.ms
    np.save(_tmp(f"M_{lpt}"), M)
    res_pair["mask_frac_lagr"] = float(M.mean())
    Vn, Vl = sn.V[M], sl.V[M]
    for R in CFG["R_list"]:
        d = res_pair["R"].setdefault(f"{R:g}", {})
        vn, vl = sn.values(R)[M], sl.values(R)[M]
        yn = gauss.gaussianize_both(vn, Vn, SEED); yl = gauss.gaussianize_both(vl, Vl, SEED)
        del vn, vl
        for wi, w in enumerate(("mass", "vol")):
            a, b = yl[wi], yn[wi]                                         # LPT, N-body scores
            for tag, arr in (("l", a), ("n", b)):
                full = np.zeros(M.size, np.float32); full[M.ravel()] = arr
                np.save(_tmp(f"y_{lpt}_{tag}_{w}_R{R:g}"), full); del full
            one = M.astype(float)
            gwn = one if w == "mass" else np.where(M, sn.V, 0.0)
            gwl = one if w == "mass" else np.where(M, sl.V, 0.0)
            # ---- step 2: Lagrangian copula
            xl = xi_grid(a, M, gwl, sn.n); xn = xi_grid(b, M, gwn, sn.n)
            fa = np.zeros(M.shape); fa[M] = a; fa *= gwl
            fb = np.zeros(M.shape); fb[M] = b; fb *= gwn
            sp = spectra(fa, gwl, fb, gwn, sn.n)
            del fa, fb, one, gwn, gwl
            # ---- step 3: element-paired ranks
            ua, ub = ndtr(a), ndtr(b)
            pw = None if w == "mass" else 0.5 * (Vn + Vl)
            rs = wpearson(ua, ub, pw)
            eta2, r2iso = monotone_stats(ua, ub, pw)
            H, _, _ = np.histogram2d(a, b, bins=60, range=[[-4, 4], [-4, 4]], weights=pw)
            del ua, ub
            d[w] = dict(xi_lpt=xl, xi_nb=xn, spectra=sp, spearman=rs, eta2=eta2, r2_iso=r2iso,
                        nonmonotone=eta2 - r2iso, hist2d=H)
        del yn, yl
        gc.collect()
        print(f"  [L] N={n} l={level} z={z:g} {lpt} R={R:g}", flush=True)
    del sn, sl, M, Vn, Vl
    gc.collect()


def _load_eul(model, n, z, level, keys):
    import h5py
    with h5py.File(eul_product(model, n, z, level), "r") as f:
        return {k: f[k][...] for k in keys}


def _eulerian_phase(n, level, z, lpt, res_pair):
    import gc
    Mf = np.load(_tmp(f"M_{lpt}")).ravel()
    A = _load_eul("nbody", n, z, level, ("nstream", "element")); B = _load_eul(lpt, n, z, level, ("nstream", "element"))
    en, el = A["element"], B["element"]
    E = (A["nstream"] == 1) & (B["nstream"] == 1) & (en >= 0) & (el >= 0)
    del A, B
    E[E] &= Mf[en[E]] & Mf[el[E]]
    enE, elE = en[E], el[E]
    del en, el, Mf
    res_pair["mask_frac_eul"] = float(E.mean())
    rho_raw = {"n": _load_eul("nbody", n, z, level, ("rho",))["rho"][E],
               "l": _load_eul(lpt, n, z, level, ("rho",))["rho"][E]}
    # node weights do not depend on R: pair-weight denominators are computed once per weighting
    W = {}
    one = np.ones(int(E.sum()))
    for w in ("mass", "vol"):
        wn = one if w == "vol" else rho_raw["n"]
        wl = one if w == "vol" else rho_raw["l"]
        gnE = np.zeros(E.shape); gnE[E] = wn
        dn_ = gauss.xi_denominator(E, gnE, L, EDGES); del gnE
        if w == "vol":
            dl_ = dn_
        else:
            glE = np.zeros(E.shape); glE[E] = wl
            dl_ = gauss.xi_denominator(E, glE, L, EDGES); del glE
        W[w] = (wn, wl, dn_, dl_)
    for R in CFG["R_list"]:
        d = res_pair["R"][f"{R:g}"]
        vals = {}
        for tag, model in (("n", "nbody"), ("l", lpt)):
            rho = _load_eul(model, n, z, level, ("rho",))["rho"]
            vals[tag] = smooth_mesh(rho, R)[E]; del rho
        for w in ("mass", "vol"):
            wn, wl, DN, DL = W[w]
            gnE = np.zeros(E.shape); gnE[E] = wn
            glE = np.zeros(E.shape); glE[E] = wl
            # ---- step 4: Eulerian Gaussianized xi and the LABELLED asymmetric remaps
            ynE = gauss.gaussianize(vals["n"], None if w == "vol" else wn, SEED)[0]
            ylE = gauss.gaussianize(vals["l"], None if w == "vol" else wl, SEED)[0]
            xnE = xi_grid(ynE, E, gnE, NG, DN); xlE = xi_grid(ylE, E, glE, NG, DL)
            del ynE, ylE
            dn = vals["n"] / (np.sum(wn * vals["n"]) / np.sum(wn)) - 1
            dl = vals["l"] / (np.sum(wl * vals["l"]) / np.sum(wl)) - 1
            xd = {"lpt": xi_grid(dl, E, glE, NG, DL), "nbody": xi_grid(dn, E, gnE, NG, DN)}
            xd["remap_lpt_to_nb"] = xi_grid(wquant_map(dl, wl, dn, wn), E, glE, NG, DL)   # LPT ranks, N-body marginal
            xd["remap_nb_to_lpt"] = xi_grid(wquant_map(dn, wn, dl, wl), E, gnE, NG, DN)   # N-body ranks, LPT marginal
            del dn, dl
            # ---- step 5 ingredients: label transport Z_ab(x) = y_a(e_b(x)), standardized on E
            xZ = {}
            for a_tag in ("l", "n"):
                ya = np.load(_tmp(f"y_{lpt}_{a_tag}_{w}_R{R:g}")).astype(np.float64)
                for b_tag, eb, gE, wE, DE in (("l", elE, glE, wl, DL), ("n", enE, gnE, wn, DN)):
                    xZ[a_tag + b_tag] = xi_grid(standardize(ya[eb], wE), E, gE, NG, DE)
                del ya
            d[w].update(xiE_lpt=xlE, xiE_nb=xnE, xi_delta=xd, xi_Z=xZ)
            del gnE, glE
            gc.collect()
        del vals
        print(f"  [E] N={n} l={level} z={z:g} {lpt} R={R:g}", flush=True)
    del E, enE, elE, rho_raw, W, one
    gc.collect()


def _drop(model, n, z, level):
    for p in (os.path.join(SCRATCH, f"eul_{model}_N{n}_z{z:g}_l{level}_ng{NG}.h5"),
              os.path.join(SCRATCH, f"prod_full_{model}_N{n}_z{z:g}_l{level}.h5")):
        if os.path.exists(p):
            os.remove(p)


def run(n, level, z):
    t0 = time.time()
    res = dict(N=n, level=level, z=z, seed=SEED, R_list=CFG["R_list"], pairs={}, config=CFG,
               mask_note="all statistics are conditional on the single-stream mask (both runs)")
    for lpt in CFG["lpt_orders"]:
        pr = dict(R={})
        _lagrangian_phase(n, level, z, lpt, pr)
        _eulerian_phase(n, level, z, lpt, pr)
        res["pairs"][lpt] = pr
        for p in glob.glob(_tmp("*")):
            os.remove(p)
        _drop(lpt, n, z, level)                     # bound disk: this LPT order is done
    _drop("nbody", n, z, level)
    fn = os.path.join(STEPS, f"steps_N{n}_l{level}_z{z:g}.npz")
    np.savez_compressed(fn, result=np.array(json.dumps(res, default=_tojson)))
    print(f"wrote {fn} ({time.time() - t0:.0f}s)", flush=True)


def _tojson(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(type(x))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    elif sys.argv[1] == "collect":
        import steps_collect
        steps_collect.main()
