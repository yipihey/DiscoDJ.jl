"""Step 1 validation of the copula pipeline (must pass before any science result).

  python validate.py a  N LEVEL      # z_initial: LPT vs N-body Gaussianized fields element by element
  python validate.py b  N LEVEL      # monotone-transform invariance of the Gaussianized xi (z = 0)
  python validate.py c  N LEVEL Z    # noise floor: tie-break seeds (and xi saved for refinement floor)
  python validate.py d               # Zel'dovich marginal vs Doroshkevich | det J > 0
  python validate.py collect         # refinement floor from saved xi, summary JSON

Every call appends its configuration, seeds and results to results/step1/<test>_*.json.
Pass criteria are read from config.json ("criteria") and were fixed before running.
"""
import glob, json, os, sys, time
import numpy as np

from common import CFG, OUT, a_of_z
from fields import Snapshot
import gauss

L = CFG["box_L"]
EDGES = gauss.xi_bins(CFG)
S1 = os.path.join(OUT, "step1")
os.makedirs(S1, exist_ok=True)
CRIT = CFG["criteria"]
SEED0 = CFG["tie_seeds"][0]
ZI = max(CFG["z_list"])


def dump(name, obj):
    obj = dict(obj, config=CFG, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(S1, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))


def masks(models, n, z, level):
    """Multistream flags of every model; returns dict model -> ms (bool)."""
    out, info = {}, {}
    for m in models:
        s = Snapshot(m, n, z, level)
        out[m] = s.ms
        info[m] = dict(nflip=s.nflip, ms_frac=float(s.ms.mean()))
    return out, info


def per_model_y(model, n, z, level, mask, R_list, seeds=(SEED0,), keep_values=False):
    """Gaussianized y (mass, vol) over `mask` for every R (and seed)."""
    s = Snapshot(model, n, z, level)
    Vm = s.V[mask]
    res = {}
    for R in R_list:
        v = s.values(R)[mask]
        for sd in seeds:
            ym, yv, nt = gauss.gaussianize_both(v, Vm, sd)
            res[(R, sd)] = dict(ym=ym.astype(np.float64), yv=yv.astype(np.float64), ties=nt)
        if keep_values:
            res[(R, "v")] = v
    return res, s.V, getattr(s, "mesh_mean", None)


def grid_xi(y, mask, w, n):
    g = np.zeros((n, n, n)); g[mask] = y
    return gauss.xi_lagrangian(g, mask, w, L, EDGES)


# ---------------------------------------------------------------------------
def test_a(n, level):
    models = ["nbody"] + CFG["lpt_orders"]
    ms, info = masks(models, n, ZI, level)
    out = dict(N=n, level=level, z=ZI, masks=info, pairs={})
    ok_all = True
    ynb, _, mm = per_model_y("nbody", n, ZI, level, ~ms["nbody"] & ~ms[models[1]] & ~ms[models[2]], CFG["R_list"])
    M = ~ms["nbody"] & ~ms[models[1]] & ~ms[models[2]]
    out["mask_frac"] = float(M.mean())
    for m in CFG["lpt_orders"]:
        ylp, _, _ = per_model_y(m, n, ZI, level, M, CFG["R_list"])
        pr = {}
        for R in CFG["R_list"]:
            d = {}
            for w in ("ym", "yv"):
                a, b = ylp[(R, SEED0)][w], ynb[(R, SEED0)][w]
                r = float(np.corrcoef(a, b)[0, 1])
                rms = float(np.sqrt(np.mean((a - b)**2)))
                ok = r >= CRIT["a_min_pearson"] and rms <= CRIT["a_max_rms_dy"]
                ok_all &= ok
                d[w] = dict(pearson=r, rms_dy=rms, max_abs_dy=float(np.max(np.abs(a - b))),
                            p99_abs_dy=float(np.quantile(np.abs(a - b), 0.99)), passed=bool(ok),
                            ties_lpt=ylp[(R, SEED0)]["ties"], ties_nb=ynb[(R, SEED0)]["ties"])
            pr[f"R{R:g}"] = d
        out["pairs"][f"{m}_vs_nbody"] = pr
    out["passed"] = bool(ok_all)
    dump(f"a_N{n}_l{level}", out)
    print(json.dumps({k: out[k] for k in ("N", "level", "mask_frac", "passed")}), flush=True)


# ---------------------------------------------------------------------------
TRANSFORMS = {
    "log": np.log,
    "cube": lambda v: v**3,
    "neg_inverse": lambda v: -1.0 / v,
    "asinh10": lambda v: np.arcsinh(10.0 * (v - 1.0)),
}


def test_b(n, level, z=0.0):
    models = ["nbody", CFG["lpt_orders"][-1]]
    ms, info = masks(models, n, z, level)
    M = ~ms[models[0]] & ~ms[models[1]]
    out = dict(N=n, level=level, z=z, mask_frac=float(M.mean()), masks=info, models={})
    ok_all = True
    for m in models:
        s = Snapshot(m, n, z, level)
        Vm = s.V[M]
        wv = np.where(M, s.V, 0.0)
        one = M.astype(float)
        res = {}
        for R in CFG["R_list"]:
            v = s.values(R)[M]
            ym, yv, nt = gauss.gaussianize_both(v, Vm, SEED0)
            base = {"mass": grid_xi(ym, M, one, s.n), "vol": grid_xi(yv, M, wv, s.n)}
            rr = {}
            for tn, f in TRANSFORMS.items():
                fv = f(v)
                ym2, yv2, nt2 = gauss.gaussianize_both(fv, Vm, SEED0)
                x2 = {"mass": grid_xi(ym2, M, one, s.n), "vol": grid_xi(yv2, M, wv, s.n)}
                dxi = {w: float(np.nanmax(np.abs(x2[w]["xi"] - base[w]["xi"]))) for w in x2}
                dz = {w: abs(x2[w]["zero_lag"] - base[w]["zero_lag"]) for w in x2}
                ok = max(max(dxi.values()), max(dz.values())) <= CRIT["b_max_abs_dxi"]
                ok_all &= ok
                rr[tn] = dict(max_abs_dxi=dxi, abs_dzero_lag=dz, ties_after=nt2,
                              y_identical=bool(np.array_equal(ym, ym2) and np.array_equal(yv, yv2)),
                              passed=bool(ok))
            res[f"R{R:g}"] = dict(ties=nt, transforms=rr)
        out["models"][m] = res
    out["passed"] = bool(ok_all)
    dump(f"b_N{n}_l{level}", out)
    print(json.dumps({k: out[k] for k in ("N", "level", "mask_frac", "passed")}), flush=True)


# ---------------------------------------------------------------------------
def test_c(n, level, z, seeds=None):
    """Tie-break seed floor, and the xi needed for the refinement floor (collect).
    Masks are pairwise (nbody & lpt order); N-body is Gaussianized once per pair mask."""
    seeds = seeds or CFG["tie_seeds"]
    models = ["nbody"] + CFG["lpt_orders"]
    ms, info = masks(models, n, z, level)
    PM_ = {m: ~ms["nbody"] & ~ms[m] for m in CFG["lpt_orders"]}
    del ms
    out = dict(N=n, level=level, z=z, masks=info, seeds=seeds,
               mask_frac={m: float(M.mean()) for m, M in PM_.items()}, runs={})
    for mod in models:
        s = Snapshot(mod, n, z, level)
        pairs = CFG["lpt_orders"] if mod == "nbody" else [mod]
        for R in CFG["R_list"]:
            vall = s.values(R)
            for p in pairs:
                M = PM_[p]
                v = vall[M]; Vm = s.V[M]
                one = M.astype(float); wv = np.where(M, s.V, 0.0)
                xs, ties = {}, []
                for sd in seeds:
                    ym, yv, nt = gauss.gaussianize_both(v, Vm, sd)
                    ties.append(nt)
                    xs[sd] = {"mass": grid_xi(ym, M, one, s.n), "vol": grid_xi(yv, M, wv, s.n)}
                    del ym, yv
                d = {}
                for w in ("mass", "vol"):
                    arr = np.array([xs[sd][w]["xi"] for sd in seeds])
                    d[w] = dict(seed_floor=float(np.nanmax(np.nanmax(arr, 0) - np.nanmin(arr, 0))) if len(seeds) > 1 else None,
                                r=xs[seeds[0]][w]["r"], xi=xs[seeds[0]][w]["xi"],
                                zero_lag=xs[seeds[0]][w]["zero_lag"])
                out["runs"][f"{mod}|mask:{p}|R{R:g}"] = dict(ties=ties, mean_sheet_mesh=getattr(s, "mesh_mean", None), **d)
            del vall
        del s
    dump(f"c_N{n}_l{level}_z{z:g}", out)
    print(f"c N={n} level={level} z={z:g} done", flush=True)


# ---------------------------------------------------------------------------
def doroshkevich(sigma, nsamp, seed, chunk=2_000_000):
    """log10 rho = -log10 det J and det J for ZA points with det J > 0 (Doroshkevich)."""
    rng = np.random.default_rng(seed)
    C = (np.full((3, 3), 1 / 15.) + np.eye(3) * 2 / 15.) * sigma**2
    Lc = np.linalg.cholesky(C)
    lr, dets = [], []
    left = nsamp
    while left > 0:
        m = min(chunk, left); left -= m
        d = rng.standard_normal((m, 3)) @ Lc.T
        o = rng.standard_normal((m, 3)) * sigma / np.sqrt(15)
        T = np.zeros((m, 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = d.T
        T[:, 0, 1] = T[:, 1, 0] = o[:, 0]; T[:, 0, 2] = T[:, 2, 0] = o[:, 1]; T[:, 1, 2] = T[:, 2, 1] = o[:, 2]
        lam = np.linalg.eigvalsh(T)
        det = np.prod(1 - lam, axis=1)
        k = det > 0
        dets.append(det[k]); lr.append(-np.log10(det[k]))
    return np.concatenate(lr), np.concatenate(dets), None


def wq(x, w, qs):
    o = np.argsort(x)
    c = np.cumsum(w[o]); c = (c - 0.5 * w[o]) / c[-1]
    return np.interp(qs, c, x[o])


def za_quantiles(n, level, z, seed, qs):
    s = Snapshot("1lpt", n, z, level, seed=seed if seed != CFG["phase_seed"] else None)
    lr = np.log10(s.Vq / s.V)
    out = {}
    for cond, sel in (("mask", ~s.ms), ("detpos", s.V > 0)):
        x = lr[sel]; V = s.V[sel]
        out[cond] = dict(mass=np.quantile(x, qs), vol=wq(x, V, qs), frac=float(sel.mean()))
    return out


def sigma_lin_element():
    """sigma of the a = 1 linear density of the fixed-amplitude field (continuum, per mode sum)."""
    from cosmo import Cosmology
    import core
    c = Cosmology()
    n = 256
    kx, ky, kz = core.kgrid(n, L, np.float64)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    from cosmo import W_TH
    P = c.Pk(k) * W_TH(k * CFG["R_F"])**2
    wgt = np.full(k.shape, 2.0); wgt[..., 0] = 1; wgt[..., -1] = 1
    kn = np.pi * n / L
    msk = (np.abs(kx) < kn) & (np.abs(ky) < kn) & (kz < kn)
    msk[0, 0, 0] = False
    return float(np.sqrt(np.sum(wgt * msk * P) / L**3)), c


def test_d():
    td = CFG["test_d"]
    qs = np.array(td["quantiles"])
    sig1, cosmo = sigma_lin_element()
    out = dict(sigma_lin_a1=sig1, quantiles=qs, z={})
    ok_all = True
    for z in (1.0, 0.0):
        D = float(cosmo.D1(a_of_z(z)))
        lr, det, _ = doroshkevich(D * sig1, td["mc_samples"], td["mc_seed"])
        th = dict(mass=np.quantile(lr, qs), vol=wq(lr, det, qs))
        band = [za_quantiles(td["N_band"], td["refine_band"], z, s, qs) for s in td["extra_seeds"]]
        study_b = za_quantiles(td["N_band"], td["refine_band"], z, CFG["phase_seed"], qs)
        study_t = za_quantiles(td["N_test"], td["refine_test"], z, CFG["phase_seed"], qs)
        res = dict(D1=D, sigma=D * sig1, theory=th)
        for cond in ("mask", "detpos"):
            rc = {}
            for w in ("mass", "vol"):
                B = np.array([b[cond][w] for b in band])
                sd = B.std(0, ddof=1)
                dev_t = (study_t[cond][w] - th[w]) / sd
                dev_b = (study_b[cond][w] - th[w]) / sd
                dev_mean = (B.mean(0) - th[w]) / (sd / np.sqrt(len(B)))
                ok = (np.all(np.abs(dev_t) <= CRIT["d_nsigma"]) and
                      np.all(np.abs(dev_mean) <= CRIT["d_nsigma"]))
                if cond == "mask":
                    ok_all &= ok
                rc[w] = dict(theory=th[w], study_Ntest=study_t[cond][w], study_Nband=study_b[cond][w],
                             band_mean=B.mean(0), band_sd=sd, nsig_study_Ntest=dev_t,
                             nsig_study_Nband=dev_b, nsig_band_mean=dev_mean, passed=bool(ok))
            rc["frac_Ntest"] = study_t[cond]["frac"]
            res[cond] = rc
        out["z"][f"{z:g}"] = res
        print(f"d z={z:g} done", flush=True)
    out["passed_primary_mask"] = bool(ok_all)
    dump("d", out)
    print(json.dumps({"passed": out["passed_primary_mask"]}), flush=True)


if __name__ == "__main__":
    t = sys.argv[1]
    if t == "a":
        test_a(int(sys.argv[2]), int(sys.argv[3]))
    elif t == "b":
        test_b(int(sys.argv[2]), int(sys.argv[3]))
    elif t == "c":
        sd = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
        test_c(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sd)
    elif t == "d":
        test_d()
