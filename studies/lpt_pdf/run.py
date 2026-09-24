"""Driver for the 2LPT vs N-body density-PDF study.

  python run.py lpt   N [--paired]
  python run.py nbody N [--mesh M] [--steps S] [--ai A] [--spacing log|a] [--paired]
  python run.py nlpt  N [--paired] [--no-dealias]   # 1LPT..4LPT (+ EdS-growth variants)

Each run writes results/<tag>.npz (PDFs, moments, shell-crossing fractions)
and _scratch/<tag>_rho.npy (CIC density on the common analysis mesh).
"""
import argparse, json, os, time
import numpy as np

import config as C
import core
from cosmo import Cosmology


def setup(n, paired):
    cosmo = Cosmology()
    ph = core.master_phases(C.N_MASTER, C.SEED, C.PHASES)
    dk = core.delta_k(cosmo, core.subgrid_modes(ph, n), n, C.L, C.R_F, paired)
    psi1, psi2 = core.lpt2(dk, n, C.L)
    return cosmo, psi1, psi2


def analyse_and_save(tag, x, meta):
    rho = core.cic_density(x, C.N_ANA, C.L)
    np.save(os.path.join(C.SCRATCH, f"{tag}_rho.npy"), rho)
    sm = core.smoothed_fields(rho, C.L, C.R_SMOOTH)
    res = core.pdfs(rho, sm)
    out = dict(meta=json.dumps(meta), bins=core.LOGBINS, R=np.array(C.R_SMOOTH))
    for R, r in res.items():
        for k, v in r.items():
            out[f"R{R:g}_{k}"] = np.asarray(v)
    np.savez(os.path.join(C.RESULTS, f"{tag}.npz"), **out)
    print(tag, json.dumps({f"R{R:g}": {k: r[k] for k in ("var", "S3", "S4")}
                           for R, r in res.items()}), flush=True)


def run_lpt(a):
    n = a.N
    tag = f"lpt_N{n}" + ("_paired" if a.paired else "")
    t0 = time.time()
    cosmo, psi1, psi2 = setup(n, a.paired)
    # shell crossing along the 2LPT trajectory (ever) and at a = 1
    flag = np.zeros((n, n, n, 6), np.uint8)
    a_hist = np.linspace(0.1, C.A_FINAL, 37)
    hist = []
    for aa in a_hist:
        x, _ = core.lpt_state(cosmo, psi1, psi2, aa, C.L, dtype=np.float32,
                              momenta=False)
        hist.append(core.update_crossing_flags(x, C.L, flag))
    now = np.zeros_like(flag)
    final = core.update_crossing_flags(x, C.L, now)
    del flag, now, psi1, psi2
    meta = dict(kind="2lpt", N=n, paired=a.paired, L=C.L, R_F=C.R_F,
                cross_ever=float(hist[-1]), cross_final=float(final),
                cross_hist_a=a_hist.tolist(), cross_hist=[float(h) for h in hist],
                seconds=time.time() - t0)
    analyse_and_save(tag, x, meta)
    print(tag, "crossed ever %.4g final %.4g" % (hist[-1], final), flush=True)


def run_nlpt(a):
    import nlpt
    n = a.N
    suffix = ("_nodealias" if a.no_dealias else "") + ("_paired" if a.paired else "")
    t0 = time.time()
    cosmo = Cosmology()
    ph = core.master_phases(C.N_MASTER, C.SEED, C.PHASES)
    dk = core.delta_k(cosmo, core.subgrid_modes(ph, n), n, C.L, C.R_F, a.paired)
    shapes = nlpt.compute_shapes(dk, n, C.L, dealias=not a.no_dealias, n_order=4)
    del dk
    print(f"shapes N={n} {time.time() - t0:.0f}s", flush=True)
    a_hist = np.linspace(0.1, C.A_FINAL, 37)
    for name, model in nlpt.models(cosmo).items():
        tag = f"{name}_N{n}{suffix}"
        flag = np.zeros((n, n, n, 6), np.uint8)
        hist = []
        for aa in a_hist:
            x = nlpt.positions(model, shapes, aa, C.L)
            hist.append(core.update_crossing_flags(x, C.L, flag))
        now = np.zeros_like(flag)
        final = core.update_crossing_flags(x, C.L, now)
        del flag, now
        meta = dict(kind="nlpt", model=name, N=n, paired=a.paired,
                    dealias=not a.no_dealias, L=C.L, R_F=C.R_F,
                    cross_ever=float(hist[-1]), cross_final=float(final),
                    cross_hist_a=a_hist.tolist(), cross_hist=[float(h) for h in hist],
                    seconds=time.time() - t0)
        analyse_and_save(tag, x, meta)
        print(tag, "crossed ever %.4g final %.4g" % (hist[-1], final), flush=True)


def run_nbody(a):
    n = a.N
    tag = (f"nb_N{n}_m{a.mesh}_s{a.steps}_ai{a.ai:g}_{a.spacing}"
           + ("_paired" if a.paired else ""))
    t0 = time.time()
    cosmo, psi1, psi2 = setup(n, a.paired)
    x, p = core.lpt_state(cosmo, psi1, psi2, a.ai, C.L)
    del psi1, psi2
    pm = core.PM(cosmo, C.L, a.mesh * n)
    hist_a, hist = [], []

    def cb(s, aa, x, p, flag):
        hist_a.append(float(aa)); hist.append(float(flag.mean()))
        if s % 10 == 0:
            print(f"  step {s} a={aa:.4f} crossed={hist[-1]:.4g} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    x, p, flag = pm.run(x, p, a.ai, C.A_FINAL, a.steps, track_crossing=True,
                        callback=cb, spacing=a.spacing)
    now = np.zeros_like(flag)
    final = core.update_crossing_flags(x, C.L, now)
    del flag, now, p
    meta = dict(kind="nbody", N=n, mesh=a.mesh, steps=a.steps, a_i=a.ai,
                spacing=a.spacing, paired=a.paired, L=C.L, R_F=C.R_F,
                cross_ever=hist[-1], cross_final=float(final),
                cross_hist_a=hist_a, cross_hist=hist, seconds=time.time() - t0)
    analyse_and_save(tag, x, meta)
    print(tag, "crossed ever %.4g final %.4g  (%.0fs)" % (hist[-1], final,
                                                        time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["lpt", "nbody", "nlpt"])
    ap.add_argument("N", type=int)
    ap.add_argument("--mesh", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--ai", type=float, default=0.04)
    ap.add_argument("--spacing", default="log", choices=["log", "a"])
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--no-dealias", action="store_true")
    a = ap.parse_args()
    {"lpt": run_lpt, "nbody": run_nbody, "nlpt": run_nlpt}[a.kind](a)
