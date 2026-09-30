"""Density PDFs of the study-1 comparison with two estimators on the SAME snapshots.

    python run_sheet_pdf.py deposit [TAG ...]   # CIC + exact sheet density on the 256^3 analysis mesh
    python run_sheet_pdf.py stats              # quantiles / moments / CDFs of every (snapshot, estimator)

Snapshots (numpy (3,N,N,N) displacements at z = 0, DiscoDJNative, study-1 initial conditions) are
read from _scratch/copula; see make_snapshots.sh.  Tags are <model>_N<N>[_paired] with model in
1lpt, 2lpt, 3lpt, 4lpt, nbody.

Estimators (both on C.N_ANA^3 = 256^3 cells of 1.17 Mpc/h, rho-bar = 1):
  cic    study-1 estimator: particles CIC-deposited (core.cic_density), CIC window deconvolved
  sheet  phase-space sheet: 6 tetrahedra per Lagrangian cube, each tetrahedron's mass spread
         uniformly over its volume and deposited by its EXACT overlap with each cell (R3D,
         sheet_deposit.jl); the pixel (cell-average) window is deconvolved
Smoothing, weighting and statistics are exactly those of study 1 (analyze.py): real-space top hat
R_s = 7, 14, 28, 42 Mpc/h; volume weighting = cells equally; mass weighting = each cell by its
unsmoothed deposited mass.  R_s = 0 is added as a diagnostic: the raw 1.17 Mpc/h cell values
(no smoothing, no window deconvolution), where the two estimators differ most.
"""
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C
import core
from cosmo import W_TH

SNAP = os.path.join(C.SCRATCH, "copula")
OUT = os.environ.get("SHEETPDF_SCRATCH", os.path.join(C.SCRATCH, "sheet_pdf"))
RES = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
JULIA = os.environ.get("JULIA", "/opt/jl/bin/julia")
JENV = dict(os.environ, JULIA_DEPOT_PATH=os.environ.get("JULIA_DEPOT_PATH", "/opt/jdepot"),
            JULIA_PKG_SERVER=os.environ.get("JULIA_PKG_SERVER", ""))

MODELS = ["1lpt", "2lpt", "3lpt", "4lpt", "nbody"]
TAGS = ([f"{m}_N{n}" for n in (64, 128, 256) for m in MODELS] +
        [f"{m}_N256_paired" for m in MODELS])
QS = np.array([1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999])       # as study 1
FINEBINS = np.linspace(np.log10(0.05), np.log10(20.0), 1201)  # as study 1
LOGBINS = core.LOGBINS
EST = ("cic", "sheet")
RS = [0.0] + list(C.R_SMOOTH)          # 0 = raw cells (diagnostic)


def psi_path(tag):
    parts = tag.split("_")
    m, n = parts[0], parts[1]
    suf = "_paired" if tag.endswith("_paired") else ""
    return os.path.join(SNAP, f"psi_{m}_{n}_z0{suf}.npy")


def rho_path(tag, est):
    return os.path.join(OUT, f"{tag}_{est}.npy")


def deposit(tag):
    psi = np.load(psi_path(tag))
    n = psi.shape[1]
    t0 = time.time()
    if not os.path.exists(rho_path(tag, "cic")):
        q = core.lattice(n, C.L)
        x = np.mod(q + psi, C.L)
        np.save(rho_path(tag, "cic"), core.cic_density(x, C.N_ANA, C.L).astype(np.float32))
        del x
    del psi
    if not os.path.exists(rho_path(tag, "sheet")):
        cmd = [JULIA, "-t", str(os.cpu_count()), f"--project={HERE}", os.path.join(HERE, "sheet_deposit.jl"),
               psi_path(tag), str(C.N_ANA), str(C.L), rho_path(tag, "sheet") + ".tmp.npy"]
        r = subprocess.run(cmd, env=JENV, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        os.replace(rho_path(tag, "sheet") + ".tmp.npy", rho_path(tag, "sheet"))
        print("  " + r.stdout.strip().splitlines()[-1], flush=True)
    print(f"deposit {tag}: {time.time() - t0:.0f} s", flush=True)


def pixel_window(ng, L):
    """Fourier transform of the cell average (unsquared sinc); core.cic_window is its square."""
    return np.sqrt(core.cic_window(ng, L))


def smoothed(rho, est):
    ng = rho.shape[0]
    rk = core.rfftn(rho.astype(np.float32))
    win = core.cic_window(ng, C.L) if est == "cic" else pixel_window(ng, C.L)
    kx, ky, kz = core.kgrid(ng, C.L, np.float64)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    out = {0.0: rho.astype(np.float32)}
    out.update({R: core.irfftn(rk * (W_TH(k * R) / win), ng).astype(np.float32) for R in C.R_SMOOTH})
    return out


def wquantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, v[o])


def stats_one(tag, est):
    rho = np.load(rho_path(tag, est))
    w = rho.ravel().astype(np.float64)
    out = dict(mean=float(w.mean()), min_raw=float(w.min()), neg_raw=float(np.mean(w < 0)))
    for R, f in smoothed(rho, est).items():
        v = f.ravel().astype(np.float64)
        lv = np.log10(np.clip(v, 1e-6, None))
        d = v - 1
        var = np.mean(d**2)
        hv, _ = np.histogram(lv, FINEBINS)
        hm, _ = np.histogram(lv, FINEBINS, weights=w)
        pv, _ = np.histogram(lv, LOGBINS, density=True)
        pm, _ = np.histogram(lv, LOGBINS, weights=w, density=True)
        out[f"R{R:g}"] = dict(qV=np.quantile(v, QS).tolist(), qM=wquantile(v, w, QS).tolist(),
                              var=float(var), S3=float(np.mean(d**3) / var**2),
                              S4=float((np.mean(d**4) - 3 * var**2) / var**3),
                              min=float(v.min()), neg=float(np.mean(v <= 0)),
                              cdfV=(np.cumsum(hv) / hv.sum()).tolist(), cdfM=(np.cumsum(hm) / hm.sum()).tolist(),
                              pdfV=pv.tolist(), pdfM=pm.tolist())
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "deposit":
        for tag in (sys.argv[2:] or TAGS):
            if not os.path.exists(psi_path(tag)):
                print(f"skip {tag}: no snapshot", flush=True); continue
            deposit(tag)
    elif cmd == "stats":
        res = dict(QS=QS.tolist(), R=RS, FINEBINS=FINEBINS.tolist(), LOGBINS=LOGBINS.tolist(),
                   N_ANA=C.N_ANA, L=C.L, runs={})
        for tag in TAGS:
            for est in EST:
                if os.path.exists(rho_path(tag, est)):
                    res["runs"][f"{tag}|{est}"] = stats_one(tag, est)
            print(f"stats {tag}", flush=True)
        with open(os.path.join(RES, "stats.json"), "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
