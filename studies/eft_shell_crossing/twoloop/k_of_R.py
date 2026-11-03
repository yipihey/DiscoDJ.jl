#!/usr/bin/env python3
"""Map scales through equal variance: Delta^2(k, D) = sigma^2(R, D), with Delta^2 = k^3 P / (2 pi^2) -> results/k_of_R_<cfg>.json, figures/figM5_*

In linear theory both sides scale as D^2, so k(R) is one curve; any spread between epochs is nonlinear.  M = rhobar V(R), Omega_m = 1, in h^-1 Msun.
Nonlinear P(k): pk_snap.jl, same deposit as sigma(R); sigma(R): random spheres from sigmaM.jl.  Errors: spread of k(R) over seeds (standard error)."""
import glob, json, os, re, sys
import h5py
import numpy as np
cfg = sys.argv[1]; figdir = sys.argv[2] if len(sys.argv) > 2 else None
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
r = json.load(open(os.path.join(here, f"sigmaM_{cfg}.json"))); RE = np.array(r["RE"])
with h5py.File(os.path.join(D0, "tl_full", "tl", "ops_sharp_L1.h5")) as h:
    pk, pP = h["pk_k"][()], h["pk_P"][()]
KUV = {"tl_full": 1.6}[cfg]
Plin = lambda k: np.where(k < KUV, np.exp(np.interp(np.log(k), np.log(pk), np.log(pP))), 1e-30)
n = np.arange(-256, 257); n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel(); KF = 2 * np.pi / 500.
n2 = n2[(n2 > 0) & (KF * np.sqrt(n2) < KUV)]; u, c = np.unique(n2, return_counts=True); q = KF * np.sqrt(u); qP = Plin(q)
tophat = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
s_lin = np.array([np.sqrt((c * qP * tophat(q * R) ** 2).sum() / 500. ** 3) for R in RE])
MASS = lambda R: 4 * np.pi / 3 * R ** 3 * 2.775e11

data = {}
for f in sorted(glob.glob(os.path.join(D0, "pk_snap", f"{cfg}_L*_s*_D*.h5"))):
    m = re.search(r"_L(\d+)_s([+-]\d)_D([0-9.]+)\.h5$", f)
    with h5py.File(f) as h:
        data.setdefault(float(m[3]), {}).setdefault(int(m[1]), []).append((h["k"][()], h["P"][()], h["n"][()]))
Ds = sorted(data)


def invert(k, Dl2, sig2):
    """k at which the rising Delta^2(k) equals sig2; log-log interpolation."""
    ok = (k > 0) & (Dl2 > 0) & (k < 1.5)
    x = np.log(k[ok]); y = np.maximum.accumulate(np.log(Dl2[ok]))
    return np.exp(np.interp(np.log(sig2), y, x, left=np.nan, right=np.nan))


k_lin = np.logspace(np.log10(0.012), np.log10(1.59), 800); D2_lin = k_lin ** 3 * Plin(k_lin) / (2 * np.pi ** 2)
kR_lin = invert(k_lin, D2_lin, s_lin ** 2)
res = dict(cfg=cfg, RE=RE.tolist(), mass=MASS(RE).tolist(), k_linear=kR_lin.tolist(), epochs={})
print(f"{cfg}: epochs {Ds}.  Linear k(R) is independent of D.")
print("R  M | k_lin kR | " + " | ".join(f"D={D:g}: k kR" for D in Ds))
rows = {}
for D in Ds:
    e = r["epochs"][f"{D:.3f}"]; sR = np.array(e["s_R"])
    per = []
    for s, runs in data[D].items():
        k = runs[0][0]; P = np.mean([x[1] for x in runs], axis=0)
        per.append(invert(k, k ** 3 * P / (2 * np.pi ** 2), sR ** 2))
    per = np.array(per); kk = np.nanmean(per, 0); ek = np.nanstd(per, 0, ddof=1) / np.sqrt(len(per))
    rows[D] = (kk, ek)
    k0 = runs[0][0]; Pm = np.mean([np.mean([x[1] for x in rr], 0) for rr in data[D].values()], 0)
    res["epochs"][f"{D:.3f}"] = dict(k=kk.tolist(), err=ek.tolist(), sigma=sR.tolist(), P_k=k0.tolist(), P=Pm.tolist())
for R0 in (5, 8, 12, 20, 30, 50, 80, 110):
    i = int(np.argmin(abs(RE - R0))); t = f"{RE[i]:5.1f} {MASS(RE[i]):.1e} | {kR_lin[i]:.3f} {kR_lin[i] * RE[i]:.2f} | "
    print(t + " | ".join(f"{rows[D][0][i]:.3f} {rows[D][0][i] * RE[i]:.2f}" for D in Ds))
print("\nValidation: P/(D^2 P_lin) at D = lowest epoch")
D = Ds[0]; k0 = data[D][next(iter(data[D]))][0][0]; Pm = np.mean([np.mean([x[1] for x in rr], 0) for rr in data[D].values()], 0)
for kk in (0.02, 0.05, 0.1, 0.3, 0.6, 1.0, 1.4):
    i = int(np.argmin(abs(k0 - kk))); print(f"  k={k0[i]:.3f} ratio {Pm[i] / (D ** 2 * Plin(k0[i])):.3f}")
json.dump(res, open(os.path.join(here, f"k_of_R_{cfg}.json"), "w"), indent=0, default=float)
if figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [D for D in (0.3, 0.5, 0.8, 1.0) if D in Ds]; col = dict(zip(show, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1), gridspec_kw=dict(wspace=0.26), facecolor="white")
    ax[0].plot(RE, kR_lin, color="#52514e", lw=1.6, ls="--", label="linear (all D)")
    ax[1].plot(kR_lin, MASS(RE), color="#52514e", lw=1.6, ls="--")
    ax[2].plot(MASS(RE), kR_lin * RE, color="#52514e", lw=1.6, ls="--")
    for D in show:
        kk = rows[D][0]
        ax[0].plot(RE, kk, color=col[D], lw=2, label=f"D = {D:g}")
        ax[1].plot(kk, MASS(RE), color=col[D], lw=2)
        ax[2].plot(MASS(RE), kk * RE, color=col[D], lw=2)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"$R$ [$h^{-1}$Mpc]"); ax[0].set_ylabel(r"$k$ [$h$ Mpc$^{-1}$]")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]"); ax[1].set_ylabel(r"$M$ [$h^{-1}M_\odot$]")
    ax[2].set_xscale("log"); ax[2].set_xlabel(r"$M$ [$h^{-1}M_\odot$]"); ax[2].set_ylabel(r"$kR$")
    for a in ax:
        a.grid(True, which="major", color="#e4e3df", lw=0.7); a.set_axisbelow(True); a.tick_params(labelsize=9)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.04))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(figdir, f"figM5_k_of_R_{cfg}.{ext}"), dpi=170, bbox_inches="tight")
    print("wrote figure")
