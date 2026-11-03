#!/usr/bin/env python3
"""Non-Gaussian ln V at fixed mass from Gaussian ICs through spherical collapse, down to low mass -> results/sc_lowmass_<cfg>.json, figures/figM4_*

Patches with non-positive signed volume (clipped at ln V = ln 1e-3) inflate the std; quantile widths are immune, as long as those patches stay in the same tail.
 w68 = (q84 - q16)/2, w90 = (q95 - q05)/3.29 (both = sigma for a Gaussian); skewness of ln V.
SC: lnV = g(D sigma_lin(R0) nu), nu Gaussian, frozen at V/V_L = 1/Delta for Delta = 50, 178, 1000; with Delta = infinity for delta_L -> delta_c it diverges, so this brackets the choice."""
import glob, json, os, re, sys
import h5py, numpy as np
cfg = sys.argv[1]; figdir = sys.argv[2] if len(sys.argv) > 2 else None
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing"); here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
th = np.linspace(1e-3, 2 * np.pi - 1e-3, 40000); et = np.linspace(1e-3, 40, 40000)
dL = np.concatenate([(-3 / 5 * (3 / 4 * (np.sinh(et) - et)) ** (2 / 3))[::-1], 3 / 5 * (3 / 4 * (th - np.sin(th))) ** (2 / 3)])
lm = np.log(np.concatenate([(9 / 2 * (np.sinh(et) - et) ** 2 / (np.cosh(et) - 1) ** 3)[::-1], 9 / 2 * (th - np.sin(th)) ** 2 / (1 - np.cos(th)) ** 3]))
o = np.argsort(dL); dL, lm = dL[o], np.maximum.accumulate(lm[o])
g = lambda d, Dv: -np.minimum(np.interp(d, dL, lm), np.log(Dv))
def stats(x, w=None):
    if w is None:
        q = np.percentile(x, [5, 16, 50, 84, 95]); m, s = x.mean(), x.std(); sk = ((x - m) ** 3).mean() / s ** 3
    else:
        i = np.argsort(x); c = np.cumsum(w[i]) / w.sum(); q = np.interp([.05, .16, .5, .84, .95], c, x[i]); W = w / w.sum(); m = (W * x).sum(); s = np.sqrt((W * (x - m) ** 2).sum()); sk = (W * (x - m) ** 3).sum() / s ** 3
    return dict(mean=float(m), std=float(s), w68=float((q[3] - q[1]) / 2), w90=float((q[4] - q[0]) / 3.29), median=float(q[2]), skew=float(sk))
nu = np.linspace(-8, 8, 200001); pn = np.exp(-nu ** 2 / 2)
R = {}
for f in sorted(glob.glob(os.path.join(D0, "patchaniso_low", f"{cfg}_L*_s*_D*.h5"))):
    m = re.search(r"_L(\d+)_s([+-]\d)_D([0-9.]+)\.h5$", f); s, sg, D = int(m[1]), int(m[2]), float(m[3])
    with h5py.File(f) as h:
        for k in h:
            G = h[k]; R0 = float(G.attrs["R0"]); lv = G["lnV"][()].astype(np.float64); ok = G["ok"][()]
            e = stats(lv); e["neg"] = float(1 - ok.mean()); e["sig"] = float(G.attrs["sig_dL"]); e["hist"] = np.histogram(lv, bins=np.linspace(-7, 3, 101), density=True)[0]
            R.setdefault((D, R0), []).append((s, e))
res = dict(cfg=cfg, rows=[])
print("D  R0 M | D*sig_lin | measured std w68 w90 skew neg | SC D=178 std w68 w90 skew | SC D=50 std w68 | SC D=1000 std w68 | Gaussian (D sig): std")
for (D, R0) in sorted(R):
    E = [x[1] for x in R[(D, R0)]]; A = lambda k: np.array([e[k] for e in E])
    sig = A("sig").mean(); lin = D * sig; M = 4 * np.pi / 3 * R0 ** 3 * 2.775e11
    sc = {Dv: stats(g(lin * nu, Dv), pn) for Dv in (50., 178., 1000.)}
    row = dict(D=D, R0=R0, M=M, lin=lin, meas={k: float(A(k).mean()) for k in ("mean", "std", "w68", "w90", "median", "skew", "neg")}, e_w68=float(A("w68").std(ddof=1) / np.sqrt(len(E))), sc=sc)
    res["rows"].append(row); a = row["meas"]
    print(f"{D:.1f} {R0:5.1f} {M:.1e} | {lin:.3f} | {a['std']:.3f} {a['w68']:.3f} {a['w90']:.3f} {a['skew']:+.2f} {a['neg']:.3f} | {sc[178.]['std']:.3f} {sc[178.]['w68']:.3f} {sc[178.]['w90']:.3f} {sc[178.]['skew']:+.2f} | {sc[50.]['std']:.3f} {sc[50.]['w68']:.3f} | {sc[1000.]['std']:.3f} {sc[1000.]['w68']:.3f}")
json.dump(res, open(os.path.join(here, f"sc_lowmass_{cfg}.json"), "w"), indent=0, default=float)
if figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [r for r in (6.0, 9.4, 14.8, 20.7)]; xb = 0.5 * (np.linspace(-7, 3, 101)[1:] + np.linspace(-7, 3, 101)[:-1])
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.8), sharey=True, gridspec_kw=dict(wspace=0.05), facecolor="white")
    for ax, r0 in zip(axs, show):
        D = 1.0; E = [x[1] for x in R[(D, r0)]]; h = np.mean([e["hist"] for e in E], 0); sig = np.mean([e["sig"] for e in E]); lin = D * sig
        ax.step(xb, h, where="mid", color="#eda100", lw=2, label="PM Lagrangian patches")
        lv = g(lin * nu, 178.); pdf = np.histogram(lv, bins=np.linspace(-7, 3, 101), weights=pn, density=True)[0]
        ax.step(xb, pdf, where="mid", color="#2a78d6", lw=1.6, ls=(0, (4, 2)), label=r"spherical collapse, frozen at 178")
        x = np.linspace(-7, 3, 500); ax.plot(x, np.exp(-x ** 2 / (2 * lin ** 2)) / np.sqrt(2 * np.pi) / lin, color="#52514e", lw=1.2, ls=":", label=r"Gaussian, $\sigma = D\sigma_{\rm lin}$")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 3); ax.set_xlim(-6.5, 2.5); ax.set_xlabel(r"$\ln(V/V_L)$"); ax.set_title(f"M = {4 * np.pi / 3 * r0 ** 3 * 2.775e11:.1e} $h^{{-1}}M_\\odot$, D = 1", fontsize=10, loc="left")
        ax.grid(True, color="#e4e3df", lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("probability density"); hh, ll = axs[0].get_legend_handles_labels(); fig.legend(hh, ll, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.08))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(figdir, f"figM4_lnV_pdf_{cfg}.{ext}"), dpi=170, bbox_inches="tight")
    print("wrote figure")
