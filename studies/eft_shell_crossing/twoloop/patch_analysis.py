#!/usr/bin/env python3
"""Does ln V of a Lagrangian patch depend only on delta_L, as spherical collapse says?  -> results/patch_anisotropy_<cfg>.json, figures/figM3_*
Var(lnV) = Var(E[lnV | nu]) + E[Var(lnV | nu)]: the second term is what any function of delta_L alone misses.  Then: how much of that residual tracks the initial shear q = (l1 - l3)?
Errors: delete-one jackknife over seeds, averaging the +- pair."""
import glob, json, os, re, sys
import h5py, numpy as np
cfg = sys.argv[1]; figdir = sys.argv[2] if len(sys.argv) > 2 else None
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing"); here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
th = np.linspace(1e-3, 2 * np.pi - 1e-3, 40000); et = np.linspace(1e-3, 40, 40000)
dL = np.concatenate([(-3 / 5 * (3 / 4 * (np.sinh(et) - et)) ** (2 / 3))[::-1], 3 / 5 * (3 / 4 * (th - np.sin(th))) ** (2 / 3)])
lm = np.log(np.concatenate([(9 / 2 * (np.sinh(et) - et) ** 2 / (np.cosh(et) - 1) ** 3)[::-1], 9 / 2 * (th - np.sin(th)) ** 2 / (1 - np.cos(th)) ** 3]))
o = np.argsort(dL); dL, lm = dL[o], np.maximum.accumulate(lm[o]); g_lnV = lambda d: -np.minimum(np.interp(d, dL, lm), np.log(178.0))
EDGES = np.linspace(-3.5, 3.5, 15); MID = 0.5 * (EDGES[1:] + EDGES[:-1])

def one(f):
    out = {}
    with h5py.File(f) as h:
        D = float(h.attrs["D"]); out["J"] = {k[2:]: float(h.attrs[k]) for k in h.attrs if k.startswith("J_")}
        for R in h:
            g = h[R]; R0 = float(g.attrs["R0"]); s = float(g.attrs["sig_dL"]); nu = g["nu"][()].astype(np.float64); lv = g["lnV"][()].astype(np.float64); ok = g["ok"][()]
            l1, l2, l3 = g["l1"][()].astype(np.float64), g["l2"][()].astype(np.float64), g["l3"][()].astype(np.float64); q = (l1 - l3) / s
            b = np.digitize(nu, EDGES) - 1; n = np.bincount(b[(b >= 0) & (b < len(MID))], minlength=len(MID))
            mean = np.full(len(MID), np.nan); std = mean.copy(); a31 = mean.copy(); c = mean.copy()
            res = np.full_like(lv, np.nan)
            for i in range(len(MID)):
                m = b == i
                if m.sum() < 30: continue
                mean[i] = lv[m].mean(); std[i] = lv[m].std(); a31[i] = g["a31"][()][m].mean(); res[m] = lv[m] - mean[i]
                if lv[m].std() > 0 and q[m].std() > 0: c[i] = np.corrcoef(lv[m], q[m])[0, 1]
            sel = np.isfinite(res); tot = lv[sel].var(); within = (res[sel] ** 2).mean()
            # fraction of the within-nu variance removed by the shear, per nu bin: linear in q and p = (l1 - 2 l2 + l3)
            qr = q.copy(); pr = ((l1 - 2 * l2 + l3) / s).copy()
            for i in range(len(MID)):
                m = b == i
                if m.sum() >= 30: qr[m] -= q[m].mean(); pr[m] -= pr[m].mean()
            X = np.c_[qr[sel], pr[sel], qr[sel] ** 2]; beta = np.linalg.lstsq(X, res[sel], rcond=None)[0]; expl = 1 - ((res[sel] - X @ beta) ** 2).mean() / within
            sc = g_lnV(D * s * nu); sc_rms = np.sqrt(((lv - sc)[sel] ** 2).mean())
            out[R0] = dict(mean=mean, std=std, a31=a31, corr=c, tot=tot, within=within, frac_within=within / tot, expl=expl, sc_rms=sc_rms, sc_mean=g_lnV(D * s * MID), s=s, n=n,
                           a31_all=float(g["a31"][()].mean()), okfrac=float(ok.mean()), std_lnV=float(lv.std()))
    return D, out

R = {}
for f in sorted(glob.glob(os.path.join(D0, os.environ.get("PATCH_DIR", "patchaniso"), f"{cfg}_L*_s*_D*.h5"))):
    m = re.search(r"_L(\d+)_s([+-]\d)_D", f); D, o_ = one(f); R[(int(m[1]), int(m[2]), D)] = o_
seeds = sorted({k[0] for k in R}); Ds = sorted({k[2] for k in R}); R0s = sorted(k for k in next(iter(R.values())) if k != "J")
def agg(D, R0, key):
    a = np.array([np.nanmean([R[(s, g, D)][R0][key] for g in (1, -1)], axis=0) for s in seeds]); m = np.nanmean(a, 0)
    if len(seeds) < 2: return m, m * np.nan
    J = np.array([np.nanmean(np.delete(a, i, 0), 0) for i in range(len(seeds))]); k = len(seeds)
    return m, np.sqrt((k - 1) / k * np.nansum((J - np.nanmean(J, 0)) ** 2, 0))
res = dict(cfg=cfg, nu=MID.tolist(), epochs={}, J={})
print(f"{cfg}: seeds {seeds}, D {Ds}")
print("Unsmoothed cells (6 tetrahedra per lattice cell, signed): D  frac inverted  std ln|J|")
for D in Ds:
    js = [R[(s, g, D)]["J"] for s in seeds for g in (1, -1)]
    res["J"][f"{D:.3f}"] = {k: float(np.mean([j[k] for j in js])) for k in js[0]}
    print(f"  {D:.1f}  {res['J'][f'{D:.3f}']['frac_neg']:.3f}  {res['J'][f'{D:.3f}']['std_ln']:.2f}")
print("\nD  R0 M | std lnV | within-nu fraction of variance ± | residual rms | rms(lnV-SC) | shear explains ± of the within-nu variance | mean a3/a1")
for D in Ds:
    e = res["epochs"].setdefault(f"{D:.3f}", {})
    for R0 in R0s:
        v = {k: agg(D, R0, k) for k in ("mean", "std", "a31", "corr", "frac_within", "expl", "sc_rms", "a31_all", "within", "okfrac", "std_lnV")}
        sc = np.nanmean([R[(s, g, D)][R0]["sc_mean"] for s in seeds for g in (1, -1)], 0)
        e[f"{R0:.1f}"] = dict({k: [np.asarray(x[0]).tolist(), np.asarray(x[1]).tolist()] for k, x in v.items()}, sc_mean=sc.tolist())
        M = 4 * np.pi / 3 * R0 ** 3 * 2.775e11
        print(f"{D:.1f} {R0:5.1f} {M:.1e} | {v['std_lnV'][0]:.3f} | {v['frac_within'][0]:.3f}±{v['frac_within'][1]:.3f} | {np.sqrt(v['within'][0]):.3f} | {v['sc_rms'][0]:.3f} | {v['expl'][0]:.3f}±{v['expl'][1]:.3f} | {v['a31_all'][0]:.3f}")
print("\nAt D=1, per nu bin: nu | E[lnV|nu] SC | std(lnV|nu) | corr(lnV, q|nu) | a3/a1")
if 1.0 in Ds:
    for R0 in R0s:
        q = res["epochs"]["1.000"][f"{R0:.1f}"]; print(f"R0={R0}")
        for i, nu in enumerate(MID):
            if np.isfinite(q["mean"][0][i]) and abs(nu) <= 3:
                print(f"  {nu:+.2f} | {q['mean'][0][i]:+.3f} {q['sc_mean'][i]:+.3f} | {q['std'][0][i]:.3f} | {q['corr'][0][i]:+.3f} | {q['a31'][0][i]:.3f}")
json.dump(res, open(os.path.join(here, f"patch_anisotropy_{cfg}.json"), "w"), indent=0, default=float)
if figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [r for r in (14.8, 20.7, 51.0) if r in R0s]; Col = {0.5: "#2a78d6", 0.8: "#eb6834", 1.0: "#1baf7a"}
    fig, axs = plt.subplots(2, len(show), figsize=(3.6 * len(show), 6), sharex=True, sharey="row", gridspec_kw=dict(hspace=0.08, wspace=0.05), facecolor="white")
    for c, r0 in enumerate(show):
        for D in (0.5, 0.8, 1.0):
            if D not in Ds: continue
            q = res["epochs"][f"{D:.3f}"][f"{r0:.1f}"]
            axs[0, c].plot(MID, q["mean"][0], color=Col[D], lw=2, label=f"PM, D = {D:g}"); axs[0, c].plot(MID, q["sc_mean"], color=Col[D], lw=1.5, ls=(0, (4, 2)), label=f"spherical collapse, D = {D:g}")
            axs[1, c].plot(MID, q["std"][0], color=Col[D], lw=2)
        M = 4 * np.pi / 3 * r0 ** 3 * 2.775e11
        axs[0, c].set_title(f"M = {M:.1e} $h^{{-1}}M_\\odot$", fontsize=10, loc="left"); axs[1, c].set_xlabel(r"$\nu = \delta_L/\sigma_{\rm lin}$", fontsize=10)
        for a in axs[:, c]:
            a.grid(True, color="#e4e3df", lw=0.7); a.set_axisbelow(True); a.tick_params(labelsize=8.5)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel(r"mean $\ln V$ of the patch at fixed $\nu$"); axs[1, 0].set_ylabel(r"scatter of $\ln V$ at fixed $\nu$"); axs[0, 0].set_xlim(-3.2, 3.2)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    os.makedirs(figdir, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(figdir, f"figM3_patch_anisotropy_{cfg}.{ext}"), dpi=170, bbox_inches="tight")
    print("wrote figure")
