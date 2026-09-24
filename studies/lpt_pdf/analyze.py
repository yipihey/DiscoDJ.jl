"""Post-process all runs: convergence of 2LPT and N-body separately, then the
2LPT-vs-N-body comparison of volume- and mass-weighted density PDFs.

Writes results/summary.json and figures/*.png.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import core
from cosmo import Cosmology, W_TH

FIG = os.environ.get("LPTPDF_FIGURES", os.path.join(C.SCRATCH, "figures"))
os.makedirs(FIG, exist_ok=True)

# palette (dataviz reference instance): categorical slots + blue/orange ramps
C_LPT, C_NB, C_LIN = "#2a78d6", "#eb6834", "#52514e"
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b", "#08254a"]
ORANGES = ["#f4b393", "#ef8f64", "#eb6834", "#c4521f", "#9a3f16"]
INK, GRID = "#0b0b0b", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": INK,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight"})

QS = np.array([1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999])
FINEBINS = np.linspace(np.log10(0.05), np.log10(20.0), 1201)


def meta(tag):
    return json.loads(str(np.load(os.path.join(C.RESULTS, tag + ".npz"))["meta"]))


def wquantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, v[o])


_cache = {}


def fields(tag):
    """Smoothed fields + raw mass weights for a run (from the saved CIC density)."""
    if tag not in _cache:
        rho = np.load(os.path.join(C.SCRATCH, tag + "_rho.npy"))
        _cache.clear()
        _cache[tag] = (rho, core.smoothed_fields(rho, C.L, C.R_SMOOTH))
    return _cache[tag]


def stats(tag):
    rho, sm = fields(tag)
    w = rho.ravel().astype(np.float64)
    out = {}
    for R, f in sm.items():
        v = f.ravel().astype(np.float64)
        lv = np.log10(np.clip(v, 1e-6, None))
        d = v - 1
        var = np.mean(d**2)
        hv, _ = np.histogram(lv, FINEBINS)
        hm, _ = np.histogram(lv, FINEBINS, weights=w)
        out[R] = dict(
            qV=np.quantile(v, QS), qM=wquantile(v, w, QS),
            var=var, S3=np.mean(d**3) / var**2,
            S4=(np.mean(d**4) - 3 * var**2) / var**3,
            cdfV=np.cumsum(hv) / hv.sum(), cdfM=np.cumsum(hm) / hm.sum())
    return out


def field_compare(t1, t2):
    _, s1 = fields(t1); s1 = {R: f.copy() for R, f in s1.items()}
    _, s2 = fields(t2)
    out = {}
    for R in C.R_SMOOTH:
        a, b = s1[R].ravel().astype(np.float64), s2[R].ravel().astype(np.float64)
        la, lb = np.log(np.clip(a, 1e-6, None)), np.log(np.clip(b, 1e-6, None))
        out[R] = dict(r=float(np.corrcoef(a, b)[0, 1]),
                      rms_rel=float(np.std(a - b) / np.std(b - 1)),
                      rms_log=float(np.std(la - lb)),
                      mean_log_diff=float(np.mean(la - lb)))
    return out, s1, s2


def pdf_from_npz(tag, R, kind):
    z = np.load(os.path.join(C.RESULTS, tag + ".npz"))
    return z["bins"], z[f"R{R:g}_{kind}"]


# ---------------------------------------------------------------------------
cosmo = Cosmology()


def sigma2_theory(R_s):
    """Linear variance of the fixed-amplitude field on the analysis grid."""
    n = C.N_ANA
    kx, ky, kz = core.kgrid(n, C.L, np.float64)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    ph = core.subgrid_modes(core.master_phases(C.N_MASTER, C.SEED, C.PHASES), n)
    msk = np.abs(ph) > 0
    wgt = np.full(k.shape, 2.0); wgt[..., 0] = 1; wgt[..., -1] = 1
    P = cosmo.Pk(k) * W_TH(k * C.R_F)**2
    return np.array([np.sum((wgt * msk * P * W_TH(k * R)**2)) / C.L**3 for R in R_s])


def tree_level():
    Rg = np.geomspace(3, 80, 120)
    s2 = sigma2_theory(Rg)
    g1 = np.gradient(np.log(s2), np.log(Rg))
    g2 = np.gradient(g1, np.log(Rg))
    S3 = 34 / 7 + g1
    S4 = 60712 / 1323 + 62 / 3 * g1 + 7 / 3 * g1**2 + 2 / 3 * g2
    f = lambda y: {R: float(np.interp(R, Rg, y)) for R in C.R_SMOOTH}
    return dict(sigma2=f(s2), S3=f(S3), S4=f(S4))


def main():
    tags = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(C.RESULTS, "*.npz")))
    lpt = sorted([t for t in tags if t.startswith("lpt_") and "paired" not in t],
                 key=lambda t: meta(t)["N"])
    nbres = sorted([t for t in tags if t.startswith("nb_") and "_m2_s100_ai0.04_log" in t
                    and "paired" not in t], key=lambda t: meta(t)["N"])
    S = {t: stats(t) for t in tags}
    M = {t: meta(t) for t in tags}
    tree = tree_level()
    summary = dict(config=dict(L=C.L, R_F=C.R_F, R_smooth=C.R_SMOOTH, N_ana=C.N_ANA,
                               quantiles=QS.tolist(), sigma_lin_RF=float(np.sqrt(cosmo.sigma2_TH(C.R_F))),
                               M_F=float(4 / 3 * np.pi * C.R_F**3 * 2.775e11 * cosmo.Om)),
                   tree_level=tree, runs={})
    for t in tags:
        summary["runs"][t] = dict(
            meta={k: v for k, v in M[t].items() if k not in ("cross_hist", "cross_hist_a")},
            **{f"R{R:g}": dict(var=S[t][R]["var"], S3=S[t][R]["S3"], S4=S[t][R]["S4"],
                               qV=S[t][R]["qV"].tolist(), qM=S[t][R]["qM"].tolist())
               for R in C.R_SMOOTH})

    # ---------------- convergence metrics ----------------
    def qdiff(t, ref, R, w):   # max relative quantile shift, 1%..99%
        a, b = S[t][R]["q" + w][1:-1], S[ref][R]["q" + w][1:-1]
        return float(np.max(np.abs(a / b - 1)))

    def ks(t, ref, R, w):
        return float(np.max(np.abs(S[t][R]["cdf" + w] - S[ref][R]["cdf" + w])))

    conv = {}
    for fam, lst in [("2lpt", lpt), ("nbody", nbres)]:
        ref = lst[-1]
        conv[fam] = {t: {f"R{R:g}": dict(dq_V=qdiff(t, ref, R, "V"), dq_M=qdiff(t, ref, R, "M"),
                                         KS_V=ks(t, ref, R, "V"), KS_M=ks(t, ref, R, "M"),
                                         dvar=S[t][R]["var"] / S[ref][R]["var"] - 1)
                         for R in C.R_SMOOTH} for t in lst[:-1]}
        conv[fam]["reference"] = ref
    nb_ref = nbres[-1]
    nb_tests = [t for t in tags if t.startswith("nb_N128")] + [t for t in tags if t.startswith("nb_N256") and "paired" not in t]
    conv["nbody_numerics"] = {t: {f"R{R:g}": dict(dq_V=qdiff(t, base, R, "V"), dq_M=qdiff(t, base, R, "M"),
                                                  dvar=S[t][R]["var"] / S[base][R]["var"] - 1)
                                  for R in C.R_SMOOTH}
                              for base in ["nb_N128_m2_s100_ai0.04_log", "nb_N256_m2_s100_ai0.04_log"]
                              for t in nb_tests if t != base and t.startswith(base[:7])}
    summary["convergence"] = conv

    # ---------------- 2LPT vs N-body ----------------
    lpt_ref = lpt[-1]
    comp = {}
    pairs = [("fixed", lpt_ref, nb_ref)]
    if "lpt_N512_paired" in S and "nb_N256_m2_s100_ai0.04_log_paired" in S:
        pairs.append(("paired", "lpt_N512_paired", "nb_N256_m2_s100_ai0.04_log_paired"))
    fieldpairs = {}
    for name, tl, tn in pairs:
        fc, s1, s2 = field_compare(tl, tn)
        if name == "fixed":
            fieldpairs = (s1, s2)
        comp[name] = {f"R{R:g}": dict(
            q_ratio_V=(S[tl][R]["qV"] / S[tn][R]["qV"]).tolist(),
            q_ratio_M=(S[tl][R]["qM"] / S[tn][R]["qM"]).tolist(),
            KS_V=ks(tl, tn, R, "V"), KS_M=ks(tl, tn, R, "M"),
            var_ratio=S[tl][R]["var"] / S[tn][R]["var"],
            S3=[S[tl][R]["S3"], S[tn][R]["S3"]], S4=[S[tl][R]["S4"], S[tn][R]["S4"]],
            field=fc[R]) for R in C.R_SMOOTH}
        comp[name]["runs"] = [tl, tn]
    summary["comparison"] = comp
    with open(os.path.join(C.RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    make_figures(S, M, lpt, nbres, nb_tests, pairs, fieldpairs, tree)
    print_tables(summary, lpt, nbres)


def _xlim(tag, R, kind, floor=1e-3, pad=0.08):
    b, p = pdf_from_npz(tag, R, kind)
    xc = 0.5 * (b[1:] + b[:-1])
    m = np.where(p > floor)[0]
    return xc[m[0]] - pad, xc[m[-1]] + pad


def _pdf_axes(nrow=2):
    fig, ax = plt.subplots(nrow, 4, figsize=(12, 5.2), sharex="col",
                           gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06, wspace=0.28))
    return fig, ax


def plot_family(S, lst, colors, label, fname, kind):
    ref = lst[-1]
    fig, ax = _pdf_axes()
    for j, R in enumerate(C.R_SMOOTH):
        for t, col in zip(lst, colors):
            b, p = pdf_from_npz(t, R, kind)
            _, pr = pdf_from_npz(ref, R, kind)
            xc = 0.5 * (b[1:] + b[:-1])
            m = pr > 1e-3
            ax[0, j].semilogy(xc, p, color=col, label=label(t))
            if t != ref:
                ax[1, j].plot(xc[m], p[m] / pr[m] - 1, color=col)
        ax[0, j].set_ylim(1e-3, 5); ax[0, j].set_title(f"$R_s$ = {R:g} Mpc/h", fontsize=9)
        ax[1, j].set_ylim(-0.1, 0.1); ax[1, j].axhline(0, color="#8a8984", lw=0.8)
        ax[1, j].set_xlabel(r"$\log_{10}(1+\delta_R)$")
        ax[0, j].set_xlim(*_xlim(ref, R, kind))
    ax[0, 0].set_ylabel("PDF (%s-weighted)" % ("volume" if kind == "vol" else "mass"))
    ax[1, 0].set_ylabel(f"ratio to {label(ref)} − 1")
    ax[0, 0].legend(fontsize=7, loc="lower center")
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def make_figures(S, M, lpt, nbres, nb_tests, pairs, fieldpairs, tree):
    lab = lambda t: f"{'2LPT' if t.startswith('lpt') else 'N-body'} {M[t]['N']}³"
    for kind in ("vol", "mass"):
        plot_family(S, lpt, BLUES[-len(lpt):], lab, f"conv_2lpt_{kind}.png", kind)
        plot_family(S, nbres, ORANGES[-len(nbres):], lab, f"conv_nbody_{kind}.png", kind)

    # N-body numerical parameters at 128^3 (and 256^3 steps)
    base = "nb_N128_m2_s100_ai0.04_log"
    others = [t for t in nb_tests if t.startswith("nb_N128") and t != base]
    if others:
        fig, ax = plt.subplots(2, 4, figsize=(12, 4.6), sharex="col", gridspec_kw=dict(hspace=0.08, wspace=0.28))
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
        for i, kind in enumerate(("vol", "mass")):
            for j, R in enumerate(C.R_SMOOTH):
                b, pr = pdf_from_npz(base, R, kind)
                xc = 0.5 * (b[1:] + b[:-1]); m = pr > 1e-3
                for t, col in zip(others + ["nb_N256_m2_s50_ai0.04_log"], cols):
                    if t not in S:
                        continue
                    refb = base if t.startswith("nb_N128") else "nb_N256_m2_s100_ai0.04_log"
                    _, p = pdf_from_npz(t, R, kind); _, prr = pdf_from_npz(refb, R, kind)
                    mm = prr > 1e-3
                    mt = M[t]
                    lb = f"{mt['N']}³ m{mt['mesh']} {mt['steps']} steps a_i={mt['a_i']:.3g}"
                    ax[i, j].plot(xc[mm], p[mm] / prr[mm] - 1, color=col, label=lb)
                ax[i, j].set_ylim(-0.05, 0.05); ax[i, j].axhline(0, color="#8a8984", lw=0.8)
                if i == 0:
                    ax[i, j].set_title(f"$R_s$ = {R:g} Mpc/h", fontsize=9)
                else:
                    ax[i, j].set_xlabel(r"$\log_{10}(1+\delta_R)$")
            ax[i, 0].set_ylabel(f"{'volume' if kind == 'vol' else 'mass'} PDF\nratio to fiducial − 1")
        h, l = ax[0, 0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
        for j, R in enumerate(C.R_SMOOTH):
            ax[1, j].set_xlim(*_xlim(base, R, "vol"))
        fig.savefig(os.path.join(FIG, "conv_nbody_numerics.png")); plt.close(fig)

    # 2LPT vs N-body: PDFs + ratio with convergence bands
    for kind in ("vol", "mass"):
        fig, ax = _pdf_axes()
        for j, R in enumerate(C.R_SMOOTH):
            for name, tl, tn in pairs:
                ls = "-" if name == "fixed" else "--"
                b, pl = pdf_from_npz(tl, R, kind); _, pn = pdf_from_npz(tn, R, kind)
                xc = 0.5 * (b[1:] + b[:-1]); m = pn > 1e-3
                if name == "fixed":
                    ax[0, j].semilogy(xc, pn, color=C_NB, label=f"N-body {M[tn]['N']}³")
                    ax[0, j].semilogy(xc, pl, color=C_LPT, label=f"2LPT {M[tl]['N']}³")
                    # Gaussian linear-theory PDF for reference (volume only)
                ax[1, j].plot(xc[m], pl[m] / pn[m] - 1, color=INK, ls=ls,
                              label=f"2LPT/N-body − 1 ({name}{' phases' if name == 'paired' else ' amp.'})")
            # convergence envelopes: difference between two highest resolutions
            for lst, col in [(lpt, C_LPT), (nbres, C_NB)]:
                b, p1 = pdf_from_npz(lst[-2], R, kind); _, p2 = pdf_from_npz(lst[-1], R, kind)
                xc = 0.5 * (b[1:] + b[:-1]); m = p2 > 1e-3
                e = np.abs(p1[m] / p2[m] - 1)
                ax[1, j].fill_between(xc[m], -e, e, color=col, alpha=0.25, lw=0)
            ax[0, j].set_ylim(1e-3, 5); ax[0, j].set_xlim(*_xlim(pairs[0][2], R, kind))
            ax[0, j].set_title(f"$R_s$ = {R:g} Mpc/h", fontsize=9)
            ax[1, j].set_ylim(-0.3, 0.3); ax[1, j].axhline(0, color="#8a8984", lw=0.8)
            ax[1, j].set_xlabel(r"$\log_{10}(1+\delta_R)$")
        ax[0, 0].set_ylabel("PDF (%s-weighted)" % ("volume" if kind == "vol" else "mass"))
        ax[1, 0].set_ylabel("2LPT / N-body − 1")
        ax[0, 0].legend(fontsize=7, loc="lower center")
        ax[1, 0].legend(fontsize=6, loc="upper left")
        fig.savefig(os.path.join(FIG, f"compare_{kind}.png")); plt.close(fig)

    # quantile comparison: 2LPT / N-body - 1 for density quantiles, with the
    # resolution-convergence estimate of each family as error bars
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    for j, R in enumerate(C.R_SMOOTH):
        for w, mk, off in (("V", "o", -0.12), ("M", "s", 0.12)):
            for name, tl, tn in pairs:
                y = S[tl][R]["q" + w] / S[tn][R]["q" + w] - 1
                eL = np.abs(S[lpt[-2]][R]["q" + w] / S[lpt[-1]][R]["q" + w] - 1)
                eN = np.abs(S[nbres[-2]][R]["q" + w] / S[nbres[-1]][R]["q" + w] - 1)
                xi = np.arange(len(QS)) + off + (0.06 if name == "paired" else 0)
                ax[j].errorbar(xi, 100 * y, yerr=100 * np.hypot(eL, eN), fmt=mk, ms=4,
                               color=C_LPT if w == "V" else C_NB, mfc="white" if name == "paired" else None,
                               capsize=2, lw=1,
                               label=f"{'volume' if w == 'V' else 'mass'}-weighted ({name})")
        ax[j].axhline(0, color="#8a8984", lw=0.8)
        ax[j].set_xticks(range(len(QS)), [f"{100*q:g}%" for q in QS], rotation=45, fontsize=7)
        ax[j].set_title(f"$R_s$ = {R:g} Mpc/h", fontsize=9); ax[j].set_xlabel("quantile of $1+\\delta_R$")
    ax[0].set_ylabel("2LPT / N-body − 1  [%]")
    ax[0].legend(fontsize=6.5, loc="lower right")
    fig.savefig(os.path.join(FIG, "compare_quantiles.png")); plt.close(fig)

    # moments vs resolution
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(wspace=0.3))
    for k, (key, ttl) in enumerate([("var", r"$\sigma^2$"), ("S3", r"$S_3$"), ("S4", r"$S_4$")]):
        for j, R in enumerate(C.R_SMOOTH):
            for lst, col, mk in [(lpt, C_LPT, "o"), (nbres, C_NB, "s")]:
                Ns = [M[t]["N"] for t in lst]
                ys = [S[t][R][key] for t in lst]
                ax[k].plot(Ns, ys, marker=mk, color=col, ms=4, lw=1.2,
                           alpha=0.4 + 0.2 * j, label=("2LPT" if col == C_LPT else "N-body") if j == 0 else None)
            if key in tree:
                ax[k].axhline(tree[key][R], color=C_LIN, lw=0.8, ls=":")
            ax[k].annotate(f"{R:g}", (Ns[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color="#52514e")
        ax[k].set_xscale("log", base=2); ax[k].set_xlabel("particles per dimension")
        ax[k].set_title(ttl + "  (dotted: linear / tree level)", fontsize=9)
        if key == "var":
            ax[k].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "moments_vs_resolution.png")); plt.close(fig)

    # shell crossing
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(wspace=0.3))
    for lst, col, nm in [(lpt, C_LPT, "2LPT"), (nbres, C_NB, "N-body")]:
        ax[0].plot([M[t]["N"] for t in lst], [100 * M[t]["cross_ever"] for t in lst], "o-", color=col, label=nm + " (ever)")
        ax[0].plot([M[t]["N"] for t in lst], [100 * M[t]["cross_final"] for t in lst], "o--", color=col, label=nm + " (at z=0)", alpha=0.6)
        t = lst[-1]
        ax[1].plot(M[t]["cross_hist_a"], 100 * np.array(M[t]["cross_hist"]), color=col, label=f"{nm} {M[t]['N']}³")
    ax[0].axhline(1.0, color="#e34948", lw=0.8, ls=":"); ax[0].text(64, 0.95, "1% target", fontsize=7, color="#52514e")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("particles per dimension"); ax[0].set_ylabel("shell-crossed mass [%]")
    ax[1].set_xlabel("scale factor a"); ax[1].set_ylabel("ever-crossed mass [%]")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "shell_crossing.png")); plt.close(fig)

    # field-level scatter 2LPT vs N-body
    if fieldpairs:
        s1, s2 = fieldpairs
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.1), gridspec_kw=dict(wspace=0.3))
        for j, R in enumerate(C.R_SMOOTH):
            a = np.log10(np.clip(s2[R].ravel()[::7], 1e-3, None))
            b = np.log10(np.clip(s1[R].ravel()[::7], 1e-3, None))
            ax[j].hist2d(a, b - a, bins=[120, 120], range=[[-1, 1], [-0.12, 0.12]], cmap="Blues", norm=matplotlib.colors.LogNorm(), rasterized=True)
            ax[j].axhline(0, color="#8a8984", lw=0.8)
            ax[j].set_title(f"$R_s$ = {R:g} Mpc/h", fontsize=9); ax[j].set_xlabel(r"$\log_{10}(1+\delta)_{\rm N\!-\!body}$")
            ax[j].grid(False)
        ax[0].set_ylabel(r"$\log_{10}\rho_{\rm 2LPT} - \log_{10}\rho_{\rm N\!-\!body}$")
        fig.savefig(os.path.join(FIG, "field_level.png")); plt.close(fig)


def print_tables(s, lpt, nbres):
    print(json.dumps(s["convergence"], indent=1, default=float)[:6000])
    print(json.dumps(s["comparison"], indent=1, default=float)[:8000])


if __name__ == "__main__":
    main()
