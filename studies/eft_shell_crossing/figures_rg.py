"""RG-running figures: DiscoDJ cutoff-running experiment vs Carrasco–Hertzberg–Senatore (2012) Fig. 3,
the BAO-scale consequence, the (k_split, D) map, and counts in spheres.

    python3 figures_rg.py [results_dir] [figures_dir]

Inputs: results/rg_ks*.h5 (scripts/analyze.jl), results/cis_rg_ks*.h5 (scripts/cis_analysis.jl),
chs/chs_running.json (chs/chs_fig3.py), chs/chs2012_fig3_digitized.json.
Julia-written arrays appear with reversed axes: R[bin, epoch, model], fits[kmax, epoch, model].
"""
import os, sys, json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv = sys.argv[:1] + sys.argv[1:]
import figures as F                      # shared style, palette, Result
from figures import (save, errbar, lpt_color, PM_COLOR, BLUE, ORANGE, AQUA, VIOLET, RED, MAGENTA, GREEN, YELLOW,
                     INK, INK2, MUTED, GRID)

RES, FIG = F.RES, F.FIG
HERE = os.path.dirname(os.path.abspath(__file__))
C_LIGHT = 299792.458
SPLITS = [0.12, 0.15, 0.20, 0.25, 0.33, 0.50, 0.70]
TAG = {0.12: "rg_ks12", 0.15: "rg_ks15", 0.20: "rg_ks20", 0.25: "rg_ks25", 0.33: "rg_ks33", 0.50: "rg_ks50", 0.70: "rg_ks70"}
CHS = json.load(open(os.path.join(HERE, "chs", "chs_running.json")))
DIG = json.load(open(os.path.join(HERE, "chs", "chs2012_fig3_digitized.json")))


def load_rg():
    out = {}
    for ks in SPLITS:
        p = os.path.join(RES, f"{TAG[ks]}.h5")
        if os.path.exists(p):
            out[ks] = F.Result(TAG[ks])
    return out


def kmax_for(ks):
    """Fit range: k ≤ min(0.08, 0.4 k_split) (well inside the shared modes); ≥ 3 bins."""
    return float(max(min(0.08, 0.4 * ks), 0.04))


def c2_table(rg, q="dens", D=1.0, order=1):
    rows = {}
    for ks, r in rg.items():
        e = r.ei(D); km = kmax_for(ks)
        rows[ks] = {m: tuple(np.asarray(v)[e] for v in r.C2(q, m, km, order)) for m in r.models}
    return rows


# CHS curve, converted: c²(Λ) [c_light²] → response C2 = c²/(9 ℋ0² f0²) [(Mpc/h)²];  Λ → sharp k_split with
# equal removed σ_v² (their cosmology).
def chs_curve(kren="0.1"):
    ch = CHS["chs"]; L = np.array(ch["Lambda"]); run = np.array(ch["running"]); c2 = ch["c2_inf"][kren] + run
    d = CHS["dictionary"]
    ks_of = lambda lam: np.interp(lam, d["Lambda"], d["k_split_equiv_chs"])
    sel = (L >= 0.1) & (L <= 1.0)
    return ks_of(L[sel]), c2[sel], c2[sel] * ch["C2_per_c2"] * 1.0


def fig_chs_comparison(rg, name="figR1_chs_fig3_comparison", D=1.0, units="response"):
    """Top: the effective coefficient vs k_split — DiscoDJ PM, 4LPT, one-loop PT, CHS Fig. 3 (reconstructed and
    digitized) and their N-body points.  Bottom: ratios.  units = "response" (C2, (Mpc/h)²) or "c2" (c_eff², 1e-6 c²)."""
    tab = c2_table(rg, D=D)
    ks = np.array(sorted(tab)); f0 = CHS["chs"]["cosmology"]["f0"]
    eds = CHS["ours"]["c2_per_C2_EdS"] * 1e6                  # c²_eff [1e-6 c²] per C2 [(Mpc/h)²] in EdS (f = 1)
    conv_ours = (lambda C: C) if units == "response" else (lambda C: C * eds)
    conv_chs_c2 = (lambda c2: c2 * CHS["chs"]["C2_per_c2"]) if units == "response" else (lambda c2: c2 * 1e6)
    ylab = (r"$C_2^\delta$ at $D=1$  [$(h^{-1}$Mpc$)^2$]  (response $\Delta\delta=-C_2k^2\delta_L$)" if units == "response"
            else r"$c^2_{\rm eff}$  [$10^{-6}c^2$]")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1]))
    pm = np.array([tab[k]["pm"][0] for k in ks]); spm = np.array([tab[k]["pm"][1] for k in ks])
    l4 = np.array([tab[k]["lpt4"][0] for k in ks]); s4 = np.array([tab[k]["lpt4"][1] for k in ks])
    l3 = np.array([tab[k]["lpt3"][0] for k in ks])
    o = CHS["ours"]; kp = np.array(o["k_split"]); pt = np.array(o["C2_pt_1loop_kext0.01"]) * D ** 2
    # CHS (their cosmology): reconstructed band k_ren = 0.10 … 0.16 and the digitized published curves
    x10, c10, _ = chs_curve("0.1"); x16, c16, _ = chs_curve("0.16")
    a1.fill_between(x10, conv_chs_c2(np.minimum(c10, c16)), conv_chs_c2(np.maximum(c10, c16)), color=VIOLET, alpha=0.18, lw=0,
                    label="CHS Eq. 62, reconstructed ($k_{\\rm ren}$ 0.10–0.16)")
    dmap = lambda lam: np.interp(lam, CHS["dictionary"]["Lambda"], CHS["dictionary"]["k_split_equiv_chs"])
    for key, ls in (("kren_0.10", "-"), ("kren_0.18", "--")):
        p = np.array(DIG[key]); a1.plot(dmap(p[:, 0]), conv_chs_c2(p[:, 1]), color=VIOLET, lw=1.0, ls=ls,
                                        label=f"CHS Fig. 3 digitized ($k_{{\\rm ren}}$ = {key[-4:]})")
    for nb in DIG["nbody"]:
        errbar(a1, [dmap(nb["Lambda"])], [conv_chs_c2(nb["c2"])], [conv_chs_c2(nb["err"])], color=VIOLET, marker="D",
               ms=5, lw=0, elinewidth=1.4)
    a1.plot([], [], color=VIOLET, marker="D", ls="none", label="CHS N-body (Consuelo), Λ = 1/6, 1/3")
    a1.plot(kp, conv_ours(pt), color=INK, lw=1.2, ls=":", label="one-loop PT, this setup (EdS, $k_{\\rm uv}$ = 1.2)")
    errbar(a1, ks, conv_ours(l4), conv_ours(s4), color=lpt_color(4), marker="s", ms=4, lw=1.4, label="DiscoDJ 4LPT")
    a1.plot(ks, conv_ours(l3), color=lpt_color(2), marker="s", ms=3, lw=1.0, label="DiscoDJ 3LPT")
    errbar(a1, ks, conv_ours(pm), conv_ours(spm), color=PM_COLOR, marker="o", ms=5, lw=1.8, label="DiscoDJ PM (collisionless)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_ylabel(ylab)
    a1.set_title(f"Running with the cutoff — DiscoDJ vs CHS (2012) Fig. 3   (D = {D})", loc="left")
    a1.legend(fontsize=6.6, loc="upper right")
    # bottom: ratios to the CHS reconstruction (k_ren = 0.10) and to one-loop PT
    chs_at = np.interp(ks, x10, conv_chs_c2(c10))
    errbar(a2, ks, conv_ours(pm) / chs_at, conv_ours(spm) / chs_at, color=PM_COLOR, marker="o", ms=4, lw=1.6, label="PM / CHS")
    a2.plot(ks, conv_ours(l4) / chs_at, color=lpt_color(4), marker="s", ms=3, lw=1.2, label="4LPT / CHS")
    pt_at = np.interp(ks, kp, pt)
    errbar(a2, ks, pm / pt_at, spm / pt_at, color=PM_COLOR, marker="o", mfc="white", ms=4, lw=1.0, ls="--",
           label="PM / one-loop PT")
    a2.axhline(1, color=MUTED, lw=0.8); a2.set_yscale("log")
    a2.set_xlabel(r"$k_{\rm split}$ [$h$/Mpc]  (CHS Λ mapped to the sharp split with equal removed $\sigma_v^2$)")
    a2.set_ylabel("ratio"); a2.legend(fontsize=7, loc="best")
    for a in (a1, a2):
        a.set_xticks([0.12, 0.15, 0.2, 0.3, 0.5, 0.7]); a.set_xticklabels(["0.12", "0.15", "0.2", "0.3", "0.5", "0.7"])
    save(fig, name if units == "response" else name + "_c2units")
    return dict(k_split=ks.tolist(), pm=pm.tolist(), spm=spm.tolist(), lpt4=l4.tolist(), lpt3=l3.tolist(),
                pt1loop=pt_at.tolist(), chs_kren010=np.interp(ks, x10, c10 * CHS["chs"]["C2_per_c2"]).tolist())


def fig_running_shape(rg, name="figR2_running_shape", Ds=(0.3, 0.5, 1.0), ref=0.5):
    """Normalization-free: C2(k_split) − C2(k_ref) for PM, 4LPT and one-loop PT, and the CHS running."""
    fig, axes = plt.subplots(1, len(Ds), figsize=(3.3 * len(Ds), 3.5), sharey=False)
    o = CHS["ours"]; kp = np.array(o["k_split"]); pt = np.array(o["C2_pt_1loop_kext0.01"])
    for ax, D in zip(axes, Ds):
        tab = c2_table(rg, D=D); ks = np.array(sorted(tab))
        for m, col, lab, mk in (("pm", PM_COLOR, "PM", "o"), ("lpt4", lpt_color(4), "4LPT", "s"), ("lpt7", lpt_color(7), "7LPT", "s")):
            v = np.array([tab[k][m][0] for k in ks]); s = np.array([tab[k][m][1] for k in ks])
            vr = v - np.interp(ref, ks, v)
            errbar(ax, ks, vr, s, color=col, marker=mk, ms=4, lw=1.5, label=lab)
        ptD = pt * D ** 2
        ax.plot(kp, ptD - np.interp(ref, kp, ptD), color=INK, ls=":", lw=1.2, label="one-loop PT")
        x10, c10, C10 = chs_curve("0.1")
        if D == 1.0:
            ax.plot(x10, C10 - np.interp(ref, x10, C10), color=VIOLET, lw=1.2, label="CHS running (response units)")
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_xscale("log")
        ax.set_title(f"D = {D}", loc="left"); ax.set_xlabel(r"$k_{\rm split}$ [$h$/Mpc]")
    axes[0].set_ylabel(rf"$C_2^\delta(k_{{\rm split}})-C_2^\delta({ref})$ [$(h^{{-1}}$Mpc$)^2$]")
    axes[0].legend(fontsize=7)
    save(fig, name)


def knl_of_D(D):
    pk = json.load(open(os.path.join(HERE, "chs", "planck18_pk.json")))
    k = np.array(pk["k"]); P = np.array(pk["Pk"])
    s2 = np.concatenate([[0], np.cumsum(0.5 * (k[1:] ** 2 * P[1:] + k[:-1] ** 2 * P[:-1]) * np.diff(k))]) / (2 * np.pi ** 2)
    return np.interp(1.0 / np.asarray(D) ** 2, s2, k)        # σ_lin(k < k_NL) D = 1


def fig_map(rg, name="figR3_c2_map", q="dens"):
    """C2(k_split, D)/D² for PM and 4LPT, with the IR member's shell-crossed fraction and k_NL(D)."""
    ks = np.array(sorted(rg)); r0 = rg[ks[0]]; D = r0.D; sel = D >= 0.05
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    vals = {}
    for m in ("pm", "lpt4"):
        Z = np.full((sel.sum(), len(ks)), np.nan); S = np.zeros_like(Z)
        for j, k in enumerate(ks):
            C, s, *_ = rg[k].C2(q, m, kmax_for(k)); Z[:, j] = C[sel] / D[sel] ** 2; S[:, j] = s[sel] / D[sel] ** 2
        vals[m] = Z
    lim = np.nanmax(np.abs(vals["pm"]))
    from matplotlib.colors import TwoSlopeNorm
    cmap = plt.get_cmap("RdBu_r")
    for ax, Z, title in ((axes[0], vals["pm"], "PM: $C_2^\\delta/D^2$"), (axes[1], vals["lpt4"], "4LPT: $C_2^\\delta/D^2$"),
                         (axes[2], vals["pm"] - vals["lpt4"], "PM − 4LPT")):
        v = np.nanmax(np.abs(Z)) if ax is axes[2] else lim
        pc = ax.pcolormesh(np.arange(len(ks) + 1) - 0.5, np.arange(sel.sum() + 1) - 0.5, Z, cmap=cmap,
                           norm=TwoSlopeNorm(0, -v, v), shading="flat")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02, label=r"$(h^{-1}$Mpc$)^2$")
        # shell crossing of the IR member (modes below each split) and of the UV member (all modes)
        inv_ir = np.array([rg[k].inv("pm", "ir")[sel] for k in ks]).T
        cs = ax.contour(np.arange(len(ks)), np.arange(sel.sum()), np.log10(np.maximum(inv_ir, 1e-7)),
                        levels=[-4, -3, -2, -1], colors=INK, linewidths=0.9)
        ax.clabel(cs, fmt=lambda v: f"IR inv. {10**v:.0e}", fontsize=6)
        knl = knl_of_D(D[sel])
        ax.plot(np.interp(knl, ks, np.arange(len(ks)), left=np.nan, right=np.nan), np.arange(sel.sum()), color=YELLOW, lw=2,
                label=r"$k_{\rm NL}(D)$: $\sigma_{\rm lin}(k<k_{\rm NL})=1$")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks]); ax.set_xlabel(r"$k_{\rm split}$ [$h$/Mpc]")
        ax.set_title(title, loc="left")
    axes[0].set_yticks(range(sel.sum())); axes[0].set_yticklabels([f"{d:g}" for d in D[sel]], fontsize=6.5)
    axes[0].set_ylabel("D"); axes[0].legend(fontsize=7, loc="upper right")
    save(fig, name)


def fig_band_contrib(rg, name="figR4_band_contribution", Ds=(0.3, 0.5, 0.7, 1.0)):
    """Contribution of each shell [k_i, k_{i+1}) to C2: [C2(k_i) − C2(k_{i+1})]/ln(k_{i+1}/k_i), PM vs one-loop PT."""
    ks = np.array(sorted(rg)); mid = np.sqrt(ks[1:] * ks[:-1])
    o = CHS["ours"]; kp = np.array(o["k_split"]); pt = np.array(o["C2_pt_1loop_kext0.01"])
    ramp = ["#fbd3bf", "#f6a77f", "#eb6834", "#7a2e0f"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for c, D in zip(ramp, Ds):
        tab = c2_table(rg, D=D)
        v = np.array([tab[k]["pm"][0] for k in ks]); s = np.array([tab[k]["pm"][1] for k in ks])
        dv = (v[:-1] - v[1:]) / np.log(ks[1:] / ks[:-1]); ds = np.sqrt(s[:-1] ** 2 + s[1:] ** 2) / np.log(ks[1:] / ks[:-1])
        ptv = np.interp(ks, kp, pt) * D ** 2; dpt = (ptv[:-1] - ptv[1:]) / np.log(ks[1:] / ks[:-1])
        errbar(a1, mid, dv, ds, color=c, marker="o", ms=4, lw=1.5, label=f"PM, D = {D}")
        a1.plot(mid, dpt, color=c, ls=":", lw=1.2)
        knl = float(knl_of_D(D))
        errbar(a2, mid / knl, dv / dpt, ds / np.abs(dpt), color=c, marker="o", ms=4, lw=1.5, label=f"D = {D}")
    a1.set_xscale("log"); a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xlabel(r"shell centre $k$ [$h$/Mpc]"); a1.set_ylabel(r"$-dC_2^\delta/d\ln k_{\rm split}$ [$(h^{-1}$Mpc$)^2$]")
    a1.set_title("Which modes carry the response (dotted: one-loop PT)", loc="left", fontsize=9)
    a1.legend(fontsize=7)
    a2.set_xscale("log"); a2.axhline(1, color=MUTED, lw=0.8); a2.axvline(1, color=MUTED, lw=0.8, ls=":")
    a2.set_xlabel(r"$k/k_{\rm NL}(D)$"); a2.set_ylabel("PM / one-loop PT, per shell")
    a2.set_title("Decoupling of modes beyond the nonlinear scale", loc="left", fontsize=9)
    a2.legend(fontsize=7)
    save(fig, name)


BAO_K = [0.03, 0.05, 0.06, 0.1]


def bao_response(r, D=1.0, model="pm"):
    """Coherent Δδ/δ = R(k) and ΔP/P = (1+R)² − 1, stochastic P_ε/P_L, total T(k) at the BAO wavenumbers
    (linear interpolation between fundamental-mode bins)."""
    e = r.ei(D); i = r.mi(model)
    R, sR = r.R("dens", model); R = R[:, e]; sR = sR[:, e]
    PL = r.h["P_lin_delta"][:, e]
    Pd = r.h["P_ddelta/P"][:, e, i]
    Peps = np.maximum(Pd - R ** 2 * PL, 0.0)
    T = r.h["T"][:, e, i]
    k = r.k; good = np.isfinite(R)
    f = lambda y: np.interp(BAO_K, k[good], y[good], left=np.nan, right=np.nan)
    return dict(k=BAO_K, R=f(R), sR=f(sR), dP_coh=f((1 + R) ** 2 - 1), dP_stoch=f(Peps / PL), T=f(T))


def fig_bao(rg, name="figR5_bao_consequence", D=1.0):
    ks = np.array(sorted(rg))
    cols = [BLUE, AQUA, ORANGE, RED]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    table = {}
    for c, kb in zip(cols, BAO_K):
        coh = []; sc = []; st = []
        for k in ks:
            b = bao_response(rg[k], D)
            j = BAO_K.index(kb)
            coh.append(b["dP_coh"][j]); sc.append(2 * b["sR"][j]); st.append(b["dP_stoch"][j])
        coh = np.array(coh); sc = np.array(sc); st = np.array(st)
        ok = ks > kb * 1.05
        errbar(a1, ks[ok], 100 * coh[ok], 100 * sc[ok], color=c, marker="o", ms=4, lw=1.5, label=f"k = {kb}")
        a2.plot(ks[ok], st[ok], color=c, marker="o", ms=4, lw=1.5, label=f"k = {kb}")
        table[kb] = dict(k_split=ks[ok].tolist(), dP_coherent=coh[ok].tolist(), dP_stochastic=st[ok].tolist())
    lam5 = np.interp(0.2, CHS["dictionary"]["Lambda"], CHS["dictionary"]["k_split_equiv_ours"])
    for a in (a1, a2):
        a.axvline(lam5, color=MUTED, ls=":", lw=1); a.set_xscale("log"); a.set_xlabel(r"$k_{\rm split}$ [$h$/Mpc]")
        a.set_xticks([0.12, 0.15, 0.2, 0.3, 0.5, 0.7]); a.set_xticklabels(["0.12", "0.15", "0.2", "0.3", "0.5", "0.7"])
    a1.text(lam5, a1.get_ylim()[1] if False else 0, " 5 Mpc/h (Λ = 0.2)", color=INK2, fontsize=7, rotation=90, va="bottom")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_ylabel(r"coherent $\Delta P/P=(1+R)^2-1$  [%]")
    a1.set_title(f"Modes above $k_{{\\rm split}}$ coherently change identical long modes (PM, D = {D})", loc="left", fontsize=8.5)
    a2.set_yscale("log"); a2.set_ylabel(r"stochastic $P_\epsilon/P_L$")
    a2.set_title("…while the stochastic (uncorrelated) power they add is tiny", loc="left", fontsize=8.5)
    a1.legend(fontsize=7.5); a2.legend(fontsize=7.5)
    save(fig, name)
    return table


def fig_cis(name="figR6_counts_in_spheres", splits=(0.15, 0.2, 0.33), D=1.0):
    files = {ks: os.path.join(RES, f"cis_{TAG[ks]}.h5") for ks in splits}
    files = {k: v for k, v in files.items() if os.path.exists(v)}
    if not files: return
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    cols = {0.12: GREEN, 0.15: AQUA, 0.2: BLUE, 0.25: VIOLET, 0.33: MAGENTA, 0.5: RED, 0.7: YELLOW}
    out = {}
    for ks, p in files.items():
        h = h5py.File(p, "r"); R = h["R"][:]; ep = h["epochs"][:]; e = int(np.argmin(abs(ep - D)))
        g = lambda k: h[k][:, e, 0]                     # (R) for PM
        gs = lambda k: h[k + "_sigma"][:, e, 0] if k + "_sigma" in h else np.zeros(len(R))
        col = cols.get(ks, INK)
        errbar(axes[0], R, 100 * g("coherent"), 100 * gs("coherent"), color=col, marker="o", ms=4, lw=1.6, label=f"coherent, $k_s$ = {ks}")
        axes[0].plot(R, 100 * g("stoch"), color=col, ls="--", marker="^", ms=3, lw=1.0, label=f"stochastic ($k<k_s$)")
        axes[0].plot(R, 100 * g("band_lin"), color=col, ls=":", lw=1.0, label="band leakage (linear)")
        errbar(axes[2], R, g("C2R"), gs("C2R"), color=col, marker="o", ms=4, lw=1.5, label=f"spheres, $k_s$ = {ks}")
        out[ks] = {k: h[k][:, e, 0].tolist() for k in ("ratio_total", "coherent", "stoch", "band_lin", "remainder", "b", "C2R", "r_eft")}
        out[ks]["R"] = R.tolist()
        # Fourier C2 of the same config
        r = F.Result(TAG[ks]); C, s, *_ = r.C2("dens", "pm", kmax_for(ks))
        axes[2].axhspan(C[r.ei(D)] - s[r.ei(D)], C[r.ei(D)] + s[r.ei(D)], color=col, alpha=0.2, lw=0)
    axes[0].axhline(0, color=MUTED, lw=0.8); axes[0].set_xscale("log"); axes[0].set_yscale("symlog", linthresh=1e-2)
    axes[0].set_xlabel(r"$R$ [$h^{-1}$Mpc]"); axes[0].set_ylabel(r"$\Delta\sigma_R^2/\sigma_R^2$ components [%]")
    axes[0].set_title(f"Sphere variance change (PM, D = {D})", loc="left", fontsize=9); axes[0].legend(fontsize=6, ncol=1)
    axes[2].set_xscale("log"); axes[2].set_xlabel(r"$R$ [$h^{-1}$Mpc]")
    axes[2].set_ylabel(r"$C_{2,R}$: slope of $\Delta\delta_R$ on $\nabla^2\delta_R$ [$(h^{-1}$Mpc$)^2$]")
    axes[2].set_title("Real-space EFT slope vs Fourier $C_2$ (bands)", loc="left", fontsize=9); axes[2].legend(fontsize=7)
    # scatter for the middle split
    ks = 0.2 if 0.2 in files else list(files)[0]
    h = h5py.File(files[ks], "r"); R = h["R"][:]; ep = h["epochs"][:]; e = int(np.argmin(abs(ep - D)))
    sc = h["scatter"][:]          # (ns, 3, nR, E, nm)
    for c, Rv in zip((BLUE, AQUA, ORANGE), (20.0, 40.0, 80.0)):
        ir = int(np.argmin(abs(R - Rv))); x = sc[:, 0, ir, e, 0]; y = sc[:, 1, ir, e, 0]
        sx = np.std(x)
        axes[1].scatter(x / sx, y, s=2, color=c, alpha=0.35, lw=0, label=f"R = {Rv:.0f}")
        sl = h["C2R"][ir, e, 0]; xx = np.linspace(-3, 3, 10)
        axes[1].plot(xx, sl * xx * sx, color=c, lw=1.6)
    axes[1].set_xlabel(r"$\nabla^2\delta_R^{L}/\sigma(\nabla^2\delta_R^{L})$"); axes[1].set_ylabel(r"$\Delta\delta_R^{<k_s}$")
    axes[1].set_title(f"$\\Delta\\delta_R$ vs $\\nabla^2\\delta_R$ ($k_s$ = {ks}, PM, D = {D})", loc="left", fontsize=9)
    axes[1].legend(fontsize=7, markerscale=4)
    save(fig, name)
    return out


if __name__ == "__main__":
    rg = load_rg()
    print("splits available:", sorted(rg))
    summary = {}
    if len(rg) >= 3:
        summary["chs"] = fig_chs_comparison(rg); fig_chs_comparison(rg, units="c2")
        fig_running_shape(rg); fig_map(rg); fig_band_contrib(rg)
        summary["bao"] = {str(k): v for k, v in fig_bao(rg).items()}
        summary["bao_table"] = {str(ks): {m: {kk: (np.asarray(vv).tolist()) for kk, vv in bao_response(rg[ks], 1.0, m).items()}
                                         for m in ("pm", "lpt4")} for ks in sorted(rg)}
    summary["cis"] = {str(k): v for k, v in (fig_cis() or {}).items()}
    json.dump(summary, open(os.path.join(RES, "rg_summary.json"), "w"), indent=1, default=float)
    print("wrote rg_summary.json")
