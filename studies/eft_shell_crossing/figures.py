"""Figures for the EFT / shell-crossing study.

    python3 figures.py [results_dir] [figures_dir]

Reads the reduced statistics written by scripts/analyze.jl (results/<config>.h5) and the PM-LPT residual
tables written by scripts/residuals.jl (results/residuals_<config>.h5).  HDF5 arrays written from Julia
appear here with reversed axes:  R[bin, epoch, model], fit C2[kmax, epoch, model], inv_uv[epoch, model].
"""
import os, sys, json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

# ── style: reference categorical palette (fixed order), one sequential ramp for LPT order ─────────
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
PM_COLOR = ORANGE
# LPT order n = 1..7: light → dark blue (sequential, ordinal)
LPT_RAMP = ["#b9d3f2", "#9cc0ec", "#7aaae4", "#5592dc", "#2a78d6", "#1d5ea9", "#12447c"]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.dpi": 110, "mathtext.fontset": "dejavusans",
})


def lpt_color(n):
    return LPT_RAMP[min(n, 7) - 1]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


class Result:
    """Reduced statistics of one configuration (results/<name>.h5)."""

    def __init__(self, name):
        self.name = name
        self.h = h5py.File(os.path.join(RES, f"{name}.h5"), "r")
        a = self.h.attrs["models"]
        self.models = (a.decode() if isinstance(a, bytes) else a).split(",")
        self.k = self.h["k"][:]
        self.D = self.h["epochs"][:]
        self.n_long = int(self.h.attrs["n_long"])

    def mi(self, model):
        return self.models.index(model)

    def ei(self, D):
        return int(np.argmin(abs(self.D - D)))

    def R(self, q, model):          # (bin, epoch)
        return self.h[f"{q}/R"][:, :, self.mi(model)], self.h[f"{q}/sigma"][:, :, self.mi(model)]

    def C2(self, q, model, kmax=0.08, order=1):
        g = self.h[f"{q}/fit{order}"]
        j = int(np.argmin(abs(g["kmax"][:] - kmax)))
        i = self.mi(model)
        return g["C2"][j, :, i], g["sC2"][j, :, i], g["C4"][j, :, i], g["sC4"][j, :, i], g["chi2dof"][j, :, i]

    def sweep(self, q, model, e, order=1):
        g = self.h[f"{q}/fit{order}"]
        i = self.mi(model)
        return g["kmax"][:], g["C2"][:, e, i], g["sC2"][:, e, i], g["C4"][:, e, i], g["sC4"][:, e, i]

    def inv(self, model, member="uv"):
        return self.h[f"inv_{member}"][:, self.mi(model)]

    def P(self, kind, model):
        g = self.h[f"P_{kind}"]
        s = g["sigma"][:, :, self.mi(model)] if "sigma" in g else None
        return g["P"][:, :, self.mi(model)], s


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────
def shanks(s3, s5, s7):
    """Shanks / geometric-tail estimate of the odd LPT series limit from partial sums S3, S5, S7, valid
    while the increment ratio r = (S7−S5)/(S5−S3) is in (0, 1); NaN otherwise."""
    d1, d2 = s5 - s3, s7 - s5
    r = np.where(d1 != 0, d2 / d1, np.nan)
    ok = (r > 0) & (r < 0.7)
    lim = s7 + d2 * r / (1 - r)
    return np.where(ok, lim, np.nan), r


def first_crossing(res, model="pm", thresh=1e-4):
    f = res.inv(model)
    i = np.where(np.nan_to_num(f) > thresh)[0]
    return res.D[i[0]] if len(i) else np.nan


def errbar(ax, x, y, s, **kw):
    ax.errorbar(x, y, yerr=s, capsize=0, elinewidth=0.9, **kw)


# ── Figure 3 (central): C2(D)/D² for every LPT order and PM, with the shell-crossed fraction ────────
def fig_c2_vs_time(r, name="fig3_c2_vs_time", kmax=0.08, q="disp"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    D = r.D
    sel = D >= 0.05
    for n in (1, 3, 5, 7):
        C, s, *_ = r.C2(q, f"lpt{n}", kmax)
        a1.plot(D[sel], C[sel] / D[sel] ** 2, color=lpt_color(n), lw=1.4, label=f"{n}LPT")
    C3, _, *_ = r.C2(q, "lpt3", kmax); C5, _, *_ = r.C2(q, "lpt5", kmax); C7, _, *_ = r.C2(q, "lpt7", kmax)
    lim, _ = shanks(C3, C5, C7)
    a1.plot(D[sel], lim[sel] / D[sel] ** 2, color=lpt_color(7), lw=1.0, ls="--", label=r"LPT$_\infty$ (Shanks 3,5,7)")
    if "pm" in r.models:
        C, s, *_ = r.C2(q, "pm", kmax)
        errbar(a1, D[sel], C[sel] / D[sel] ** 2, s[sel] / D[sel] ** 2, color=PM_COLOR, marker="o", ms=4, lw=1.6,
               label="PM (collisionless)")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_ylim(-8, 2.5)
    a1.set_ylabel(r"$C_2(D)/D^2$  [$(h^{-1}\mathrm{Mpc})^2$]")
    a1.set_title(r"Long-mode displacement response to the added band: $\Delta\psi_L=-C_2k^2\psi_L$  "
                 f"($k\\leq{kmax}\\,h$/Mpc)", loc="left")
    a1.legend(ncol=2, loc="lower left")
    Dsc = first_crossing(r) if "pm" in r.models else first_crossing(r, "lpt7")
    for a in (a1, a2):
        a.axvline(Dsc, color=MUTED, lw=0.9, ls=":")
    a1.text(Dsc, 2.2, " first shell crossing", color=INK2, fontsize=7.5, va="top")
    for n in (1, 3, 7):
        f = r.inv(f"lpt{n}"); a2.plot(D[sel], np.where(f[sel] > 0, f[sel], np.nan), color=lpt_color(n), lw=1.2,
                                      label=f"UV {n}LPT")
    if "pm" in r.models:
        a2.plot(D[sel], r.inv("pm")[sel], color=PM_COLOR, marker="o", ms=3, label="UV PM")
        fi = r.inv("pm", "ir"); a2.plot(D[sel], np.where(fi[sel] > 0, fi[sel], np.nan), color=PM_COLOR, ls="--",
                                        lw=1.0, label="IR PM")
    a2.set_yscale("log"); a2.set_ylim(1e-5, 1)
    a2.set_ylabel("inverted mass fraction")
    a2.set_xlabel(r"growth factor $D=a$ (EdS)")
    a2.set_xscale("log"); a2.set_xticks([0.05, 0.1, 0.2, 0.3, 0.5, 1.0]); a2.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.5", "1"])
    a2.legend(ncol=2, loc="lower right")
    save(fig, name)


# ── Figure 4: LPT order convergence — residual PM − LPTn and the series limit ──────────────────────
def fig_lpt_convergence(r, name="fig4_lpt_convergence", kmax=0.08, q="disp"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    D = r.D; sel = D >= 0.05
    Cp, sp, *_ = r.C2(q, "pm", kmax)
    C = {n: r.C2(q, f"lpt{n}", kmax)[0] for n in range(1, 8)}
    lim, rat = shanks(C[3], C[5], C[7])
    for n in (1, 3, 5, 7):
        d = Cp - C[n]
        a1.plot(D[sel], np.abs(d[sel]), color=lpt_color(n), lw=1.4, label=f"|PM − {n}LPT|")
        neg = sel & (d < 0)
        a1.plot(D[neg], np.abs(d[neg]), ls="none", marker="o", mfc="white", mec=lpt_color(n), ms=3.5)
    d = Cp - lim
    a1.plot(D[sel], np.abs(d[sel]), color=INK, lw=1.8, label=r"|PM − LPT$_\infty$|")
    if f"resid_{q}" in r.h:      # jackknife error of the realization-level PM − 7LPT residual
        g = r.h[f"resid_{q}"]; j = int(np.argmin(abs(g["kmax"][:] - kmax)))
        a1.plot(D[sel], g["sC2"][j, :, r.mi("lpt7")][sel], color=MUTED, lw=1.0, ls=":",
                label=r"$1\sigma$ of PM − 7LPT (jackknife)")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_ylim(1e-5, 30)
    a1.set_xlabel("D"); a1.set_ylabel(r"$|C_2^{\rm PM}-C_2^{(n)}|$  [$(h^{-1}$Mpc$)^2$]")
    a1.set_title("Residual of the long-mode response (open: negative)", loc="left")
    a1.legend(loc="upper left", fontsize=7.5)
    # right: ratio of successive odd-order increments (series convergence) and crossing
    a2.plot(D[sel], rat[sel], color=lpt_color(7), lw=1.6, label=r"$(C_2^{(7)}-C_2^{(5)})/(C_2^{(5)}-C_2^{(3)})$")
    a2.axhline(1, color=RED, lw=1, ls="--"); a2.text(0.06, 1.05, "series diverges above", color=INK2, fontsize=7.5)
    Dsc = first_crossing(r)
    for a in (a1, a2):
        a.axvline(Dsc, color=MUTED, lw=0.9, ls=":")
    a2.text(Dsc, 1.9, " first shell crossing (PM)", color=INK2, fontsize=7.5, va="top")
    a2.set_xscale("log"); a2.set_ylim(0, 2.0)
    a2.set_xlabel("D"); a2.set_title("Convergence of the odd LPT series", loc="left")
    a2.legend(loc="upper left", fontsize=7.5)
    save(fig, name)


# ── Figure 2: field-level response R(k) at three epochs ──────────────────────────────────────────
def fig_response_k(r, epochs=(0.15, 0.3, 0.5), name="fig2_response_k", q="disp", kmax=0.08):
    fig, axes = plt.subplots(1, len(epochs), figsize=(3.2 * len(epochs), 3.4), sharey=False)
    k = r.k
    for ax, D in zip(axes, epochs):
        e = r.ei(D)
        for m, col, lab in (("lpt3", lpt_color(3), "3LPT"), ("lpt7", lpt_color(7), "7LPT"), ("pm", PM_COLOR, "PM")):
            if m not in r.models: continue
            R, s = r.R(q, m)
            errbar(ax, k, R[:, e] / k ** 2, s[:, e] / k ** 2, color=col, marker="o" if m == "pm" else None,
                   ms=3.5, lw=1.3, label=lab)
            C2, *_ = r.C2(q, m, kmax)
            ax.plot([0, kmax], [-C2[e]] * 2, color=col, lw=0.8, ls="--")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(r"$k$ [$h$/Mpc]"); ax.set_title(f"D = {r.D[e]:.2f}, inverted {100*np.nan_to_num(r.inv('pm')[e] if 'pm' in r.models else r.inv('lpt7')[e]):.2f}%", loc="left")
        ax.axvspan(kmax, k.max() * 1.05, color=GRID, alpha=0.5, lw=0)
    axes[0].set_ylabel(r"$R(k)/k^2=\langle\Delta\psi_\parallel\psi_{L\parallel}^*\rangle/(k^2\langle|\psi_{L\parallel}|^2\rangle)$")
    axes[0].legend(loc="best")
    save(fig, name)


# ── Figure 5: EFT fit-range stability C2(kmax) and C2,C4 ─────────────────────────────────────────
def fig_kmax(r, epochs=(0.15, 0.3, 0.5), name="fig5_kmax_stability", q="disp", model="pm"):
    fig, axes = plt.subplots(1, len(epochs), figsize=(3.2 * len(epochs), 3.2))
    for ax, D in zip(axes, epochs):
        e = r.ei(D)
        km, C2, s2, _, _ = r.sweep(q, model, e, order=1)
        errbar(ax, km, C2, s2, color=PM_COLOR if model == "pm" else lpt_color(7), marker="o", ms=3.5, lw=1.4,
               label=r"$-C_2k^2$")
        km, C2b, s2b, C4, s4 = r.sweep(q, model, e, order=2)
        errbar(ax, km * 1.01, C2b, s2b, color=VIOLET, marker="s", ms=3.5, lw=1.2, label=r"$-C_2k^2+C_4k^4$")
        ax.set_xlabel(r"$k_{\max}$ [$h$/Mpc]"); ax.set_title(f"{model.upper()}, D = {r.D[e]:.2f}", loc="left")
    axes[0].set_ylabel(r"fitted $C_2$ [$(h^{-1}$Mpc$)^2$]"); axes[0].legend(loc="best")
    save(fig, name)


# ── Figure 8: Eulerian density vs Lagrangian displacement estimators; kinematic smearing ──────────
def fig_density_vs_disp(r, sigma_psi_band, name="fig8_density_vs_displacement", kmax=0.08):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    D = r.D; sel = D >= 0.05
    for m, col, lab in (("lpt1", lpt_color(1), "1LPT"), ("lpt7", lpt_color(7), "7LPT"), ("pm", PM_COLOR, "PM")):
        if m not in r.models: continue
        Cd, sd, *_ = r.C2("dens", m, kmax); Cp, *_ = r.C2("disp", m, kmax)
        errbar(ax, D[sel], Cd[sel] / D[sel] ** 2, sd[sel] / D[sel] ** 2, color=col, marker="s", ms=3, lw=1.3,
               label=f"{lab}: density $\\delta$")
        ax.plot(D[sel], Cp[sel] / D[sel] ** 2, color=col, lw=1.0, ls="--", label=f"{lab}: displacement $\\psi$")
    ax.axhline(sigma_psi_band ** 2 / 6, color=INK, lw=1.0, ls=":",
               label=r"kinematic smearing $\sigma^2_{\psi,\rm band}/6$")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(-8, 4.5)
    ax.set_xlabel("D"); ax.set_ylabel(r"$C_2/D^2$ [$(h^{-1}$Mpc$)^2$]")
    ax.set_title("Eulerian density response is dominated by displacement smearing", loc="left")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    save(fig, name)



# ── Figure 1: T(k, D) = P_UV/P_IR − 1 (Eulerian density power) ────────────────────────────────────
def fig_T(r, epochs=(0.1, 0.2, 0.3, 0.5, 1.0), name="fig1_power_ratio"):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    T = r.h["T"][:]; sT = r.h["T_sigma"][:] if "T_sigma" in r.h else np.zeros_like(T)
    k = r.k
    ramp = ["#fbd3bf", "#f6a77f", "#eb6834", "#b84a1c", "#7a2e0f"]          # one hue, light → dark = later
    for ax, model, title in ((axes[0], "pm", "PM (collisionless)"), (axes[1], "lpt7", "7LPT (single stream)")):
        if model not in r.models: continue
        for c, D in zip(ramp, epochs):
            e = r.ei(D)
            errbar(ax, k, T[:, e, r.mi(model)], sT[:, e, r.mi(model)], color=c, marker="o", ms=3, lw=1.3,
                   label=f"D = {r.D[e]:.2g}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(r"$k$ [$h$/Mpc]  (all $k<k_{\rm split}$: identical initial modes)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(r"$T(k,D)=P_{\rm UV}/P_{\rm IR}-1$")
    axes[0].legend(loc="lower left", fontsize=7.5)
    save(fig, name)


# ── Figure 6: deterministic vs stochastic ────────────────────────────────────────────────────────
def fig_stochastic(r, epochs=(0.2, 0.35, 0.6, 1.0), name="fig6_deterministic_vs_stochastic", model="pm"):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    k = r.k
    ramp = ["#fbd3bf", "#f6a77f", "#eb6834", "#7a2e0f"]
    fits = {}
    for ax, q, pk, plin, lab, slope in ((axes[0], "disp", "cond_psi", "P_lin_psi", r"displacement $\psi_\parallel$", 2),
                                         (axes[1], "dens", "cond_delta", "P_lin_delta", r"density $\delta$", 4)):
        if f"P_{pk}" not in r.h: continue
        Pe, sPe = r.P(pk, model)
        PL = r.h[plin][:]                                  # (bin, epoch)
        R, _ = r.R(q, model)
        for c, D in zip(ramp, epochs):
            e = r.ei(D)
            ax.plot(k, (R[:, e] ** 2) * PL[:, e], color=c, lw=1.4, ls="-")
            ax.plot(k, Pe[:, e], color=c, lw=1.2, ls="--", marker="o", ms=2.5)
            sel = (k <= 0.1) & (Pe[:, e] > 0)
            if sel.sum() > 3:
                n, a = np.polyfit(np.log(k[sel]), np.log(Pe[sel, e]), 1)
                fits[(q, float(r.D[e]))] = n
        e = r.ei(epochs[-1]); kk = np.array([k[0], 0.1])
        ax.plot(kk, Pe[np.argmin(abs(k - 0.1)), e] * (kk / 0.1) ** slope, color=MUTED, lw=1, ls=":")
        ax.text(kk[0] * 1.1, Pe[np.argmin(abs(k - 0.1)), e] * (kk[0] / 0.1) ** slope, f"$k^{slope}$", color=INK2, fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_title(f"{lab}: solid $R^2P_L$ (deterministic), dashed $P_\\epsilon$ (stochastic)", loc="left", fontsize=8.5)
    axes[0].set_ylabel(r"$P(k)$ [$(h^{-1}$Mpc$)^{3}$ × units$^2$]")
    from matplotlib.lines import Line2D
    axes[1].legend([Line2D([], [], color=c) for c in ramp], [f"D = {d}" for d in epochs], loc="lower right", fontsize=7.5)
    save(fig, name)
    return fits


# ── Figure 7: R(k) of PM against 4LPT and 7LPT across crossing ────────────────────────────────────
def fig_pm_vs_lpt_k(r, epochs=(0.2, 0.3, 0.4, 0.6), name="fig7_pm_vs_lpt", q="disp"):
    """Top: R(k) of PM, 4LPT, 7LPT.  Bottom: the realization-level departure (R_PM − R_7LPT)/k² — flat in k
    if the departure has the leading EFT form −ΔC₂ k²."""
    resid = f"resid_{q}" in r.h
    fig, axes = plt.subplots(2 if resid else 1, len(epochs), figsize=(2.9 * len(epochs), 6.0 if resid else 3.2),
                             squeeze=False)
    for ax in axes[0, 1:]:
        ax.sharey(axes[0, 0])
    k = r.k
    for j, D in enumerate(epochs):
        ax = axes[0, j]; e = r.ei(D)
        for m, col, lab in (("lpt4", lpt_color(4), "4LPT"), ("lpt7", lpt_color(7), "7LPT"), ("pm", PM_COLOR, "PM")):
            R, s = r.R(q, m)
            errbar(ax, k, -R[:, e] / k ** 2 / D ** 2, s[:, e] / k ** 2 / D ** 2, color=col, lw=1.4,
                   marker="o" if m == "pm" else None, ms=3, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylim(-5, 3)
        ax.set_title(f"D = {r.D[e]:.2f}  ({100*r.inv('pm')[e]:.2g}% inverted)", loc="left", fontsize=8.5)
        if resid:
            g = r.h[f"resid_{q}"]; i7 = r.mi("lpt7")
            dR = g["dR"][:, e, i7]; sd = g["sigma"][:, e, i7]
            ax2 = axes[1, j]
            errbar(ax2, k, -dR / k ** 2, sd / k ** 2, color=INK, marker="o", ms=3, lw=1.3)
            jj = int(np.argmin(abs(g["kmax"][:] - 0.08)))
            c = g["C2"][jj, e, i7]; sc = g["sC2"][jj, e, i7]
            ax2.axhspan(c - sc, c + sc, xmax=0.08 / (k.max() * 1.05), color=PM_COLOR, alpha=0.3, lw=0)
            ax2.axhline(0, color=MUTED, lw=0.8); ax2.set_xlim(0, k.max() * 1.05)
            top = np.nanmax(np.abs(-dR / k ** 2) + sd / k ** 2); ax2.set_ylim(-0.3 * top, 1.45 * top)
            ax2.set_xlabel(r"$k$ [$h$/Mpc]")
            ax2.text(0.03, 0.93, f"$\\Delta C_2$ = {c:.3g} ± {sc:.2g}", transform=ax2.transAxes, fontsize=7.5, color=INK2,
                     va="top", bbox=dict(fc="white", ec="none", alpha=0.8))
        else:
            ax.set_xlabel(r"$k$ [$h$/Mpc]")
    axes[0, 0].set_ylabel(r"$-R(k)/(k^2D^2)$  [$(h^{-1}$Mpc$)^2$]")
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    if resid:
        axes[1, 0].set_ylabel(r"$-(R_{\rm PM}-R_{\rm 7LPT})/k^2$  [$(h^{-1}$Mpc$)^2$]")
    save(fig, name)


# ── Figure 9: velocity response = d(D R)/dD (kinematic consistency) ───────────────────────────────
def fig_velocity(r, name="fig9_velocity_response", kmax=0.08):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    D = r.D; sel = D >= 0.05
    for m, col in (("lpt3", lpt_color(3)), ("lpt7", lpt_color(7)), ("pm", PM_COLOR)):
        Cv, sv, *_ = r.C2("vel", m, kmax); Cd, *_ = r.C2("disp", m, kmax)
        pred = np.gradient(Cd * D, D)            # Δv = d(Δψ)/dD  ⇒  C2_v = d(D·C2_ψ)/dD with x_v = ψ₁
        errbar(ax, D[sel], Cv[sel] / D[sel], sv[sel] / D[sel], color=col, marker="o", ms=3, lw=1.3, label=f"{m.upper()} measured")
        ax.plot(D[sel], pred[sel] / D[sel], color=col, lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_ylim(-25, 15)
    ax.set_xlabel("D"); ax.set_ylabel(r"$C_2^{v}/D$  [$(h^{-1}$Mpc$)^2$]")
    ax.set_title(r"Velocity response; dashed: $d(D\,C_2^\psi)/dD$ from the displacement", loc="left", fontsize=8.5)
    ax.legend(fontsize=7.5)
    save(fig, name)


# ── Figure 10: PM numerical convergence (single realization, PM − 7LPT) ────────────────────────────
def fig_pm_convergence(name="fig10_pm_convergence"):
    runs = [("conv_pm256m2", "CIC, 256³ / 512³ mesh", lpt_color(2), "-"),
            ("conv_pm384m2", "CIC, 384³ / 768³", lpt_color(4), "-"),
            ("conv_pm512m2lean", "CIC, 512³ / 1024³", lpt_color(7), "-"),
            ("conv_pm384m2s400", "CIC, 384³ / 768³, 2× steps", lpt_color(4), ":"),
            ("conv_pm512m15", "CIC, 512³ / 768³", VIOLET, "-"),
            ("conv_pm384m2rs", "CIC + sheet resampling, 384³", AQUA, "-"),
            ("conv_pm256m2bf", "suppressed*, 256³ / 512³ (production)", PM_COLOR, "-"),
            ("conv_pm384m2bf", "suppressed*, 384³ / 768³", RED, "--")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for f, lab, col, ls in runs:
        p = os.path.join(RES, f"residuals_{f}.json")
        if not os.path.exists(p): continue
        t = json.load(open(p)); D = np.array(t["D"]); d = np.array(t["resid_7"]); s = np.array(t["sigma_7"])
        for ax in (a1, a2):
            ax.plot(D, d, color=col, ls=ls, lw=1.4, label=lab)
    # Richardson extrapolation of the CIC 256/384/512 sequence (p fitted at D = 0.2)
    try:
        t = [json.load(open(os.path.join(RES, f"residuals_{f}.json"))) for f in ("conv_pm256m2", "conv_pm384m2", "conv_pm512m2lean")]
        D = np.array(t[0]["D"]); a, b, c = (np.array(x["resid_7"]) for x in t)
        e = int(np.argmin(abs(D - 0.2)))
        pexp = np.log((a[e] - b[e]) / (b[e] - c[e])) / np.log(1.5) if (a[e] - b[e]) * (b[e] - c[e]) > 0 else 2.0
        rich = c - (b - c) / ((512 / 384) ** pexp - 1)
        for ax in (a1, a2):
            ax.plot(D, rich, color=INK, lw=1.8, ls="--", label=f"CIC Richardson limit (p = {pexp:.1f})")
    except Exception as ex:
        print("richardson:", ex)
    a1.set_xlim(0.04, 0.32); a1.set_ylim(-0.005, 0.07)
    a2.set_xscale("log"); a2.set_yscale("symlog", linthresh=1e-3); a2.set_xlim(0.04, 1.05)
    for ax in (a1, a2):
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_xlabel("D")
    a1.set_ylabel(r"$C_2^{\rm PM}-C_2^{\rm 7LPT}$ [$(h^{-1}$Mpc$)^2$]")
    a1.set_title("before/at first crossing (D ≲ 0.3)", loc="left")
    a2.set_title("full range (symlog)", loc="left")
    a1.legend(fontsize=6.8, loc="upper left")
    fig.text(0.01, -0.02, "*PCS + deconvolution + spectral gradient + interlacing + 2³ Fourier sheet resampling", fontsize=7, color=INK2)
    save(fig, name)


# ── Figure 11: IC controls — the pair construction itself ─────────────────────────────────────────
def fig_controls(name="fig11_ic_controls"):
    p = os.path.join(RES, "controls.json")
    if not os.path.exists(p): return
    c = json.load(open(p)); D = np.array(c["epochs"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for key, lab, col in (("gauss", f"Gaussian pair R={c['R_UV']:.0f}/{c['R_IR']:.0f} Mpc/h", RED),
                          ("gauss_mlv", "Gaussian pair + matched variance", MAGENTA),
                          ("sharp", "sharp split (this study)", BLUE), ("null", "null (no band)", MUTED)):
        a1.plot(D, np.abs(np.array(c[key]["C2_1lpt"])) + 1e-6, color=col, lw=1.6, marker="o", ms=3, label=lab)
        a2.plot(D, np.abs(np.array(c[key]["C2_3lpt"])) + 1e-6, color=col, lw=1.6, marker="o", ms=3, label=lab)
    a1.axhline(abs(c["C2_gauss_1lpt_predicted"]), color=INK, lw=0.9, ls=":", label=r"$|R_{\rm UV}^2-R_{\rm IR}^2|/2$")
    for ax, t in ((a1, "1LPT: no dynamics"), (a2, "3LPT")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 300); ax.set_xlabel("D"); ax.set_title(t, loc="left")
    a1.set_ylabel(r"$|C_2|$ [$(h^{-1}$Mpc$)^2$]  (floor $10^{-6}$ = exactly 0)")
    a1.legend(fontsize=7, loc="center right")
    save(fig, name)


# ── Figure 12: variants — which modes, which amplitude; departure vs shell-crossed mass ────────────
def fig_variants(names, name="fig12_band_variants", kmax=0.08):
    rs = [(n, lab, col, Result(n)) for n, lab, col in names if os.path.exists(os.path.join(RES, f"{n}.h5"))]
    if not rs: return
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for n, lab, col, r in rs:
        D = r.D; sel = D >= 0.05
        Cp, sp, *_ = r.C2("disp", "pm", kmax); C3, *_ = r.C2("disp", "lpt3", kmax)
        C5, *_ = r.C2("disp", "lpt5", kmax); C7, *_ = r.C2("disp", "lpt7", kmax)
        lim, _ = shanks(C3, C5, C7)
        errbar(axes[0], D[sel], Cp[sel] / D[sel] ** 2, sp[sel] / D[sel] ** 2, color=col, marker="o", ms=3, lw=1.4, label=f"{lab} PM")
        axes[0].plot(D[sel], C3[sel] / D[sel] ** 2, color=col, lw=0.9, ls="--")
        if f"resid_disp" in r.h:
            g = r.h["resid_disp"]; j = int(np.argmin(abs(g["kmax"][:] - kmax)))
            d7 = g["C2"][j, :, r.mi("lpt7")]; s7 = g["sC2"][j, :, r.mi("lpt7")]
        else:
            d7 = Cp - C7; s7 = sp
        dinf = Cp - lim
        f = r.inv("pm")
        ok = sel & (f > 0)
        errbar(axes[1], f[ok], d7[ok] / np.abs(C3[ok]), s7[ok] / np.abs(C3[ok]), color=col, marker="o", ms=3, lw=1.3, label=f"{lab}: PM − 7LPT")
        axes[1].plot(f[ok], dinf[ok] / np.abs(C3[ok]), color=col, lw=0.9, ls=":", )
        errbar(axes[2], D[sel], d7[sel], s7[sel], color=col, marker="o", ms=3, lw=1.3, label=lab)
    axes[0].set_xscale("log"); axes[0].set_ylim(-4.5, 2.5); axes[0].set_xlabel("D")
    axes[0].set_ylabel(r"$C_2/D^2$ (solid PM, dashed 3LPT)"); axes[0].legend(fontsize=7, loc="lower left")
    axes[1].set_xscale("log"); axes[1].set_xlim(1e-4, 0.5); axes[1].set_ylim(-0.05, 1.0)
    axes[1].set_xlabel("inverted (shell-crossed) mass fraction of the UV member (PM)")
    axes[1].set_ylabel(r"$(C_2^{\rm PM}-C_2^{\rm LPT})/|C_2^{\rm 3LPT}|$")
    axes[1].set_title("dots: vs 7LPT; dotted: vs LPT$_\\infty$", loc="left", fontsize=8.5)
    axes[2].set_xscale("log"); axes[2].set_yscale("symlog", linthresh=1e-3); axes[2].set_xlabel("D")
    axes[2].set_ylabel(r"$C_2^{\rm PM}-C_2^{\rm 7LPT}$")
    for ax in axes: ax.axhline(0, color=MUTED, lw=0.8)
    save(fig, name)


def export_tables(r, kmax=0.08, out="summary"):
    """Compact CSV/JSON of the time × order matrix C2(D, n), PM, LPT∞ and shell-crossed fractions."""
    rows = []; D = r.D
    C = {m: r.C2("disp", m, kmax) for m in r.models}
    lim, rat = shanks(C["lpt3"][0], C["lpt5"][0], C["lpt7"][0])
    hdr = ["D"] + [f"C2_{m}" for m in r.models] + [f"sC2_{m}" for m in r.models] + ["C2_lptinf_shanks", "series_ratio", "inv_uv_pm", "inv_uv_lpt7", "inv_ir_pm"]
    for e in range(len(D)):
        row = [D[e]] + [C[m][0][e] for m in r.models] + [C[m][1][e] for m in r.models] + [lim[e], rat[e],
               r.inv("pm")[e] if "pm" in r.models else np.nan, r.inv("lpt7")[e], r.inv("pm", "ir")[e] if "pm" in r.models else np.nan]
        rows.append(row)
    with open(os.path.join(RES, f"{out}_{r.name}.csv"), "w") as f:
        f.write(f"# C2 [(Mpc/h)^2] from Delta psi_L = -C2 k^2 psi_L, k <= {kmax} h/Mpc; config {r.name}; {r.n_long} long seeds\n")
        f.write(",".join(hdr) + "\n")
        for row in rows:
            f.write(",".join("" if (isinstance(v, float) and np.isnan(v)) else f"{v:.6g}" for v in row) + "\n")
    print("wrote", f"{out}_{r.name}.csv")

if __name__ == "__main__":
    which = os.environ.get("EFTSC_FIG_CONFIG", "fiducial")
    r = Result(which)
    fig_T(r); fig_response_k(r); fig_c2_vs_time(r); fig_kmax(r)
    if "pm" in r.models:
        fig_lpt_convergence(r); fig_pm_vs_lpt_k(r); fig_velocity(r)
        fits = fig_stochastic(r)
        print("stochastic low-k slopes:", fits)
    fig_density_vs_disp(r, sigma_psi_band=4.005)
    fig_pm_convergence(); fig_controls()
    fig_variants([("fiducial", "band 0.2–1.2", PM_COLOR), ("amp07", "×0.7 amplitude", VIOLET),
                  ("kuv06", "band 0.2–0.6", BLUE)])
    export_tables(r)
