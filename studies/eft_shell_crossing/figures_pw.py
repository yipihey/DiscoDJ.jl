"""Plane-wave (ε → 0) counterterm figures: c_s(k_split, D), the loop / counterterm ladder, the long-background scan,
the second-order kernel, fit adequacy, and the time-step convergence.

    python3 figures_pw.py [results_dir] [figures_dir]

Inputs: results/pw_prod_*.h5 (scripts/pw_production.jl), results/ab_kernel_prod_ks20.h5 (scripts/ab_kernel.jl),
results/pw_robustness.json (scripts/pw_robustness.jl), results/step_convergence.json (scripts/step_convergence.jl),
results/summary_fiducial.csv (random-long-field ensemble, §6).
Julia-written arrays appear with reversed axes: R[seed, wave, epoch, model], fits[coef, order, epoch, model].
"""
import os, sys, json, csv
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figures as F
from figures import save, errbar, lpt_color, PM_COLOR, BLUE, ORANGE, AQUA, VIOLET, RED, MAGENTA, GREEN, YELLOW, INK, INK2, MUTED

RES, FIG = F.RES, F.FIG
SPLITS = [0.12, 0.15, 0.20, 0.25, 0.33, 0.50, 0.70]
TAG = {0.12: "prod_ks12", 0.15: "prod_ks15", 0.20: "prod_ks20", 0.25: "prod_ks25", 0.33: "prod_ks33", 0.50: "prod_ks50",
       0.70: "prod_ks70"}
MODELS = ["pm", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "pm-lpt7"]
MI = {m: i for i, m in enumerate(MODELS)}


class PW:
    def __init__(self, name):
        with h5py.File(os.path.join(RES, f"pw_{name}.h5"), "r") as h:
            self.R = h["R"][:]                 # (seed, wave, epoch, model[0:8])
            self.k = h["k"][:]; self.D = h["epochs"][:]; self.inv = h["inv"][:]
            self.fits = h["fits"][:]; self.sig = h["fits_sigma"][:]       # (coef, order, epoch, model+1)
            self.ks = float(h.attrs["k_split"]); self.bg = float(h.attrs["background"])

    def C(self, m, coef=0, order=2):           # coefficient series over epochs (C2: coef 0, C4: coef 1)
        i = MI[m]; return self.fits[coef, order - 1, :, i], self.sig[coef, order - 1, :, i]

    def lptinf(self):                          # Shanks limit of the odd partial sums 3, 5, 7 (two-term C2)
        s3, s5, s7 = (self.C(f"lpt{n}")[0] for n in (3, 5, 7))
        d = (s7 - s5) - (s5 - s3)
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = s7 - (s7 - s5) ** 2 / d
        r = (s7 - s5) / (s5 - s3)
        return np.where(np.abs(r) < 0.7, lim, np.nan)


def load():
    return {ks: PW(TAG[ks]) for ks in SPLITS if os.path.exists(os.path.join(RES, f"pw_{TAG[ks]}.h5"))}


def ensemble_pm():
    rows = [r for r in csv.reader(l for l in open(os.path.join(RES, "summary_fiducial.csv")) if not l.startswith("#"))]
    h = rows[0]; D = []; C = []; S = []
    for r in rows[1:]:
        if r[h.index("C2_pm")]:
            D.append(float(r[0])); C.append(float(r[h.index("C2_pm")])); S.append(float(r[h.index("sC2_pm")]))
    return np.array(D), np.array(C), np.array(S)


def fig_cs_time(pw, name="figP1_cs_vs_time"):
    """k_split = 0.2: the ε → 0 coefficient vs D (LPT orders, PM; random-long-field ensemble for comparison) and the
    departure from the pressureless series with the shell-crossed fraction."""
    p = pw[0.20]; bg = {a: PW(f"prod_ks20_bg{t}") for a, t in ((0.5, "05"), (1.0, "10"))}
    D = p.D; sel = D > 0.15 + 1e-9
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for n in (3, 5, 7):
        c, _ = p.C(f"lpt{n}"); a1.plot(D[D >= 0.05], c[D >= 0.05] / D[D >= 0.05] ** 2, color=lpt_color(n), lw=1.3, label=f"{n}LPT")
    a1.plot(D[sel], p.lptinf()[sel] / D[sel] ** 2, color=lpt_color(7), lw=1.0, ls="--", label=r"LPT$_\infty$ (Shanks)")
    c, s = p.C("pm"); errbar(a1, D[sel], c[sel] / D[sel] ** 2, s[sel] / D[sel] ** 2, color=PM_COLOR, marker="o", ms=3.5, lw=1.8,
                             label=r"PM, plane waves ($\varepsilon\to0$)")
    for a, col in ((0.5, YELLOW), (1.0, RED)):
        c, s = bg[a].C("pm"); errbar(a1, D[sel], c[sel] / D[sel] ** 2, s[sel] / D[sel] ** 2, color=col, marker="s", ms=3, lw=1.2,
                                     label=f"PM, waves on long background a = {a:g}")
    De, Ce, Se = ensemble_pm(); m = De > 0.15
    errbar(a1, De[m], Ce[m] / De[m] ** 2, Se[m] / De[m] ** 2, color=INK2, marker="x", ms=4, lw=0, label="PM, random long field (§6)")
    a1.axhline(0, color=MUTED, lw=0.8); a1.set_ylim(-4.2, 2.2)
    a1.set_xlabel(r"$D$"); a1.set_ylabel(r"$C_2/D^2$  [$(h^{-1}$Mpc$)^2$]"); a1.set_title(r"(a) displacement response coefficient, $k_{\rm split}=0.2$", loc="left")
    a1.legend(fontsize=7, loc="lower left")
    for a, col, lab in ((0.0, PM_COLOR, "a = 0 (pure counterterm)"), (0.5, YELLOW, "a = 0.5"), (1.0, RED, "a = 1")):
        q = p if a == 0 else bg[a]
        d, s = q.C("pm-lpt7"); ok = sel & (d > 0)
        errbar(a2, D[ok], d[ok], s[ok], color=col, marker="o", ms=3.5, lw=1.5, label=r"$C_2^{\rm PM}-C_2^{7\rm LPT}$, " + lab)
    f = np.nanmean(p.inv[:, :, MI["pm"]], axis=0)
    a2.plot(D[sel], f[sel], color=MUTED, lw=1.0, ls=":", label="shell-crossed mass fraction (PM)")
    a2.set_yscale("log"); a2.set_ylim(1e-4, 40); a2.set_xlabel(r"$D$")
    a2.set_title("(b) departure from the pressureless series", loc="left"); a2.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, name)


def fig_running(pw, name="figP2_cs_running"):
    """C2 and the departure vs k_split (the CHS cutoff Λ) at several D."""
    ks = np.array(sorted(pw))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    cols = {0.3: AQUA, 0.5: BLUE, 0.7: VIOLET, 1.0: PM_COLOR}
    for D, col in cols.items():
        e = [int(np.argmin(abs(pw[k].D - D))) for k in ks]
        c = np.array([pw[k].C("pm")[0][i] for k, i in zip(ks, e)]); s = np.array([pw[k].C("pm")[1][i] for k, i in zip(ks, e)])
        l3 = np.array([pw[k].C("lpt3")[0][i] for k, i in zip(ks, e)])
        errbar(a1, ks, c, s, color=col, marker="o", ms=4, lw=1.6, label=f"PM, D = {D:g}")
        a1.plot(ks, l3, color=col, lw=0.9, ls="--")
        d = np.array([pw[k].C("pm-lpt7")[0][i] for k, i in zip(ks, e)]); sd = np.array([pw[k].C("pm-lpt7")[1][i] for k, i in zip(ks, e)])
        ok = d > 0; errbar(a2, ks[ok], d[ok], sd[ok], color=col, marker="o", ms=4, lw=1.6, label=f"D = {D:g}")
    a1.plot([], [], color=MUTED, ls="--", lw=0.9, label="3LPT (one loop), same D")
    a1.axhline(0, color=MUTED, lw=0.8); a1.set_xscale("log"); a2.set_xscale("log"); a2.set_yscale("log")
    for ax in (a1, a2):
        ax.set_xticks(SPLITS); ax.set_xticklabels([f"{t:g}" for t in SPLITS]); ax.minorticks_off()
        ax.set_xlabel(r"$k_{\rm split}$  [$h$ Mpc$^{-1}$]  (modes $\geq k_{\rm split}$ integrated out)")
    a1.set_ylabel(r"$C_2$  [$(h^{-1}$Mpc$)^2$]"); a1.set_title(r"(a) $\varepsilon\to0$ coefficient vs cutoff", loc="left")
    a1.legend(fontsize=7, loc="lower right"); a1.set_ylim(-2.2, 0.15)
    a2.set_ylabel(r"$C_2^{\rm PM}-C_2^{7\rm LPT}$  [$(h^{-1}$Mpc$)^2$]"); a2.set_title("(b) departure from the pressureless series", loc="left")
    a2.legend(fontsize=7)
    fig.tight_layout(); save(fig, name)


def fig_ladder(pw, name="figP3_loop_ladder"):
    """k_split = 0.2: every contribution to the k² coefficient, isolated by construction."""
    p = pw[0.20]; b1 = PW("prod_ks20_bg10"); D = p.D
    g = lambda q, m: q.C(m)[0]
    terms = [
        ("one loop (3LPT), short×short", g(p, "lpt3"), lpt_color(3), "-", D >= 0.05),
        ("two loop short×short (5LPT − 3LPT)", g(p, "lpt5") - g(p, "lpt3"), lpt_color(5), "-", D >= 0.05),
        ("≥ three loop short×short (7LPT − 5LPT)", g(p, "lpt7") - g(p, "lpt5"), lpt_color(7), "-", D >= 0.05),
        ("long×short, perturbative (7LPT: a=1 − a=0)", g(b1, "lpt7") - g(p, "lpt7"), VIOLET, "-", D >= 0.05),
        ("beyond PT, short×short (PM − 7LPT, a=0)", g(p, "pm-lpt7"), PM_COLOR, "-", D > 0.15),
        ("beyond PT, long×short (PM − 7LPT: a=1 − a=0)", g(b1, "pm-lpt7") - g(p, "pm-lpt7"), RED, "-", D > 0.15),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for lab, v, col, ls, ok in terms:
        pos = ok & (v > 0); neg = ok & (v < 0)
        ax.plot(D[ok], np.abs(v[ok]), color=col, lw=1.6, label=lab)
        ax.plot(D[pos], np.abs(v[pos]), ls="none", marker="o", ms=3.5, color=col)
        ax.plot(D[neg], np.abs(v[neg]), ls="none", marker="o", ms=3.5, mfc="white", mec=col)
    ax.axhline(3.4e-4, color=MUTED, lw=0.8, ls=":")
    ax.text(0.98, 3.9e-4, "PM numerical floor (100 steps, 7LPT ICs at D = 0.15; §14.2)", color=INK2, fontsize=6.5, ha="right", transform=ax.get_yaxis_transform())
    ax.set_yscale("log"); ax.set_ylim(1e-5, 30); ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$|$contribution to $C_2|$  [$(h^{-1}$Mpc$)^2$]")
    ax.set_title(r"Loop / counterterm ladder of the $k^2$ response, $k_{\rm split}=0.2$  (filled: > 0 pressure-like, open: < 0)", loc="left", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, name)


def fig_background(pw, name="figP4_background_scan"):
    """Response vs the amplitude a of a fixed random long background: a² (two-loop) test."""
    q = {0.0: pw[0.20], 0.5: PW("prod_ks20_bg05"), 1.0: PW("prod_ks20_bg10")}; a = np.array(sorted(q))
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, D in zip(axs, (0.5, 0.7, 1.0)):
        e = int(np.argmin(abs(q[0.0].D - D)))
        for m, col, lab in (("pm", PM_COLOR, "PM"), ("lpt7", lpt_color(7), "7LPT"), ("lpt3", lpt_color(3), "3LPT")):
            c = np.array([q[x].C(m)[0][e] for x in a]); s = np.array([q[x].C(m)[1][e] for x in a])
            errbar(ax, a ** 2, c, s, color=col, marker="o", ms=4, lw=0, label=lab)
            ax.plot([0, 1], [c[0], c[0] + (c[1] - c[0]) / 0.25], color=col, lw=0.9, ls="--")   # a² line through a = 0, 0.5
        De, Ce, Se = ensemble_pm(); i = int(np.argmin(abs(De - D)))
        errbar(ax, [1.0], [Ce[i]], [Se[i]], color=INK2, marker="x", ms=6, lw=0, label="random long field (§6)")
        ax.set_xlabel(r"$a^2$  (long-background power / $\Lambda$CDM)"); ax.set_title(f"D = {D:g}", loc="left")
    axs[0].set_ylabel(r"$C_2$  [$(h^{-1}$Mpc$)^2$]"); axs[0].legend(fontsize=7)
    fig.suptitle("Long×short loops: dashed = a² (two-loop) extrapolation from a = 0, 0.5", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); save(fig, name)


def fig_kernel(name="figP5_second_order_kernel"):
    p = os.path.join(RES, "ab_kernel_prod_ks20.h5")
    if not os.path.exists(p): return
    with h5py.File(p, "r") as h:
        K = h["K"][:]; Kd = h["Kdens"][:]; kt = h["kt"][:]; D = h["epochs"][:]
        models = h.attrs["models"]; models = (models.decode() if isinstance(models, bytes) else models).split(",")
    im = {m: i for i, m in enumerate(models)}; sel = D > 0.15 + 1e-9
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for m, col, lab in (("lpt3", lpt_color(3), "3LPT"), ("lpt5", lpt_color(5), "5LPT"), ("lpt7", lpt_color(7), "7LPT"), ("pm", PM_COLOR, "PM")):
        v = np.nanmean(K[:, :, :, im[m]], axis=(0, 1)) - 1
        ok = sel if m == "pm" else D >= 0.05
        a1.plot(D[ok], v[ok], color=col, lw=1.6, marker="o" if m == "pm" else None, ms=3.5, label=lab)
    dK = K[:, :, :, im["pm"]] - K[:, :, :, im["lpt7"]]            # (seed, pair, epoch)
    pairmean = np.nanmean(dK, axis=0); m_ = np.nanmean(pairmean, axis=0); se = np.nanstd(pairmean, axis=0) / np.sqrt(pairmean.shape[0])
    errbar(a1, D[sel], m_[sel], se[sel], color=RED, marker="s", ms=3.5, lw=1.4, label="PM − 7LPT (± s.e.m. over pairs)")
    dd = np.nanmean(Kd[:, :, :, im["pm"]] - Kd[:, :, :, im["lpt7"]], axis=(0, 1))
    a1.plot(D[sel], dd[sel], color=RED, lw=1.0, ls=":", label="PM − 7LPT, Eulerian density")
    a1.axhline(0, color=MUTED, lw=0.8); a1.set_xlabel(r"$D$"); a1.set_ylabel(r"$\langle K\rangle - 1$  (2LPT tree $\equiv$ 0)")
    a1.set_title(r"(a) second-order kernel at $k_a\pm k_b$, mean over 70 pairs", loc="left"); a1.legend(fontsize=7, loc="lower left")
    for D0, col in ((0.5, BLUE), (0.7, VIOLET), (1.0, PM_COLOR)):
        e = int(np.argmin(abs(D - D0))); y = np.nanmean(dK[:, :, e], axis=0); s = np.nanstd(dK[:, :, e], axis=0) / np.sqrt(dK.shape[0])
        errbar(a2, kt, y, s, color=col, marker="o", ms=3, lw=0, label=f"D = {D0:g}")
    a2.axhline(0, color=MUTED, lw=0.8); a2.set_xlabel(r"$|k_a\pm k_b|$  [$h$ Mpc$^{-1}$]"); a2.set_ylabel(r"$K_{\rm PM}-K_{7\rm LPT}$")
    a2.set_title("(b) per pair (4 realizations)", loc="left"); a2.legend(fontsize=7)
    fig.tight_layout(); save(fig, name)


def fig_waves(pw, name="figP6_per_wave_fits"):
    """Fit adequacy: per-wave −R/k² with the two- and three-term fits (k_split = 0.2 and 0.5)."""
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.4))
    for row, ks in enumerate((0.20, 0.50)):
        p = pw[ks]; o = np.argsort(p.k); k = p.k[o]
        for col, (m, lab) in enumerate((("pm", "PM"), ("pm-lpt7", "PM − 7LPT"))):
            ax = axs[row, col]
            for D0, c in ((0.5, BLUE), (0.7, VIOLET), (1.0, PM_COLOR)):
                e = int(np.argmin(abs(p.D - D0)))
                R = p.R[:, o, e, 0] if m == "pm" else p.R[:, o, e, 0] - p.R[:, o, e, MI["lpt7"]]
                y = -R / k ** 2; errbar(ax, k, y.mean(0), y.std(0) / np.sqrt(y.shape[0]), color=c, marker="o", ms=3, lw=0, label=f"D = {D0:g}")
                kk = np.linspace(0, k.max() * 1.03, 100)
                f2 = p.fits[:, 1, e, MI[m]]; f3 = p.fits[:, 2, e, MI[m]]
                ax.plot(kk, f2[0] - f2[1] * kk ** 2, color=c, lw=1.2)
                ax.plot(kk, f3[0] - f3[1] * kk ** 2 + f3[2] * kk ** 4, color=c, lw=0.9, ls="--")
            ax.set_title(f"k_split = {ks:g}: {lab}  (solid: C₂ + C₄, dashed: + C₆)", loc="left", fontsize=8.5)
            ax.set_xlabel(r"$k$ of the plane wave  [$h$ Mpc$^{-1}$]"); ax.set_ylabel(r"$-R/k^2$  [$(h^{-1}$Mpc$)^2$]")
            if m == "pm-lpt7": ax.set_yscale("symlog", linthresh=0.05)
        axs[row, 0].legend(fontsize=7)
    fig.tight_layout(); save(fig, name)


def fig_steps(name="figP7_step_convergence"):
    p = os.path.join(RES, "step_convergence.json")
    if not os.path.exists(p): return
    d = json.load(open(p)); ref = d["reference"]
    runs = {k: v for k, v in d.items() if isinstance(v, dict) and "rows" in v}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for late, mk, lab in ((False, "o", r"$D_{\rm ini}=0.02$, 3LPT ICs"), (True, "s", r"$D_{\rm ini}=0.15$, 7LPT ICs")):
        for D0, col in ((0.3, AQUA), (0.5, BLUE), (1.0, PM_COLOR)):
            pts = sorted((v["steps"], abs(r["dC2"])) for kname, v in runs.items() if ("di15" in kname) == late
                         for r in v["rows"] if abs(r["D"] - D0) < 1e-6)
            if pts:
                n, y = zip(*pts); a1.plot(n, y, marker=mk, color=col, lw=1.2, ls="-" if late else ":", label=f"D = {D0:g}, " + lab)
        pts = sorted((v["steps"], max(abs(r["dC2"]) for r in v["rows"])) for kname, v in runs.items() if ("di15" in kname) == late)
        if pts:
            n, y = zip(*pts); a2.plot(n, y, marker=mk, color=INK2, lw=1.2, ls="-" if late else ":", label=lab)
    nn = np.array([8, 400]); a2.plot(nn, 0.3 * (nn / 8.0) ** -2, color=MUTED, lw=0.8, ls="--", label=r"$\propto n^{-2}$")
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("PM steps (uniform in D)")
    a1.set_ylabel(r"$|C_2(n)-C_2(400)|$  [$(h^{-1}$Mpc$)^2$]"); a1.legend(fontsize=6.5); a1.set_title("(a) per epoch", loc="left")
    a2.set_ylabel("worst epoch"); a2.legend(fontsize=7); a2.set_title(f"(b) worst over D = 0.2–1 (reference: {ref}, 400 steps)", loc="left", fontsize=8.5)
    fig.tight_layout(); save(fig, name)


if __name__ == "__main__":
    pw = load(); print("splits:", sorted(pw))
    fig_cs_time(pw); fig_running(pw); fig_ladder(pw); fig_background(pw); fig_kernel(); fig_waves(pw); fig_steps()
