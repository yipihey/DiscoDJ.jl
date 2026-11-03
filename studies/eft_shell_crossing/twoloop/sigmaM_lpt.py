#!/usr/bin/env python3
"""How well do nLPT and spherical collapse match PM's sigma(M), and down to what scale? -> results/sigmaM_lpt_<cfg>.json, figures/figM6_*, figM7_*, figM8_*

    SIGMAM_MODEL=lpt<n> python sigmaM_analysis.py <cfg>   (first, for each order, and once for pm)
    python sigmaM_lpt.py <cfg> [figdir]

Statistics: fixed radius sigma(delta_R); fixed mass sigma(ln V) for random centres, mass-drawn centres and Lagrangian patches.
Ratios to PM are seed-matched: ratio of each seed's +- pair means, then mean and delete-one jackknife over seeds.
Reach at tolerance t: the smallest mass on the grid above which every grid point has |ratio - 1| <= t.  "floor" means the grid ran out first:
4 h^-1 Mpc for fixed radius, R0 = 6 for patches, and for random or mass-drawn centres where 2 % of the distribution leaves the radius range.
Spherical collapse values come from sigmaM_sc.py and sc_lowmass.py; PM's errors there are jackknife over seeds.  M = rhobar V(R), Omega_m = 1."""
import json, os, sys
import numpy as np
cfg = sys.argv[1]; figdir = sys.argv[2] if len(sys.argv) > 2 else None
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ORDERS = {"tl_full": [1, 2, 3, 5, 7], "tl_div": [1, 3, 7]}[cfg]
EP = [0.3, 0.5, 0.8, 1.0]; TOL = [0.01, 0.02, 0.05, 0.10]
RHO = 2.775e11; mass = lambda R: 4 * np.pi / 3 * np.asarray(R) ** 3 * RHO
r = json.load(open(os.path.join(here, f"sigmaM_{cfg}.json"))); RE = np.array(r["RE"]); RL = np.array(r["RL"])
sc = json.load(open(os.path.join(here, f"sigmaM_sc_{cfg}.json")))["epochs"]
low = json.load(open(os.path.join(here, f"sc_lowmass_{cfg}.json")))["rows"]
P = np.load(os.path.join(D0, "sigmaM_perseed", f"{cfg}_pm.npz"))
L = {n: np.load(os.path.join(D0, "sigmaM_perseed", f"{cfg}_lpt{n}.npz")) for n in ORDERS}
STATS = {"R": ("s_R", RE, None), "rand": ("s_rand", RL, "t_rand"), "mass": ("s_mass", RL, "t_mass"), "lag": ("s_lag", RL, None)}


def jack(a):
    k = len(a); m = a.mean(0); J = np.array([np.delete(a, i, 0).mean(0) for i in range(k)])
    return m, np.sqrt((k - 1) / k * ((J - J.mean(0)) ** 2).sum(0))


def ep(z, D):
    return int(np.argmin(abs(z["Ds"] - D)))


def ratio(n, stat, D):
    f, _, t = STATS[stat]; A = L[n][f][:, ep(L[n], D)]; B = P[f][:, ep(P, D)]
    if not np.array_equal(L[n]["seeds"], P["seeds"]): raise ValueError("seed mismatch")
    m, e = jack(A / B)
    if t:
        bad = (L[n][t][:, ep(L[n], D)].max(0) > .02) | (P[t][:, ep(P, D)].max(0) > .02)
        m = np.where(bad, np.nan, m); e = np.where(bad, np.nan, e)
    return m, e


def reach(x, y, tol):
    """Smallest mass with every finite grid point above it inside the tolerance; (M, at_floor)."""
    ok = np.isfinite(y)
    if not ok.any(): return None, False
    xx, yy = x[ok], y[ok]; good = abs(yy - 1) <= tol
    if good.all(): return float(xx[0]), True
    i = np.where(~good)[0].max()
    return (float(xx[i + 1]), False) if i + 1 < len(xx) else (None, False)


out = dict(cfg=cfg, RE=RE.tolist(), RL=RL.tolist(), M_RE=mass(RE).tolist(), M_RL=mass(RL).tolist(), tol=TOL, lpt={}, sc={}, ok_lag={})
for n in ORDERS:
    for D in EP:
        if not np.any(abs(L[n]["Ds"] - D) < 1e-6): continue
        for st, (_, grid, _) in STATS.items():
            m, e = ratio(n, st, D)
            out["lpt"][f"{n}|{st}|{D}"] = dict(r=m.tolist(), e=e.tolist(), reach={str(t): reach(mass(grid), m, t) for t in TOL})
        out["ok_lag"][f"{n}|{D}"] = L[n]["ok_lag"][:, ep(L[n], D)].mean(0).tolist()
for D in EP:
    k = f"{D:.3f}"; e = r["epochs"][k]; q = sc[k]
    SC = {"R": (RE, np.array(q["sc_fixRE_d"]), np.array(e["s_R"]), np.array(e["e_s_R"])),
          "R_nocontr": (RE, np.array(q["sc_nocontrE_d"]), np.array(e["s_R"]), np.array(e["e_s_R"])),
          "lnmu": (RE, np.array(q["sc_fixRE_lnmu"]), np.array(q["s_lnmu_RE"]), np.zeros(len(RE))),
          "lag": (RL, np.array(q["sc_lag"]), np.array(e["s_lag"]), np.array(e["e_s_lag"])),
          "rand": (RL, np.where(np.array(e["t_rand"]) > .02, np.nan, q["sc_rand"]), np.array(e["s_rand"]), np.array(e["e_s_rand"]))}
    for st, (g, a, b, er) in SC.items():
        y = a / b
        out["sc"][f"{st}|{D}"] = dict(r=y.tolist(), e=(y * er / b).tolist(), reach={str(t): reach(mass(g), y, t) for t in TOL})
    w = sorted([x for x in low if abs(x["D"] - D) < 1e-6], key=lambda x: x["R0"])
    if w:
        R0 = np.array([x["R0"] for x in w]); y = np.array([x["sc"]["178.0"]["w68"] / x["meas"]["w68"] for x in w])
        out["sc"][f"w68|{D}"] = dict(R0=R0.tolist(), r=y.tolist(), reach={str(t): reach(mass(R0), y, t) for t in TOL})
json.dump(out, open(os.path.join(here, f"sigmaM_lpt_{cfg}.json"), "w"), indent=0, default=float)

fm = lambda v: "none" if v[0] is None else (("<" if v[1] else "") + f"{v[0]:.1e}")
print(f"== {cfg}: reach in M (tolerance 1 %, 5 %, 10 %); '<' = good to the grid floor")
for st in STATS:
    for D in EP:
        cells = [f"{n}LPT " + "/".join(fm(out["lpt"][f"{n}|{st}|{D}"]["reach"][str(t)]) for t in (0.01, 0.05, 0.1)) for n in ORDERS]
        s = {"R": "R", "rand": "rand", "lag": "lag"}.get(st)
        if s: cells.append("SC " + "/".join(fm(out["sc"][f"{s}|{D}"]["reach"][str(t)]) for t in (0.01, 0.05, 0.1)))
        print(st, D, " | ".join(cells))
for D in EP:
    if f"w68|{D}" in out["sc"]: print("w68", D, "SC " + "/".join(fm(out["sc"][f"w68|{D}"]["reach"][str(t)]) for t in (0.01, 0.05, 0.1)), "(table only to R0 = 20.7)")
print("\nRatios to PM at selected R (R0 for fixed-mass statistics)")
for st, (_, grid, _) in STATS.items():
    idx = [int(np.argmin(abs(grid - x))) for x in ((5, 6, 10, 15, 20, 30, 50) if st == "R" else (6, 10, 15, 20, 30, 50))]
    print(st, "R:", " ".join(f"{grid[i]:6.1f}" for i in idx))
    for D in EP:
        for n in ORDERS:
            a = out["lpt"][f"{n}|{st}|{D}"]; print(f"  D={D} {n}LPT", " ".join(f"{a['r'][i]:6.3f}" for i in idx), " max e", f"{np.nanmax(np.array(a['e'])[idx]):.4f}")
print("ok_lag (uninverted fraction of patches) at R0 =", [round(float(RL[i]), 1) for i in (0, 4, 8)], {k: [round(v[i], 4) for i in (0, 4, 8)] for k, v in out["ok_lag"].items() if k.endswith("|1.0")})

if figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUT, GRID = "#1b1b1a", "#52514e", "#e4e3df"
    COL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 5: "#eda100", 7: "#e87ba4"}; SCC = MUT
    TIT = {"R": r"fixed radius, $\sigma(\delta_R)$", "rand": r"random centres, $\sigma(\ln V)$", "mass": r"mass-drawn centres, $\sigma(\ln V)$", "lag": r"Lagrangian patch, $\sigma(\ln V)$"}
    def style(a):
        a.set_xscale("log"); a.grid(True, which="major", color=GRID, lw=0.7); a.set_axisbelow(True); a.tick_params(labelsize=8.5, color="#8a8a85")
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    def save(fig, name):
        for ext in ("png", "pdf"): fig.savefig(os.path.join(figdir, f"{name}_{cfg}.{ext}"), dpi=170, bbox_inches="tight")
    # figM6: nLPT / PM
    fig, ax = plt.subplots(4, 4, figsize=(12.6, 10), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.09, wspace=0.05), facecolor="white")
    for i, st in enumerate(STATS):
        for j, D in enumerate(EP):
            a = ax[i, j]; style(a); a.set_yscale("log"); a.axhline(1, color=INK, lw=1.2); a.axhspan(.95, 1.05, color=GRID, alpha=.6, lw=0)
            for n in ORDERS:
                a.plot(mass(STATS[st][1]), np.clip(out["lpt"][f"{n}|{st}|{D}"]["r"], .3, 3), color=COL[n], lw=2, label=f"{n}LPT")
            if i == 0: a.set_title(f"D = {D:g}", fontsize=11, loc="left", color=INK)
            if j == 0: a.set_ylabel(TIT[st] + "\nnLPT / PM", fontsize=9)
            if i == 3: a.set_xlabel(r"$M$ [$h^{-1}M_\odot$]", fontsize=10)
    ax[0, 0].set_ylim(.4, 2.5); ax[0, 0].set_xlim(1e14, 6e17); ax[0, 0].set_yticks([.5, .7, 1, 1.5, 2]); ax[0, 0].set_yticklabels(["0.5", "0.7", "1", "1.5", "2"]); ax[0, 0].minorticks_off()
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=len(ORDERS), frameon=False, fontsize=10, bbox_to_anchor=(.5, .95)); save(fig, "figM6_sigma_lpt"); plt.close(fig)
    # figM7: spherical collapse / PM
    SL = {"R": ("fixed radius, $\\sigma(\\delta_R)$, contraction", "#2a78d6", "-"), "R_nocontr": ("the same, no contraction", "#2a78d6", ":"), "lnmu": ("fixed radius, $\\sigma(\\ln\\mu_R)$, contraction", "#eda100", "-"),
          "lag": ("patch, std of $\\ln V$", "#eb6834", "-"), "w68": ("patch, core width of $\\ln V$", "#eb6834", "--"), "rand": ("random centres, $\\sigma(\\ln V)$", "#1baf7a", "-")}
    fig, ax = plt.subplots(1, 4, figsize=(12.6, 3.5), sharey=True, gridspec_kw=dict(wspace=0.05), facecolor="white")
    for j, D in enumerate(EP):
        a = ax[j]; style(a); a.axhline(1, color=INK, lw=1.2); a.axhspan(.95, 1.05, color=GRID, alpha=.6, lw=0)
        for st, (lab, c, ls) in SL.items():
            q = out["sc"].get(f"{st}|{D}")
            if not q: continue
            g = np.array(q["R0"]) if st == "w68" else (RE if st in ("R", "R_nocontr", "lnmu") else RL)
            a.plot(mass(g), np.clip(q["r"], .6, 1.8), color=c, ls=ls, lw=2, label=lab)
        a.set_title(f"D = {D:g}", fontsize=11, loc="left", color=INK); a.set_xlabel(r"$M$ [$h^{-1}M_\odot$]", fontsize=10); a.set_xlim(1e14, 6e17)
    ax[0].set_ylim(.75, 1.7); ax[0].set_ylabel("spherical collapse / PM", fontsize=10)
    h, l = ax[-1].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, 1.16)); save(fig, "figM7_sigma_sc_reach"); plt.close(fig)
    # figM8: reach at 5 % against D
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True, gridspec_kw=dict(wspace=0.06), facecolor="white")
    for a, st, tt in ((ax[0], "R", TIT["R"]), (ax[1], "lag", TIT["lag"])):
        a.set_yscale("log"); a.grid(True, which="major", color=GRID, lw=0.7); a.set_axisbelow(True); a.tick_params(labelsize=8.5, color="#8a8a85")
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for n in ORDERS:
            y = [out["lpt"][f"{n}|{st}|{D}"]["reach"]["0.05"] for D in EP]
            a.plot(EP, [v[0] if v[0] else np.nan for v in y], color=COL[n], lw=2, marker="o", ms=8, mec="white", mew=1.5, label=f"{n}LPT")
        y = [out["sc"][f"{st}|{D}"]["reach"]["0.05"] for D in EP]
        a.plot(EP, [v[0] if v[0] else np.nan for v in y], color=SCC, lw=2, ls="--", marker="s", ms=8, mec="white", mew=1.5, label="spherical collapse")
        a.axhline(mass(STATS[st][1])[0], color="#8a8a85", lw=1, ls=":"); a.set_title(tt, fontsize=11, loc="left", color=INK); a.set_xlabel("D", fontsize=10)
    ax[0].set_ylabel(r"smallest $M$ matched to 5 % [$h^{-1}M_\odot$]", fontsize=10); ax[0].legend(frameon=False, fontsize=9); save(fig, "figM8_sigma_reach"); plt.close(fig)
    print("wrote figures")
