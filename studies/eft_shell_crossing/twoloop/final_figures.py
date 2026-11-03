#!/usr/bin/env python3
"""Report figures from committed results: figT1 running, figT2 stress against D, figT3 time-local closure, figU1 UV response at fixed radius, figM5b no-wiggle k(R)."""
import json, os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); F = os.path.join(HERE, "..", "figures")
L = lambda f: json.load(open(os.path.join(R, f)))
INK = "#52514e"; CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # dataviz default order, fixed per entity


def style(a):
    a.grid(True, which="major", color="#e4e3df", lw=0.7); a.set_axisbelow(True); a.tick_params(labelsize=9)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), dpi=170, bbox_inches="tight")
    print("wrote", name)


SPECTRA = ["tl_full", "tl_uv10", "tl_uv06", "tl_div", "tl_div10"]; LAB = dict(zip(SPECTRA, ("k < 1.6", "k < 1.0", "k < 0.6", "divergent, k < 1.6", "divergent, k < 1.0")))
COL = dict(zip(SPECTRA, CAT))

# figT1: running of the linear stress coefficient at D = 0.1
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4), gridspec_kw=dict(wspace=0.25), facecolor="white")
for c in SPECTRA:
    r = L(f"tl_fit3_{c}_minus_IR.json"); I = []; s = []; d = []
    for lam in ("0.100", "0.150", "0.200", "0.300"):
        x = r[f"pm|L{lam}"]; p = x["per_epoch"].get("0.100")
        if p: I.append(x["I"]); s.append(p["stress"]["sum"][0]); d.append(p["dens"]["sum"][0])
    ax[0].plot(I, s, "o-", color=COL[c], lw=2, ms=5, label=LAB[c]); ax[1].plot(I, d, "o-", color=COL[c], lw=2, ms=5)
for a, t in zip(ax, ("stress", "density")):
    x = np.array([0, 48]); a.plot(x, 61 / 70 * x, color=INK, lw=1.4, ls="--", label=r"$\frac{61}{70}I$"); a.set_xlabel(r"$I(\Lambda)$ [$h^{-2}$Mpc$^2$]"); a.set_ylabel(rf"$\bar c_{{\rm lin}}$, {t}"); style(a)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.03)); save(fig, "figT1_running")

# figT2: stress coefficient against D, Lambda = 0.1, PM vs LPT
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, gridspec_kw=dict(wspace=0.05), facecolor="white")
MC = {"pm": CAT[0], "lpt3": CAT[1], "lpt5": CAT[2], "lpt7": CAT[3]}
for a, c in zip(ax, ("tl_full", "tl_div")):
    r = L(f"tl_fit3_{c}_minus_IR.json")
    for m in MC:
        x = r[f"{m}|L0.100"]; D = [float(k) for k in x["per_epoch"] if np.isfinite(x["per_epoch"][k]["stress"]["sum"][0])]; y = [x["per_epoch"][f"{d:.3f}"]["stress"]["sum"][0] / x["I"] for d in D]
        a.plot(D, y, "o-" if m == "pm" else "-", color=MC[m], lw=2, ms=5, label=m.upper().replace("LPT", "LPT ") if m != "pm" else "PM")
    a.axhline(61 / 70, color=INK, ls="--", lw=1.3, label=r"$61/70$"); a.set_ylim(0.6, 2.2); a.set_xlabel("D"); a.set_title(LAB[c] + r", $\Lambda = 0.1$", fontsize=10, loc="left"); style(a)
ax[0].set_ylabel(r"$\bar c_{\rm lin}/I$, stress"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.05)); save(fig, "figT2_stress_time")

# figT3: is the stress local in time?
t = L("time_local_closure.json"); fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True, gridspec_kw=dict(wspace=0.05), facecolor="white")
for a, key in zip(ax, ("tl_full|0.15", "tl_uv06|0.15", "tl_div|0.15")):
    rows = t[key]["rows"]; D = [q["D"] for q in rows]
    a.plot(D, [q["stress"] for q in rows], "o-", color=CAT[3], lw=2, ms=4, label="measured stress")
    a.plot(D, [q["direct"] for q in rows], "o-", color=CAT[0], lw=2, ms=4, label="measured density response")
    a.plot(D, [q["pred_dens"] for q in rows], color=CAT[0], lw=1.6, ls=(0, (4, 2)), label="density from time-local stress")
    a.plot(D, [q["mom"] for q in rows], "o-", color=CAT[2], lw=2, ms=4, label="measured momentum response")
    a.plot(D, [q["pred_mom"] for q in rows], color=CAT[2], lw=1.6, ls=(0, (4, 2)), label="momentum from time-local stress")
    a.set_xlabel("D"); a.set_title(LAB[key.split("|")[0]] + r", $\Lambda = 0.15$", fontsize=10, loc="left"); style(a)
ax[0].set_ylabel(r"$\bar c/I$"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.05)); save(fig, "figT3_time_local")

# figU1: UV response at fixed radius
PAIRS = [("tl_full", "tl_ir020", "add k > 0.2"), ("tl_div", "tl_full", "raise power above k = 0.3"), ("tl_full", "tl_uv06", "add k = 0.6-1.6"), ("tl_div", "tl_div10", "divergent, add k = 1.0-1.6"), ("tl_full", "tl_uv10", "add k = 1.0-1.6")]
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw=dict(wspace=0.25), facecolor="white")
for (a_, b_, lab), col in zip(PAIRS, CAT):
    r = L(f"sigmaM_pair_{a_}_{b_}.json")["epochs"]; e = r["1.000"]
    ax[0].plot([q["R"] for q in e if q["R"] >= 9.5], [-100 * q["response"] for q in e if q["R"] >= 9.5], "o-", color=col, lw=2, ms=4, label=lab)
    Ds = sorted(float(k) for k in r if any(abs(q["R"] - 20.2) < 1.5 for q in r[k])); y = [-100 * next(q["response"] for q in r[f"{d:.3f}"] if abs(q["R"] - 20.2) < 1.5) for d in Ds]
    ax[1].plot(Ds, y, "o-", color=col, lw=2, ms=4)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"$R$ [$h^{-1}$Mpc]"); ax[0].set_ylabel("variance suppression at D = 1 [%]")
ax[0].set_xticks([10, 20, 30, 50, 80]); ax[0].set_xticklabels(["10", "20", "30", "50", "80"]); ax[0].minorticks_off()
ax[1].set_yscale("log"); ax[1].set_xlabel("D"); ax[1].set_ylabel(r"variance suppression at $R = 20\,h^{-1}$Mpc [%]")
for a in ax: style(a)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04)); save(fig, "figU1_uv_response")
