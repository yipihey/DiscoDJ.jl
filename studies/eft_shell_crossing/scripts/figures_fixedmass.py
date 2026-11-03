#!/usr/bin/env python3
"""figF1: fixed-mass statistics of the fiducial pair: variance response split into added power and coherent response; F(mu) of PM vs LPT; conditional budget at D = 1."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results"); r = json.load(open(os.path.join(R, "fixedmass_response.json"))); s = json.load(open(os.path.join(R, "fixedmass_summary.json")))
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]; INK = "#52514e"; ML = {0: "PM", 3: "3LPT", 7: "7LPT"}; MC = dict(zip((0, 3, 7), CAT))
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4), gridspec_kw=dict(wspace=0.27), facecolor="white")
for m in (0, 3, 7):
    q = [x for x in r["rows"] if abs(x["R"] - 6.69) < 0.02 and x["model"] == m]
    ax[0].plot([x["D"] for x in q], [-100 * x["response"][0] for x in q], "o-", color=MC[m], lw=2, ms=5, label=ML[m])
    ax[0].fill_between([x["D"] for x in q], [-100 * (x["response"][0] + x["response"][1]) for x in q], [-100 * (x["response"][0] - x["response"][1]) for x in q], color=MC[m], alpha=0.2)
ax[0].set_yscale("log"); ax[0].set_ylim(0.1, 100); ax[0].set_xlabel("D"); ax[0].set_ylabel(r"coherent suppression $-2\langle\Delta\delta_{IR}\rangle/\langle\delta_{IR}^2\rangle$ [%], $R = 6.7$")
Rs = sorted({x["R"] for x in r["rows"]})
for D, col in zip((0.5, 1.0), CAT):
    q = [x for x in r["rows"] if x["D"] == D and x["model"] == 0]
    ax[1].plot([x["R"] for x in q], [x["add_over_linear"] for x in q], "o-", color=col, lw=2, ms=5, label=f"D = {D:g}")
ax[1].axhline(1, color=INK, ls="--", lw=1.2); ax[1].set_xscale("log"); ax[1].set_xlabel(r"$R$ [$h^{-1}$Mpc]"); ax[1].set_ylabel(r"added power $\langle\Delta^2\rangle / D^2\sigma^2_{\rm band}$, PM")
Fq = [x for x in s["F"] if x["D"] == 1.0 and abs(x["R"] - 5.0) < 0.2]
for x in Fq:
    ax[2].plot(x["mu"], np.maximum(x["F_uv"], 1e-6), "o-", color=MC[x["model"]], lw=2, ms=5, label=ML[x["model"]])
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel(r"$\mu_R$, $R = 5\,h^{-1}$Mpc, D = 1"); ax[2].set_ylabel(r"$F(R,\mu)=P(\mu_R\geq\mu)$, UV member")
for a in ax:
    a.grid(True, which="major", color="#e4e3df", lw=0.7); a.set_axisbelow(True); a.tick_params(labelsize=9); a.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
for ext in ("png", "pdf"): fig.savefig(os.path.join(H, "..", "figures", f"figF1_fixedmass.{ext}"), dpi=170, bbox_inches="tight")
print("wrote figF1")
