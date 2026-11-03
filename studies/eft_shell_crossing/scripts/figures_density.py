#!/usr/bin/env python3
"""figP8: Eulerian density response, corrected (README 14.11).  Reads results/pw_density_summary.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); r = json.load(open(os.path.join(H, "..", "results", "pw_density_summary.json")))
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK = "#52514e"
SP = {"prod_ks12": 0.12, "prod_ks20": 0.20, "prod_ks33": 0.33, "prod_ks70": 0.70}
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4), gridspec_kw=dict(wspace=0.26), facecolor="white")
for (b, ks), col in zip(SP.items(), CAT):
    q = [x for x in r["corrected"] if x["band"] == b]; D = np.array([x["D"] for x in q])
    ax[0].plot(D, [x["C2d"] / x["D"] ** 2 for x in q], "o-", color=col, lw=2, ms=5, label=rf"$k_{{\rm split}} = {ks:g}$")
    ax[0].axhline(q[0]["oneloop"], color=col, lw=1.2, ls="--")
    ax[0].plot(D, [x["raw"] / x["D"] ** 2 for x in q], color=col, lw=1.2, ls=":")
ax[0].set_ylim(0, 3.6); ax[0].set_xlabel("D"); ax[0].set_ylabel(r"$C_{2,\delta}/D^2$ [$h^{-2}$Mpc$^2$]")
ks = [0.12, 0.15, 0.20, 0.25, 0.33, 0.50, 0.70]; names = ["prod_ks12", "prod_ks15", "prod_ks20", "prod_ks25", "prod_ks33", "prod_ks50", "prod_ks70"]
g = lambda b, k: next(x[k] for x in r["corrected"] if x["band"] == b and x["D"] == 1.0)
ax[1].plot(ks, [g(b, "C2d") for b in names], "o-", color=CAT[0], lw=2, ms=5, label=r"density $C_{2,\delta}$")
ax[1].plot(ks, [g(b, "oneloop") for b in names], color=CAT[0], lw=1.4, ls="--", label="one loop (P13)")
ax[1].plot(ks, [g(b, "psi") for b in names], "o-", color=CAT[1], lw=2, ms=5, label=r"displacement $C_{2}$")
ax[1].axhline(0, color=INK, lw=0.8); ax[1].set_xscale("log"); ax[1].set_xlabel(r"$k_{\rm split}$ [$h$ Mpc$^{-1}$]"); ax[1].set_ylabel(r"$C_2$ at D = 1 [$h^{-2}$Mpc$^2$]")
sh = [x for x in r["shells"] if x["D"] == 1.0]; lab = [f"{x['lo']:g}-{x['hi']:g}" for x in sh]
ax[2].plot(range(len(sh)), [x["C2d"] / x["oneloop"] for x in sh], "o-", color=CAT[0], lw=2, ms=5, label="D = 1")
sh5 = [x for x in r["shells"] if x["D"] == 0.5]; ax[2].plot(range(len(sh5)), [x["C2d"] / x["oneloop"] for x in sh5], "o-", color=CAT[1], lw=2, ms=5, label="D = 0.5")
ax[2].axhline(1, color=INK, ls="--", lw=1.2); ax[2].set_xticks(range(len(sh))); ax[2].set_xticklabels(lab, rotation=40, ha="right", fontsize=8); ax[2].set_xlabel(r"shell [$h$ Mpc$^{-1}$]"); ax[2].set_ylabel("density shell / one loop")
for a in ax:
    a.grid(True, which="major", color="#e4e3df", lw=0.7); a.set_axisbelow(True); a.tick_params(labelsize=9); a.legend(frameon=False, fontsize=8.5)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
for ext in ("png", "pdf"): fig.savefig(os.path.join(H, "..", "figures", f"figP8_density_response.{ext}"), dpi=170, bbox_inches="tight")
print("wrote figP8")
