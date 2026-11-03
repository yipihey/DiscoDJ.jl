#!/usr/bin/env python3
"""Is the measured stress local in time?  Pass the measured c(D) through the EdS Green's function and compare with the density and momentum responses -> results/time_local_<cfg>.json

    y'' + (3/2D) y' - (3/2D^2) y = D c(D),    y(D) = (2/5) Int_0^D [D - D'^(5/2) D^(-3/2)] D' c(D') dD'
c_density(D) = 9 y/D^3, c_momentum(D) = 3 y'/D^2: a constant c gives c back, so these are the channels' fixed-epoch coefficients for a stress that is local in time and linear.
c(D) is linear between measured epochs, constant below the first (D = 0.1), and known only up to the last epoch above the n_Lambda floor.  Density: direct C2 k<=0.08 is compared as 9 C2/D^2."""
import json, os, sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
out = {}
print("cfg Lambda | D | c stress | predicted density, measured fixed-epoch density, direct 9C2/D^2 | predicted momentum, measured momentum.  All /I, PM")
for c in ("tl_full", "tl_uv10", "tl_uv06", "tl_div", "tl_div10"):
    for lam in ("0.1", "0.15"):
        r = json.load(open(os.path.join(R, f"tl_closure_{c}_L{lam}.json"))); I = r["I"]; fe = r["fixed_epoch"]
        Ds = np.array([x["D"] for x in fe]); st = np.array([x["stress"][0] for x in fe]); ok = np.isfinite(st)
        Dm = Ds[ok]; cm = st[ok] * I
        grid = np.linspace(1e-3, Dm[-1], 4000); cg = np.interp(grid, Dm, cm); S = grid * cg
        def y(D):
            m = grid <= D; g = grid[m]
            return 2 / 5 * np.trapz((D - g ** 2.5 * D ** -1.5) * S[m], g)
        def yp(D):
            m = grid <= D; g = grid[m]
            return 2 / 5 * np.trapz((1 + 1.5 * g ** 2.5 * D ** -2.5) * S[m], g)
        rows = []
        for x, D in zip(fe, Ds):
            if D > Dm[-1]: continue
            pd = 9 * y(D) / D ** 3 / I; pm = 3 * yp(D) / D ** 2 / I
            dc = next((q for q in r["direct_C2"] if q["D"] == D), None); dir_ = 9 * dc["k08"][0] / D ** 2 / I if dc else np.nan
            rows.append(dict(D=float(D), stress=x["stress"][0], pred_dens=pd, dens=x["dens"][0], e_dens=x["dens"][1], direct=dir_, pred_mom=pm, mom=x["mom"][0], e_mom=x["mom"][1]))
            print(f"{c:9s} {lam:5s} | {D:.1f} | {x['stress'][0]:.3f} | {pd:.3f} {x['dens'][0]:.3f}±{x['dens'][1]:.3f} {dir_:.3f} | {pm:.3f} {x['mom'][0]:.3f}±{x['mom'][1]:.3f}")
        out[f"{c}|{lam}"] = dict(I=I, rows=rows)
json.dump(out, open(os.path.join(R, "time_local_closure.json"), "w"), indent=0, default=float)
