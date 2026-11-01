#!/usr/bin/env python3
"""Plane-wave Eulerian-density response with the exact deposit-bias correction -> results/pw_density_summary.json

Inputs (all k^2-only fits on the waves with k <= 0.1, so a bias measured in one model subtracts linearly):
  results/pw_density.json           CIC-deposited density response, all models, 4 realizations, jackknife errors
  results/pw_density_exact_all.json direct-sum density of nLPT at the waves vs its deposit (realization 1), one-loop R1
  results/pw_density_exact_s2.json  same, realization 2 of two bands: realization dependence of the bias
  results/pw_prod_*.h5              displacement response R (seed, wave, epoch, model)

Estimator:  C2d(PM) = C2d_deposit(PM) - b,  b = C2_deposit(7LPT) - C2_exact(7LPT), same D, realization 1.
PM's particles are not stored in the prod_* runs, so b is measured on 7LPT, whose particles equal PM's before crossing.
The systematic quoted is |b(7LPT) - b(3LPT)|, plus the realization difference where a second seed exists.
Signs: R = -C2 k^2.  C2 > 0 suppresses.  dP/P = 2R for the Eulerian density only.
"""
import json
import os
import sys

import h5py
import numpy as np

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
load = lambda f: json.load(open(os.path.join(here, f)))
dep = load("pw_density.json")
ex = load("pw_density_exact_all.json")
ex2 = {}
for f in ("pw_density_exact_s2.json", "pw_density_exact_s2_ks15.json"):
    if os.path.exists(os.path.join(here, f)):
        ex2.update(load(f))
BANDS = ["prod_ks12", "prod_ks15", "prod_ks20", "prod_ks25", "prod_ks33", "prod_ks50", "prod_ks70"]
BANDS = [b for b in BANDS if b in ex]
MODELS = ["pm", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7"]


def exact(name, D, n, src=ex):
    for r in src[name]["rows"]:
        if abs(r["D"] - D) < 1e-9 and r["n"] == n:
            return r
    return None


def row(name, m, D):
    for r in dep[name]["models"][m]:
        if abs(r["D"] - D) < 1e-9:
            return r
    return None


def bias(name, D, n, src=ex):
    r = exact(name, D, n, src)
    return r["C2_deposit"] - r["C2_exact"]


def disp_k2(name, D, m):
    """Displacement C2 on the same waves and fit as the density (k^2-only, k <= 0.1), jackknife over realizations."""
    with h5py.File(os.path.join(here, f"pw_{name}.h5")) as h:
        R = h["R"][()]
        k = h["k"][()]
        E = h["epochs"][()]
    e = int(np.argmin(abs(E - D)))
    lo = k <= 0.1
    x = -k[lo] ** 2
    y = R[:, lo, e, MODELS.index(m)]
    fit = lambda v: float((x @ v) / (x @ x))
    full = fit(y.mean(0))
    n = y.shape[0]
    jk = np.array([fit(np.delete(y, i, 0).mean(0)) for i in range(n)])
    return full, float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))


out = {"bias": [], "oneloop_check": [], "corrected": [], "channels": [], "shanks": [], "background": [], "pk": []}

print("== Exact deposit bias b = C2(deposit) - C2(exact), realization 1, k<=0.1")
print("band       D    b(1LPT)   b(3LPT)   b(7LPT)   | exact 3LPT/D2  7LPT/D2 | b(7LPT) seed 2")
for name in BANDS:
    for D in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        b1, b3, b7 = (bias(name, D, n) for n in (1, 3, 7))
        s2 = bias(name, D, 7, ex2) if name in ex2 else None
        print(f"{name:10s} {D:.2f} {b1:+9.4f} {b3:+9.4f} {b7:+9.4f} | {exact(name, D, 3)['C2_exact'] / D**2:8.3f} {exact(name, D, 7)['C2_exact'] / D**2:8.3f} | "
              + (f"{s2:+.4f}" if s2 is not None else ""))
        out["bias"].append(dict(band=name, D=D, b1=b1, b3=b3, b7=b7, b7_s2=s2))

print("\n== Validation: exact 3LPT C2/D2 -> one-loop (P13, no free parameters) as D -> 0")
print("band       one-loop | D=0.1   0.15    0.2    0.3 | (61/105) sigma2/6 | sigma2/6 realized")
for name in BANDS:
    ol = ex[name]["C2_oneloop"]
    vals = [exact(name, D, 3)["C2_exact"] / D**2 for D in (0.1, 0.15, 0.2, 0.3)]
    kin = np.mean(dep[name]["sigma2_psi_band"]) / 6
    print(f"{name:10s} {ol:7.3f} | " + " ".join(f"{v:6.3f}" for v in vals) + f" | {61 / 105 * kin:6.3f} | {kin:6.3f}")
    out["oneloop_check"].append(dict(band=name, C2_oneloop=ol, exact3=vals, kin=kin))

print("\n== PM density response, bias-corrected (k^2-only, k<=0.1).  C2d = C2d_dep(PM) - b7.  Errors: jackknife, syst")
print("band       D     C2d/D2  +-stat  +-syst  raw/D2 | oneloop  ct=C2d-1loop*D2 | dP/P(k=.05) | C2psi(PM)")
for name in BANDS:
    ol = ex[name]["C2_oneloop"]
    for D in (0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        p = row(name, "pm", D)
        if p is None:
            continue
        b7, b3 = bias(name, D, 7), bias(name, D, 3)
        syst = abs(b7 - b3)
        if name in ex2:
            syst = max(syst, abs(b7 - bias(name, D, 7, ex2)))
        c = p["C2_k2only"] - b7
        stat = p["sC2_k2only"]
        psi, spsi = disp_k2(name, D, "pm")
        print(f"{name:10s} {D:.2f} {c / D**2:7.3f} {stat / D**2:7.3f} {syst / D**2:7.3f} {p['C2_k2only'] / D**2:7.3f} | {ol:6.3f} {c - ol * D**2:+8.4f} | {-200 * c * 0.05**2:+.2f}% | {psi:+7.3f} +- {spsi:.3f}")
        out["corrected"].append(dict(band=name, D=D, C2d=c, stat=stat, syst=syst, raw=p["C2_k2only"], ct=c - ol * D**2, psi=psi, spsi=spsi,
                                     dPoverP_k005=-2 * c * 0.05**2))

ps = load("pw_density_perseed.json")


def cube(name, key, m=None):
    """(epoch, wave, seed) array of a per-seed response; JSON stores [epoch][wave][seed]."""
    v = ps[name][key] if m is None else ps[name][key][m]
    return np.array(v, dtype=float)


def k2fit(name, y):
    """k^2-only fit, k<=0.1, of y (wave, ...) -> C2 (...)."""
    k = np.array(ps[name]["k"])
    lo = k <= 0.1
    x = -k[lo] ** 2
    return np.tensordot(x, y[lo], axes=(0, 0)) / (x @ x)


def jk(f, n):
    full = f(list(range(n)))
    parts = np.array([f([i for i in range(n) if i != j]) for j in range(n)])
    return full, float(np.sqrt((n - 1) / n * ((parts - parts.mean(0)) ** 2).sum(0)))


def eidx(name, D):
    return int(np.argmin(abs(np.array(ps[name]["epochs"]) - D)))


def sh(a, b, c):
    d1, d2 = b - a, c - b
    r = d2 / d1 if d1 != 0 else np.inf
    return (c + d2 * r / (1 - r) if abs(r) < 0.7 else np.nan), r


print("\n== Density, seed-matched PM - LPT (deposit bias largely cancels).  Null test before crossing, then the departure")
print("band       D   | PM-3LPT +-      PM-7LPT +-      | r    PM-LPTinf +-   | syst |b7-b3| | psi: PM-LPTinf +-")
for name in BANDS:
    for D in (0.2, 0.25, 0.3, 0.4, 0.5):
        e = eidx(name, D)
        Y = {m: cube(name, "R", m)[e] for m in ("pm", "lpt3", "lpt5", "lpt7")}
        C = lambda m, idx: k2fit(name, Y[m][:, idx].mean(1))
        n = Y["pm"].shape[1]
        d3 = jk(lambda i: C("pm", i) - C("lpt3", i), n)
        d7 = jk(lambda i: C("pm", i) - C("lpt7", i), n)
        ifull, r = sh(*[C(m, list(range(n))) for m in ("lpt3", "lpt5", "lpt7")])
        dinf = jk(lambda i: C("pm", i) - sh(C("lpt3", i), C("lpt5", i), C("lpt7", i))[0], n) if abs(r) < 0.7 else (np.nan, np.nan)
        pp = [disp_k2(name, D, m)[0] for m in ("lpt3", "lpt5", "lpt7")]
        ip, rp = sh(*pp)
        psi, sp = disp_k2(name, D, "pm")
        bb = abs(bias(name, D, 7) - bias(name, D, 3)) if D != 0.25 else np.nan
        print(f"{name:10s} {D:.2f} | {d3[0]:+8.4f} {d3[1]:.4f} {d7[0]:+8.4f} {d7[1]:.4f} | {r:5.2f} {dinf[0]:+8.4f} {dinf[1]:.4f} | {bb:.4f} | {psi - ip:+8.4f} {sp:.4f}")
        out["shanks"].append(dict(band=name, D=D, pm_lpt3=d3, pm_lpt7=d7, r=float(r), pm_inf=dinf, syst=bb, psi_pm_inf=float(psi - ip), spsi=sp, r_psi=float(rp)))

print("\n== Small-D noise test: exact 3LPT C2/D2 fit c0 + A/D + B D^2 over D=0.1,0.15,0.2,0.3.  A is linear in the band's modes, so it should change with seed")
for name in BANDS:
    Ds = np.array([0.1, 0.15, 0.2, 0.3])
    y = np.array([exact(name, D, 3)["C2_exact"] / D**2 for D in Ds])
    M = np.c_[np.ones(4), 1 / Ds, Ds**2]
    c = np.linalg.lstsq(M, y, rcond=None)[0]
    t = f"{name:10s} one-loop {ex[name]['C2_oneloop']:.3f}  s1: c0 {c[0]:.3f} A {c[1]:+.4f} B {c[2]:+.3f}"
    if name in ex2:
        y2 = np.array([exact(name, D, 3, ex2)["C2_exact"] / D**2 for D in Ds])
        c2 = np.linalg.lstsq(M, y2, rcond=None)[0]
        t += f"  | s2: c0 {c2[0]:.3f} A {c2[1]:+.4f} B {c2[2]:+.3f}"
    print(t)
    out["oneloop_check"].append(dict(band=name, c0=float(c[0]), A=float(c[1]), B=float(c[2])))

print("\n== Long x short, k_s=0.2, relative to the band-free member, clean = Rref_m - Rref_1LPT + R_ZA; seed-matched shift from a=0")
print("D    | C2d a=0 clean   a=.5           a=1            | shift a=.5    a=1 | C2psi shift a=1 (matched)")
for D in (0.5, 0.6, 0.7, 0.8, 1.0):
    Cl = {}
    for n in ("prod_ks20", "prod_ks20_bg05", "prod_ks20_bg10"):
        e = eidx(n, D)
        Cl[n] = cube(n, "Rref", "pm")[e] - cube(n, "Rref", "lpt1")[e] + cube(n, "RZA")[e]
    F = lambda n, i: k2fit(n, Cl[n][:, i].mean(1))
    N = Cl["prod_ks20"].shape[1]
    v = [jk(lambda i, n=n: F(n, i), N) for n in Cl]
    s5 = jk(lambda i: F("prod_ks20_bg05", i) - F("prod_ks20", i), N)
    s1 = jk(lambda i: F("prod_ks20_bg10", i) - F("prod_ks20", i), N)
    pa = [disp_k2(n, D, "pm") for n in Cl]
    print(f"{D:.2f} | {v[0][0]:.3f} +- {v[0][1]:.3f}  {v[1][0]:.3f} +- {v[1][1]:.3f}  {v[2][0]:.3f} +- {v[2][1]:.3f} | {s5[0]:+.3f} +- {s5[1]:.3f}  {s1[0]:+.3f} +- {s1[1]:.3f} | {pa[2][0] - pa[0][0]:+.3f}")
    # cross-check without the Gaussian R_ZA: raw response relative to the band-free member, a=0 corrected by the exact bias
    Raw = {n: cube(n, "Rref", "pm")[eidx(n, D)] for n in Cl}
    Fr = lambda n, i: k2fit(n, Raw[n][:, i].mean(1)) - (bias("prod_ks20", D, 7) if n == "prod_ks20" and D in (0.5, 0.7, 1.0) else 0.0)
    r5 = jk(lambda i: Fr("prod_ks20_bg05", i) - Fr("prod_ks20", i), N)
    r1 = jk(lambda i: Fr("prod_ks20_bg10", i) - Fr("prod_ks20", i), N)
    print(f"     raw estimator: a=0,.5,1 = {Fr('prod_ks20', list(range(N))):.3f} {Fr('prod_ks20_bg05', list(range(N))):.3f} {Fr('prod_ks20_bg10', list(range(N))):.3f}; shift a=.5 {r5[0]:+.3f} +- {r5[1]:.3f}, a=1 {r1[0]:+.3f} +- {r1[1]:.3f}")
    out["background"].append(dict(D=D, raw_shift05=r5, raw_shift10=r1, clean=[x[0] for x in v], s=[x[1] for x in v], shift05=s5, shift10=s1, psi_shift10=pa[2][0] - pa[0][0]))

print("\n== Random-long ensemble of sec.6 (results/fiducial.h5, dens, k^2 fit, PM, IR member subtracted)")
with h5py.File(os.path.join(here, "fiducial.h5")) as h:
    g = h["dens"]["fit1"]
    km = g["kmax"][()]
    j = int(np.argmin(abs(km - 0.08)))
    E = h["epochs"][()]
    for D in (0.3, 0.5, 0.7, 1.0):
        e = int(np.argmin(abs(E - D)))
        print(f"D={D:.1f} kmax={km[j]:.3f}  C2d = {g['C2'][j, e, 7]:.3f} +- {g['sC2'][j, e, 7]:.3f}")
        out.setdefault("ensemble", []).append(dict(D=D, C2d=float(g["C2"][j, e, 7]), s=float(g["sC2"][j, e, 7])))

print("\n== Lagrangian velocity identity (Pi = dPsi/dD): C2v vs d(D C2psi)/dD, PM, k_s=0.2")
E = np.array(ps["prod_ks20"]["epochs"])
kk = np.array(ps["prod_ks20"]["k"])
Cv = np.array([k2fit("prod_ks20", cube("prod_ks20", "Rv", "pm")[e].mean(1)) for e in range(len(E))])
Cp = np.array([disp_k2("prod_ks20", d, "pm")[0] for d in E])
der = np.gradient(E * Cp, E)
for D in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    e = int(np.argmin(abs(E - D)))
    print(f"D={D:.1f} C2v {Cv[e]:+.3f}  d(D C2psi)/dD {der[e]:+.3f}")
    out.setdefault("velocity", []).append(dict(D=D, C2v=float(Cv[e]), deriv=float(der[e])))

print("\n== Counterterm relative to one loop: ct = C2d(PM) - C2_1loop D^2, running with k_s and local time slope")
print("(LCDM z=0, full spectrum: C2 = 1.04 +- 0.07 propagator, 1.13 +- 0.09 P(k); not like-for-like, see README)")
rows = {(r["band"], r["D"]): r for r in out["corrected"]}
print("late-time estimators at D=1, a=0: exact-corrected vs clean (R_ZA Gaussian) | 1LPT exact - R_ZA fit")
for name in BANDS:
    e = eidx(name, 1.0)
    cl = k2fit(name, (cube(name, "R", "pm")[e] - cube(name, "R", "lpt1")[e] + cube(name, "RZA")[e]).mean(1))
    za = k2fit(name, cube(name, "RZA")[e][:, 0])
    print(f"  {name:10s} {rows[(name, 1.0)]['C2d']:.3f} {cl:.3f} | s1: {exact(name, 1.0, 1)['C2_exact'] - za:+.4f} (of {za:.3f})")
for name in BANDS:
    t = [rows[(name, D)]["ct"] for D in (0.4, 0.5, 0.7, 1.0)]
    sl = [np.log(t[i + 1] / t[i]) / np.log(b / a) for i, (a, b) in enumerate(((0.4, 0.5), (0.5, 0.7), (0.7, 1.0)))]
    print(f"{name:10s} ct(D=.4,.5,.7,1) " + " ".join(f"{x:+.4f}" for x in t) + "  slopes " + " ".join(f"{x:.2f}" for x in sl))

with open(os.path.join(here, "pw_density_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote results/pw_density_summary.json", file=sys.stderr)
