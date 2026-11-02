#!/usr/bin/env python3
"""Plane-wave Eulerian-density response, fully corrected -> results/pw_density_summary.json (README 14.11)

Inputs
  pw_density_perseed.json   deposited density response, per realization (epoch, wave, seed), waves sorted by |k|
  pw_density_exact_all.json direct-sum vs deposited density of nLPT, realization 1, and the one-loop R1 at the waves
  pw_density_exact_s2.json  the same, realization 2 of three bands
  pw_density_waves.json     direct-sum density of the waves alone (finite-epsilon smearing)
  pw_prod_*.h5              displacement response (seed, wave, epoch, model), waves in config order

Estimator at a = 0, per wave:  R = R_deposit(PM) - [R_deposit(7LPT) - R_exact(7LPT)] - R_waves(7LPT).
  second term: CIC deposit bias, measured on 7LPT, realization 1 (PM's particles are not stored in these runs);
  third term: the waves smear one another at finite epsilon, which the +/- estimator keeps because it is odd in epsilon.
PM - nLPT on the same realization needs neither correction to first order.
Fit: R = -C2 k^2 on k <= 0.1, excluding waves j reached by wave x band coupling (k_j - s k_m in the band for any wave m,
s = +-1, including 2k_j); that term is linear in the band, so it is realization noise proportional to D.
Errors: delete-one jackknife over the 4 realizations.  dP/P = 2R applies to the density only.
Run with a Python that has h5py and numpy.
"""
import json
import os

import h5py
import numpy as np

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
load = lambda f: json.load(open(os.path.join(here, f)))
ps = load("pw_density_perseed.json")
ex = load("pw_density_exact_all.json")
ex2 = load("pw_density_exact_s2.json")
wv = load("pw_density_waves.json")
BANDS = [b for b in ["prod_ks12", "prod_ks15", "prod_ks20", "prod_ks25", "prod_ks33", "prod_ks50", "prod_ks70"] if b in wv]
SPLIT = dict(zip(["prod_ks12", "prod_ks15", "prod_ks20", "prod_ks25", "prod_ks33", "prod_ks50", "prod_ks70"], (0.12, 0.15, 0.20, 0.25, 0.33, 0.50, 0.70)))
KUV = 1.2
KF = 2 * np.pi / 500.0
PM_MODELS = ["pm", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7"]


def disp_data(name):
    with h5py.File(os.path.join(here, f"pw_{name}.h5")) as h:
        return h["R"][()], h["k"][()], h["n"][()], h["epochs"][()]


def coupled(name):
    """Mask (config order) of waves reached by wave x band coupling."""
    _, k, n, _ = disp_data(name)
    kv = KF * n.astype(float)
    bad = np.zeros(len(k), bool)
    for j in range(len(k)):
        for m in range(len(k)):
            for s in (1, -1):
                q = np.linalg.norm(kv[j] - s * kv[m])
                if q > 1e-9 and SPLIT.get(name, 0.2) <= q < KUV:
                    bad[j] = True
    return bad, k


def sorted_keep(name, kmax=0.1, drop=True):
    bad, kc = coupled(name)
    k = np.array(ps[name]["k"])
    b = np.array([bad[np.argmin(abs(kc - q))] for q in k])
    return (k <= kmax) & (~b if drop else True), k


def fit(y, k, keep):
    """-C2 k^2 on the kept waves; y has the wave axis first."""
    x = -k[keep] ** 2
    return np.tensordot(x, y[keep], axes=(0, 0)) / (x @ x)


def cube(name, key, m=None):
    v = ps[name][key] if m is None else ps[name][key][m]
    return np.array(v, dtype=float)


def eidx(name, D):
    return int(np.argmin(abs(np.array(ps[name]["epochs"]) - D)))


def exact_wave(name, D, n, field, src=ex):
    """Per-wave R of the exact file, sorted by |k| (that file stores the config order)."""
    for r in src[name]["rows"]:
        if abs(r["D"] - D) < 1e-9 and r["n"] == n:
            o = np.argsort(r["k"])
            return np.array(r[field])[o]
    return None


def waves_R(name, D, n=7):
    w = wv[name]
    return np.array(w["n"][str(n)][int(np.argmin(abs(np.array(w["epochs"]) - D)))])


def bias_w(name, D, n=7, src=ex):
    return exact_wave(name, D, n, "R_deposit", src) - exact_wave(name, D, n, "R_exact", src)


def jk(f, n=4):
    full = f(list(range(n)))
    parts = np.array([f([i for i in range(n) if i != j]) for j in range(n)])
    return float(full), float(np.sqrt((n - 1) / n * ((parts - parts.mean()) ** 2).sum()))


def sh(a, b, c):
    d1, d2 = b - a, c - b
    r = d2 / d1 if d1 != 0 else np.inf
    return (c + d2 * r / (1 - r) if abs(r) < 0.7 else np.nan), r


def oneloop(name, keep):
    o = np.argsort(ex[name]["k"])
    return float(fit(np.array(ex[name]["R1_oneloop"])[o], np.array(ex[name]["k"])[o], keep))


def disp_seed(name, D, m, keep_kmax=0.1):
    R, k, _, E = disp_data(name)
    bad, _ = coupled(name)
    e = int(np.argmin(abs(E - D)))
    keep = (k <= keep_kmax) & ~bad
    return fit(R[:, :, e, PM_MODELS.index(m)].T, k, keep)  # one C2 per seed


out = {}
print("== Waves dropped for wave x band coupling (k <= 0.1 kept)")
for name in BANDS:
    keep, k = sorted_keep(name)
    kept_all, _ = sorted_keep(name, drop=False)
    print(f"{name}: dropped k = {np.round(k[kept_all & ~keep], 3).tolist()}, kept {keep.sum()} waves up to k = {k[keep].max():.3f}")

print("\n== Waves alone: fitted smearing C2/D2 (7LPT direct sum) | closed form incl. self term | 1LPT exact - R_ZA - waves (null), D=1 and D=.5")
for name in BANDS:
    keep, k = sorted_keep(name)
    _, kc, n, _ = disp_data(name)
    kv = KF * n.astype(float)
    km = np.linalg.norm(kv, axis=1)
    hat = kv / km[:, None]
    e2 = wv[name]["amp"] ** 2
    Rc = np.array([-e2 * (sum((kv[j] @ hat[m]) ** 2 / (4 * km[m] ** 2) for m in range(len(km)) if m != j) + 1 / 8) for j in range(len(km))])
    Rcs = Rc[np.argsort(km)]
    w1 = fit(waves_R(name, 1.0), k, keep)
    null = []
    for D in (1.0, 0.5):
        rza = cube(name, "RZA")[eidx(name, D)][:, 0]
        null.append(float(fit(exact_wave(name, D, 1, "R_exact") - waves_R(name, D, 1) - rza, k, keep)))
    print(f"{name} {w1:.4f} | {fit(Rcs, k, keep):.4f} | {null[0]:+.4f} {null[1]:+.4f}")
    out.setdefault("waves", []).append(dict(band=name, C2_per_D2=float(w1), closed=float(fit(Rcs, k, keep)), null_1lpt_D1=null[0], null_1lpt_D05=null[1]))

print("\n== Validation: exact 3LPT, waves removed, C2/D2 = c0 + A/D + B D^2 over D=.1,.15,.2,.3, vs one loop on the same waves")
print("band      one-loop | D=.1 D=.15 D=.2 D=.3 | c0 (s1) A B | c0 (s2) A | c0/1-loop - 1")
for name in BANDS:
    keep, k = sorted_keep(name)
    ol = oneloop(name, keep)
    Ds = np.array([0.1, 0.15, 0.2, 0.3])
    M = np.c_[np.ones(4), 1 / Ds, Ds**2]
    val = lambda src: np.array([fit(exact_wave(name, D, 3, "R_exact", src) - waves_R(name, D), k, keep) / D**2 for D in Ds])
    y = val(ex)
    c = np.linalg.lstsq(M, y, rcond=None)[0]
    t = f"{name} {ol:7.3f} | " + " ".join(f"{v:.3f}" for v in y) + f" | {c[0]:.3f} {c[1]:+.4f} {c[2]:+.3f}"
    rec = dict(band=name, oneloop=ol, exact3=y.tolist(), c0=float(c[0]), A=float(c[1]), B=float(c[2]))
    if name in ex2:
        c2 = np.linalg.lstsq(M, val(ex2), rcond=None)[0]
        t += f" | {c2[0]:.3f} {c2[1]:+.4f}"
        rec.update(c0_s2=float(c2[0]), A_s2=float(c2[1]))
    print(t + f" | {c[0] / ol - 1:+.3f}")
    out.setdefault("oneloop_check", []).append(rec)

print("\n== Deposit bias on the kept waves: b(7LPT), b(3LPT), b(7LPT, realization 2)")
for name in BANDS:
    keep, k = sorted_keep(name)
    for D in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        b7, b3 = float(fit(bias_w(name, D), k, keep)), float(fit(bias_w(name, D, 3), k, keep))
        s2 = float(fit(bias_w(name, D, 7, ex2), k, keep)) if name in ex2 else None
        print(f"{name} {D:.2f} {b7:+.4f} {b3:+.4f} " + (f"{s2:+.4f}" if s2 is not None else ""))
        out.setdefault("bias", []).append(dict(band=name, D=D, b7=b7, b3=b3, b7_s2=s2))

print("\n== Table 6: PM density, corrected.  C2d/D2 +-stat +-syst | raw/D2 | one-loop | C2d-1loop*D2 | PM-3LPT matched | dP/P k=.05 | C2psi")
for name in BANDS:
    keep, k = sorted_keep(name)
    ol = oneloop(name, keep)
    for D in (0.3, 0.4, 0.5, 0.7, 1.0):
        e = eidx(name, D)
        Rp = cube(name, "R", "pm")[e]
        bw, ww = bias_w(name, D), waves_R(name, D)
        C = lambda i: fit(Rp[:, i].mean(1) - bw - ww, k, keep)
        c, st = jk(C)
        syst = abs(float(fit(bias_w(name, D) - bias_w(name, D, 3), k, keep)))
        if name in ex2:
            syst = max(syst, abs(float(fit(bias_w(name, D) - bias_w(name, D, 7, ex2), k, keep))))
        raw = float(fit(Rp.mean(1), k, keep))
        R3 = cube(name, "R", "lpt3")[e]
        d3, s3 = jk(lambda i: fit((Rp[:, i] - R3[:, i]).mean(1), k, keep))
        ps_, sp_ = jk(lambda i: disp_seed(name, D, "pm")[i].mean())
        print(f"{name} {D:.2f} {c / D**2:7.3f} {st / D**2:.3f} {syst / D**2:.3f} | {raw / D**2:7.3f} | {ol:.3f} | {c - ol * D**2:+.4f} | {d3:+.4f} {s3:.4f} | {-200 * c * 0.05**2:+.2f}% | {ps_:+.3f} {sp_:.3f}")
        out.setdefault("corrected", []).append(dict(band=name, D=D, C2d=c, stat=st, syst=syst, raw=raw, oneloop=ol, ct=c - ol * D**2, pm_3lpt=d3, s_pm_3lpt=s3, psi=ps_, spsi=sp_, dPoverP_k005=-2 * c * 0.05**2))

rows = {(r["band"], r["D"]): r for r in out["corrected"]}
if len(BANDS) == 7:
    print("\n== Density by shell, D=1 and D=.5 (differences of neighbouring splits, seed-matched, each on its own waves): PM, one loop, PM/one-loop, share")
    for D in (0.5, 1.0):
        tot = rows[("prod_ks12", D)]["C2d"]
        for a, b in zip(BANDS[:-1], BANDS[1:]):
            ka, kk = sorted_keep(a)
            kb, kl = sorted_keep(b)
            ea, eb = eidx(a, D), eidx(b, D)
            Ca = lambda i: fit(cube(a, "R", "pm")[ea][:, i].mean(1) - bias_w(a, D) - waves_R(a, D), kk, ka)
            Cb = lambda i: fit(cube(b, "R", "pm")[eb][:, i].mean(1) - bias_w(b, D) - waves_R(b, D), kl, kb)
            v, s = jk(lambda i: Ca(i) - Cb(i))
            ol = (oneloop(a, ka) - oneloop(b, kb)) * D**2
            print(f"D={D} {SPLIT[a]:.2f}-{SPLIT[b]:.2f}: {v:+.4f} +- {s:.4f}  1loop {ol:+.4f}  ratio {v / ol:.2f}  share {v / tot:.2f}")
            out.setdefault("shells", []).append(dict(D=D, lo=SPLIT[a], hi=SPLIT[b], C2d=v, s=s, oneloop=ol, share=v / tot))
        r = rows[("prod_ks70", D)]
        print(f"D={D} 0.70-1.20: {r['C2d']:+.4f} +- {r['stat']:.4f}  1loop {r['oneloop'] * D**2:+.4f}  ratio {r['C2d'] / (r['oneloop'] * D**2):.2f}  share {r['C2d'] / tot:.2f}")
        out["shells"].append(dict(D=D, lo=0.7, hi=1.2, C2d=r["C2d"], s=r["stat"], oneloop=r["oneloop"] * D**2, share=r["C2d"] / tot))

print("\n== Seed-matched PM - LPT in both channels; density: deposited, so bias and waves largely cancel")
print("band D | dens PM-7LPT +- | r PM-LPTinf +- | psi: r PM-7LPT +- PM-LPTinf +-")
for name in BANDS:
    keep, k = sorted_keep(name)
    for D in (0.2, 0.25, 0.3, 0.4, 0.5):
        e = eidx(name, D)
        Y = {m: cube(name, "R", m)[e] for m in ("pm", "lpt3", "lpt5", "lpt7")}
        C = lambda m, i: fit(Y[m][:, i].mean(1), k, keep)
        d7 = jk(lambda i: C("pm", i) - C("lpt7", i))
        _, r = sh(*[C(m, list(range(4))) for m in ("lpt3", "lpt5", "lpt7")])
        dinf = (float("nan"), float("nan"))
        if abs(r) < 0.7:
            parts = [sh(C("lpt3", i), C("lpt5", i), C("lpt7", i))[0] for i in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])]
            if np.all(np.isfinite(parts)):
                dinf = jk(lambda i: C("pm", i) - sh(C("lpt3", i), C("lpt5", i), C("lpt7", i))[0])
        P = {m: disp_seed(name, D, m) for m in ("pm", "lpt3", "lpt5", "lpt7")}
        _, rp = sh(*[P[m].mean() for m in ("lpt3", "lpt5", "lpt7")])
        p7 = jk(lambda i: (P["pm"][i] - P["lpt7"][i]).mean())
        pinf = (float("nan"), float("nan"))
        if abs(rp) < 0.7:
            pinf = jk(lambda i: P["pm"][i].mean() - sh(P["lpt3"][i].mean(), P["lpt5"][i].mean(), P["lpt7"][i].mean())[0])
        print(f"{name} {D:.2f} | {d7[0]:+.5f} {d7[1]:.5f} | {r:5.2f} {dinf[0]:+.5f} {dinf[1]:.5f} | {rp:5.2f} {p7[0]:+.5f} {p7[1]:.5f} {pinf[0]:+.5f} {pinf[1]:.5f}")
        out.setdefault("series", []).append(dict(band=name, D=D, pm_7=d7, r=float(r), pm_inf=dinf, r_psi=float(rp), psi_pm_7=p7, psi_pm_inf=pinf))
print("max |PM - 7LPT| density at D<=0.25:", max(abs(s["pm_7"][0]) for s in out["series"] if s["D"] <= 0.25))

if "prod_ks20" in BANDS:
    print("\n== Long x short, k_s=.2: shift from a=0, seed-matched, relative to the band-free member.  a=0 is deposit-bias and wave corrected")
    print("D | clean a=.5 +-, a=1 +- | raw-corrected a=.5 +-, a=1 +- | ratio a=1/a=.5 (clean, raw) | displacement shift a=.5, a=1")
    for D in (0.5, 0.7, 1.0):
        nm = ("prod_ks20", "prod_ks20_bg05", "prod_ks20_bg10")
        keep, k = sorted_keep("prod_ks20")
        Cl = {n: cube(n, "Rref", "pm")[eidx(n, D)] - cube(n, "Rref", "lpt1")[eidx(n, D)] + cube(n, "RZA")[eidx(n, D)] for n in nm}
        Raw = {n: cube(n, "Rref", "pm")[eidx(n, D)] for n in nm}
        cor = bias_w("prod_ks20", D) + waves_R("prod_ks20", D)
        # at a = 0 the band-free member is x, so "clean" keeps R_waves(7LPT) - R_waves(1LPT); at a > 0 the IR member cancels it
        Cl["prod_ks20"] = Cl["prod_ks20"] - (waves_R("prod_ks20", D, 7) - waves_R("prod_ks20", D, 1))[:, None]
        F = lambda n, i: fit(Cl[n][:, i].mean(1), k, keep)
        G = lambda n, i: fit(Raw[n][:, i].mean(1) - (cor if n == "prod_ks20" else 0), k, keep)
        c5, c1 = jk(lambda i: F(nm[1], i) - F(nm[0], i)), jk(lambda i: F(nm[2], i) - F(nm[0], i))
        r5, r1 = jk(lambda i: G(nm[1], i) - G(nm[0], i)), jk(lambda i: G(nm[2], i) - G(nm[0], i))
        P = {n: disp_seed(n, D, "pm") for n in nm}
        p5, p1 = jk(lambda i: (P[nm[1]][i] - P[nm[0]][i]).mean()), jk(lambda i: (P[nm[2]][i] - P[nm[0]][i]).mean())
        ratio_c = jk(lambda i: (F(nm[2], i) - F(nm[0], i)) / (F(nm[1], i) - F(nm[0], i)))
        ratio_r = jk(lambda i: (G(nm[2], i) - G(nm[0], i)) / (G(nm[1], i) - G(nm[0], i)))
        print(f"  ratio a=1/a=.5 jackknife: clean {ratio_c[0]:.2f} +- {ratio_c[1]:.2f}, raw {ratio_r[0]:.2f} +- {ratio_r[1]:.2f}")
        print(f"{D:.1f} | {c5[0]:+.3f} {c5[1]:.3f}, {c1[0]:+.3f} {c1[1]:.3f} | {r5[0]:+.3f} {r5[1]:.3f}, {r1[0]:+.3f} {r1[1]:.3f} | {c1[0] / c5[0]:.2f}, {r1[0] / r5[0]:.2f} | {p5[0]:+.3f} {p1[0]:+.3f} +- {p1[1]:.3f}")
        out.setdefault("background", []).append(dict(D=D, clean05=c5, clean10=c1, raw05=r5, raw10=r1, psi05=p5, psi10=p1))

print("\n== Random-long member vs eps->0, every split, D=1 (Table 2 config rg_ks*; k<=min(.08,.4 k_s)) minus Table 6: different windows, no bias correction")
for name in BANDS:
    tag = name.replace("prod_", "rg_")
    with h5py.File(os.path.join(here, f"{tag}.h5")) as h:
        g = h["dens"]["fit1"]
        km = g["kmax"][()]
        j = int(np.argmin(abs(km - min(0.08, 0.4 * SPLIT[name]))))
        e = int(np.argmin(abs(h["epochs"][()] - 1.0)))
        c, s = float(g["C2"][j, e, 7]), float(g["sC2"][j, e, 7])
    t = rows[(name, 1.0)]
    print(f"{name} rg {c:.3f} +- {s:.3f} (kmax {km[j]:.2f}) | Table 6 {t['C2d']:.3f} +- {t['stat']:.3f} | shift {c - t['C2d']:+.3f}  | rg - 1loop {c - t['oneloop']:+.3f}")
    out.setdefault("random_long", []).append(dict(band=name, rg=c, s=s, table6=t["C2d"], shift=c - t["C2d"], rg_minus_1loop=c - t["oneloop"]))

print("\n== Excess over one loop vs D (Table 6): local slopes at D>=.5; caution, 3LPT alone also departs from pure one loop")
for name in BANDS:
    t = [rows[(name, D)]["ct"] for D in (0.5, 0.7, 1.0)]
    print(f"{name} " + " ".join(f"{x:+.4f}" for x in t) + f" slopes {np.log(t[1] / t[0]) / np.log(1.4):.2f} {np.log(t[2] / t[1]) / np.log(1 / 0.7):.2f}")

with open(os.path.join(here, "pw_density_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
