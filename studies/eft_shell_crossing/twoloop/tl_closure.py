"""Time-resolved closure of the linear stress coefficient in the full - IR estimator (results/tl_closure_<A>.json).

    python tl_closure.py <A> [Lam=0.2] [model=pm]

With c(D) = c0 + c2 D², the D²-normalised stress drives, through the EdS Green's function (C2 ∝ D^m: stress = m(m+5/2) C2/D²...),
    stress    c_eff = c0 +        c2 D²
    density   c_eff = c0 + (9/26) c2 D²        [C2 = c0 D²/9 + c2 D⁴/26]
    momentum  c_eff = c0 + (5/3)(9/26) c2 D²   [velocity weight (1 + m)/3]
Density and momentum are the fixed-epoch sums of tl_fit3, whose regressors also contain δ̃³ (D⁵): an approximation at late D.
Also the direct density coefficient C2 = -Re[Δδ δ1*]/(D |δ1|² k²) on k <= kmax, for comparison with README §7 and §14.11.
"""
import json, os, sys
import numpy as np
import tl_fit3 as F
from tl_fit2 import load_ops2

A = sys.argv[1]; Lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2; model = sys.argv[3] if len(sys.argv) > 3 else "pm"
OP = {}; K = None
for s in F.SEEDS:
    try:
        OP[s], nvec, K, pk = load_ops2(s)
    except (OSError, KeyError):
        pass
ne, raw, Ds, kmask = F.build(A, "IR", model, Lam, OP, K, 0.12)
seeds = sorted({k[0] for k in ne.A})
WD, WM = 9 / 26, (5 / 3) * (9 / 26)


def sums(ss, D):
    out = {}
    p, *_ = F.fit_set(ne, [D], ("tau_odd",), F.LIN, ss)
    out["stress"] = p[0] + p[1] if D in ne.stress_epochs else np.nan
    out["dens"] = F.fit_lin_sum(ne, [D], ("dens_odd",), ss)[0]
    out["mom"] = F.fit_lin_sum(ne, [D], ("mom_odd",), ss)[0]
    return out


def linfit(ss, Dset, use):
    X = []; y = []; 
    for D in Dset:
        s = sums(ss, D)
        for tag, w in (("stress", 1.0), ("dens", WD), ("mom", WM)):
            if tag in use and np.isfinite(s[tag]):
                X.append([1, w * D ** 2]); y.append(s[tag])
    return np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0], np.array(X), np.array(y)


def jack(f):
    full = f(seeds)
    parts = np.array([f([s for s in seeds if s != s0]) for s0 in seeds]); n = len(seeds)
    return full, np.sqrt((n - 1) / n * ((parts - parts.mean(0)) ** 2).sum(0))


I = F.I_band(pk, Lam, {"tl_full": 1.6, "tl_uv06": 0.6, "tl_uv10": 1.0, "tl_div": 1.6, "tl_div10": 1.0}[A], mod=(0.3, -1.0) if "div" in A else None)
res = dict(A=A, Lam=Lam, model=model, I=I, seeds=seeds, kappa_d1=61 / 70)
print(f"{A} − IR, Λ = {Lam}, {model}, seeds {seeds}, I = {I:.3f}, 61/70 = {61/70:.3f}.  All /I")
Dw = [D for D in Ds if D <= 0.3]
print("\nc_eff at fixed epoch:  D  stress  density  momentum")
for D in Ds:
    v, e = jack(lambda ss: np.array([sums(ss, D)[k] for k in ("stress", "dens", "mom")]))
    print(f"  {D:.1f}  {v[0]/I:.3f}±{e[0]/I:.3f}  {v[1]/I:.3f}±{e[1]/I:.3f}  {v[2]/I:.3f}±{e[2]/I:.3f}")
    res.setdefault("fixed_epoch", []).append(dict(D=D, stress=[v[0]/I, e[0]/I], dens=[v[1]/I, e[1]/I], mom=[v[2]/I, e[2]/I]))
print("\nc(D) = c0 + c2 D², fit over D = 0.1–0.3:")
for name, use, dset in (("stress only", ("stress",), Dw), ("density only", ("dens",), Dw), ("momentum only", ("mom",), Dw), ("density + momentum", ("dens", "mom"), Dw),
                        ("all three", ("stress", "dens", "mom"), Dw), ("all three, D = 0.2–0.3", ("stress", "dens", "mom"), [d for d in Dw if d >= 0.2])):
    v, e = jack(lambda ss: linfit(ss, dset, use)[0])
    print(f"  {name:24s} c0 = {v[0]/I:.3f} ± {e[0]/I:.3f}   c2 = {v[1]/I:.2f} ± {e[1]/I:.2f}")
    res.setdefault("c0c2", []).append(dict(fit=name, c0=[v[0]/I, e[0]/I], c2=[v[1]/I, e[1]/I]))
v, X, y = linfit(seeds, Dw, ("stress", "dens", "mom"))
print("  residuals of the all-three fit (data − model)/I, D=.1,.2,.3, stress dens mom:", np.round((y - X @ v) / I, 3).tolist())
res["resid_all3"] = ((y - X @ v) / I).tolist()

print("\nDirect density response, odd part, sharp-filtered modes, k²-only fit:  D  C2(k≤0.08)  ±  C2(k≤0.05)  ±")
for D in Ds:
    def f(ss, km):
        m = (K > 0) & (K <= km); num = 0; den = 0
        for s in ss:
            o = raw[(s, D)]["odd"]["delta"][m]; x = D * OP[s][Lam]["delta1"][m]
            r = np.real(o * np.conj(x)); w = np.abs(x) ** 2
            num += (-K[m] ** 2 * r).sum(); den += (K[m] ** 4 * w).sum()
        return num / den
    a = jack(lambda ss: f(ss, 0.08)); b = jack(lambda ss: f(ss, 0.05))
    print(f"  {D:.1f}  {a[0]:.3f} ± {a[1]:.3f}   {b[0]:.3f} ± {b[1]:.3f}")
    res.setdefault("direct_C2", []).append(dict(D=D, k08=[float(a[0]), float(a[1])], k05=[float(b[0]), float(b[1])]))

print("\nWindow D ≤ 0.3, stress only: combinations /I (analytic d1=.87, rd1=2.04, d2+dd1=3.64, d3+rdd1=2.23)")
def eff(ss):
    p, *_ = ne.solve(lambda k: k[0] in ss and k[1] in Dw and k[2] in ("tau_odd", "tau_even"), F.LABELS, lambda k: F.weights(ne, ("tau_odd", "tau_even"))[k])
    e = F.T.to_effective(dict(zip(F.LABELS, p)))
    return np.array([e["d1"], e["rd1"], e["d2"] + e["dd1"], e["d3"] + e["rdd1"], e["d2"], e["dd1"], e["d3"], e["rdd1"], e["rd2"]])
v, er = jack(eff)
for n, a, b in zip(("d1", "rd1", "d2+dd1", "d3+rdd1", "d2", "dd1", "d3", "rdd1", "rd2"), v, er):
    print(f"  {n:8s} {a/I:+7.2f} ± {b/I:.2f}")
    res.setdefault("window", {})[n] = [float(a/I), float(b/I)]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"tl_closure_{A}.json")
json.dump(res, open(out, "w"), indent=1, default=float)
