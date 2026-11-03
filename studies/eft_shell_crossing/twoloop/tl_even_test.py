"""Second-order coefficients from the even stress alone, one epoch at a time (light CPU):  python tl_even_test.py <A> <B|IR> <Lam> <models>
Fits the 7 D⁴ C^(2) operators of tau_even; prints raw c/I, the paper's effective combinations (A.21–A.30, restricted to the second-order labels), and analytic κ."""
import sys
import numpy as np
import tl_fit3 as F
from tl_fit2 import load_ops2
A, B, Lam, models = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4].split(",")
OP = {}; K = None
for s in F.SEEDS:
    try:
        OP[s], nvec, K, pk = load_ops2(s)
    except (OSError, KeyError):
        pass
KUV = {"tl_full": 1.6, "tl_uv10": 1.0, "tl_uv06": 0.6}
I = F.I_band(pk, Lam if B == "IR" else KUV[B], KUV[A])
L7 = F.EVEN7
def eff(p):
    c = dict(zip(L7, p))
    return np.array([c["d2"] + 7/4*c["rr1"] - 3/4*c["r2"] + 7/4*c["r1"], c["rd1"] + c["rr1"] + c["r2"] - c["r1"], c["dd1"] - 7/4*c["rr1"] + 3/4*c["r2"] - 3/4*c["r1"]])
print(f"{A} - {B}, Lam={Lam}, I={I:.3f}. analytic eff d2=2.11 rd1=2.04 dd1=1.53 sum(d2+dd1)=3.64")
for m in models:
    ne, raw, Ds, kmask = F.build(A, B, m, Lam, OP, K, 0.12)
    seeds = sorted({k[0] for k in ne.A})
    for D in ne.stress_epochs:
        def f(ss):
            return ne.solve(lambda k: k[0] in ss and k[1] == D and k[2] == "tau_even", L7, lambda k: 1.0)[0]
        p = f(seeds); J = np.array([f([s for s in seeds if s != s0]) for s0 in seeds]); n = len(seeds)
        e = eff(p); Je = np.array([eff(q) for q in J]); se = np.sqrt((n-1)/n*((Je-Je.mean(0))**2).sum(0))
        sm = np.sqrt((n-1)/n*((Je[:,0]+Je[:,2]-(Je[:,0]+Je[:,2]).mean())**2).sum())
        Y = sum(ne.y2[k] for k in ne.A if k[1]==D and k[2]=="tau_even")
        chi = sum(ne.chi2(k, np.array(list(p)+[0]*0) if False else ne.solve(lambda q:q==k, L7, lambda q:1.0)[0], L7) for k in ne.A if k[1]==D and k[2]=="tau_even")
        print(f" {m:5s} D={D:.1f} raw/I " + " ".join(f"{l}={v/I:+.2f}" for l, v in zip(L7, p)) + f" | eff d2={e[0]/I:+.2f}±{se[0]/I:.2f} rd1={e[1]/I:+.2f}±{se[1]/I:.2f} dd1={e[2]/I:+.2f}±{se[2]/I:.2f} sum={(e[0]+e[2])/I:+.2f}±{sm/I:.2f} | resid {chi/Y:.3f}", flush=True)
