"""Band-edge control: exact one-loop (EdS, App. C P13 kernel) density response to an added band [k_s, k_uv) vs
its UV-limit k² form, and what C2(+C4) fits over k ≤ k_max return.  → chs/band_edge.json

    python3 chs/band_edge.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chs_fig3 as C

HERE = os.path.dirname(os.path.abspath(__file__))
pk = json.load(open(os.path.join(HERE, "planck18_pk.json"))); P = C.Spectrum(np.array(pk["k"]), np.array(pk["Pk"]))
kuv = 1.2
kk = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18])
out = {"k": kk.tolist(), "k_uv": kuv, "splits": {}}
for ks in (0.12, 0.15, 0.2, 0.25, 0.33, 0.5, 0.7):
    W2 = lambda q, ks=ks: ((np.asarray(q) >= ks) & (np.asarray(q) < kuv)).astype(float)
    R = np.array([C.p13(k, P, W2) / (2 * P(k)) for k in kk])
    q = np.exp(np.linspace(np.log(ks), np.log(kuv), 20000)); sv2 = np.trapz(P(q), q) / (6 * np.pi ** 2)
    C2uv = (61 / 105) * sv2 / 2
    fits = {}
    for kmax in (0.08, 0.12, 0.18):
        s = (kk <= kmax) & (kk < 0.9 * ks); A = np.vstack([-kk[s] ** 2, kk[s] ** 4]).T
        if s.sum() >= 3:
            c2, c4 = np.linalg.lstsq(A, R[s], rcond=None)[0]; fits[str(kmax)] = {"C2": c2, "C4": c4}
    out["splits"][str(ks)] = {"R_over_UVlimit": (R / (-C2uv * kk ** 2)).tolist(), "C2_uv": C2uv, "fits": fits}
    print(f"k_s = {ks}: R/R_UV at k = 0.08, 0.12, 0.18: "
          + ", ".join(f"{x:.3f}" for x in (R / (-C2uv * kk ** 2))[[6, 8, 10]]) + f";  fits {fits}")
json.dump(out, open(os.path.join(HERE, "band_edge.json"), "w"), indent=1)
