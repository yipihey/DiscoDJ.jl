#!/usr/bin/env python3
"""Stage 7 summary of all tl_fit3 and closure fits -> results/stage7_summary.json.  Prints every deliverable; quantities with unstable or singular fits are flagged."""
import json, os, sys
import numpy as np, h5py
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); SCR = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
L = lambda f: json.load(open(os.path.join(R, f)))
KAP = {"d1": 61 / 70, "d2": 12409 / 5880, "d3": 167099 / 64680, "rd1": 6997 / 3430, "rd2": 27241 / 44100, "dd1": 63149 / 41160, "rdd1": -60145999 / 169785000}
out = {}
print("== 1. Closure: c(D) = c0 + c2 D^2, PM, D = 0.1-0.3, /I  (61/70 = 0.871).  stress | density | momentum | all three.  Each c0 +- , c2 +- ")
for c in ("tl_full", "tl_uv10", "tl_uv06", "tl_div", "tl_div10"):
    for lam in ("0.1", "0.15", "0.2", "0.3"):
        f = os.path.join(R, f"tl_closure_{c}_L{lam}.json")
        if not os.path.exists(f): continue
        r = json.load(open(f)); x = {y["fit"]: y for y in r["c0c2"]}
        g = lambda n: x[n]
        t = " | ".join(f"{g(n)['c0'][0]:.3f}±{g(n)['c0'][1]:.3f} {g(n)['c2'][0]:.2f}±{g(n)['c2'][1]:.2f}" for n in ("stress only", "density only", "momentum only", "all three"))
        print(f"{c:9s} L={lam:4s} I={r['I']:7.2f} | {t}")
        out.setdefault("closure", []).append(dict(cfg=c, lam=float(lam), I=r["I"], fits=r["c0c2"], window=r.get("window"), direct=r.get("direct_C2")))

print("\n== 2. Running of the linear coefficient: stress c_lin at D = 0.1 against I(Lambda), full - IR; slope should be 61/70 in the UV limit")
for c in ("tl_full", "tl_uv10", "tl_uv06", "tl_div", "tl_div10"):
    r = L(f"tl_fit3_{c}_minus_IR.json")
    for m in ("lpt3", "lpt7", "pm"):
        I = []; v = []; d = []
        for lam in ("0.100", "0.150", "0.200", "0.300"):
            k = f"{m}|L{lam}"
            if k not in r: continue
            p = r[k]["per_epoch"].get("0.100")
            if p is None: continue
            I.append(r[k]["I"]); v.append(p["stress"]["sum"][0]); d.append(p["dens"]["sum"][0])
        if len(I) >= 2:
            s = np.polyfit(I, v, 1); sd = np.polyfit(I, d, 1)
            print(f"{c:9s} {m:5s} I " + " ".join(f"{x:.1f}" for x in I) + " | c/I " + " ".join(f"{a / b:.3f}" for a, b in zip(v, I)) + f" | dc/dI stress {s[0]:.3f} dens {sd[0]:.3f}")
            out.setdefault("running", []).append(dict(cfg=c, model=m, I=I, stress=v, dens=d, slope=float(s[0])))

print("\n== 3. PM vs 7LPT, stress c_lin/I, full - IR, Lambda = 0.1 (longest stress window).  D | PM | 3LPT | 7LPT")
for c in ("tl_full", "tl_div"):
    r = L(f"tl_fit3_{c}_minus_IR.json")
    for D in ("0.100", "0.200", "0.300", "0.400", "0.500", "0.600"):
        v = []
        for m in ("pm", "lpt3", "lpt7"):
            x = r[f"{m}|L0.100"]; p = x["per_epoch"].get(D); v.append(p["stress"]["sum"][0] / x["I"] if p else np.nan)
        print(f"{c} D={D} {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}")

print("\n== 4. The seven effective coefficients /I, stress windows D<=0.3, per config and Lambda; analytic: " + " ".join(f"{k}={v:+.2f}" for k, v in KAP.items()))
print("  (full - IR with hard modes near Lambda; the band fits, far from Lambda, are the fairer UV test)")
for f in ("tl_full_minus_IR", "tl_div_minus_IR", "tl_full_minus_tl_uv06", "tl_full_minus_tl_uv10", "tl_div_minus_tl_div10"):
    r = L(f"tl_fit3_{f}.json")
    for k, x in r.items():
        m, l = k.split("|")
        if m not in ("pm", "lpt3"): continue
        w = x["window"].get("D<=0.3")
        if not w: continue
        e = w["stress"]["eff"]; I = x["I"]
        print(f"{f[:22]:22s} {k:13s} I={I:6.2f} cond {w['stress']['cond']:.0e} " + " ".join(f"{g}={e[g][0] / I:+6.2f}±{e[g][1] / I:.2f}" for g in KAP))
        out.setdefault("eff", []).append(dict(fit=f, key=k, I=I, cond=w["stress"]["cond"], eff={g: [e[g][0] / I, e[g][1] / I] for g in KAP}))

print("\n== 5. Stochastic trace residual, tl_full - IR, per k bin (L^3 normalised), odd, slope against k; expected white (k^0) at low k")
r = L("tl_fit3_tl_full_minus_IR.json")
for k, x in r.items():
    if not k.startswith(("pm|", "lpt7|")): continue
    for D, p in x["per_epoch"].items():
        st = p.get("stochastic_trace")
        if not st: continue
        e = np.array(st["k_edges"]); kc = np.sqrt(e[:-1] * np.maximum(e[1:], 1e-9)); ov = np.array(st["odd"], float); ev = np.array(st["even"], float)
        use = np.isfinite(ov) & (e[:-1] > 0) & (kc < 0.13)
        sl = np.polyfit(np.log(kc[use]), np.log(ov[use]), 1)[0]; sle = np.polyfit(np.log(kc[use]), np.log(ev[use]), 1)[0]
        print(f"{k:10s} D={D} P_odd(k) " + " ".join(f"{a:.1e}" for a in ov[use]) + f" slope odd {sl:+.2f} even {sle:+.2f}")
        out.setdefault("stochastic", []).append(dict(key=k, D=D, k=kc[use].tolist(), odd=ov[use].tolist(), even=ev[use].tolist(), slope_odd=float(sl), slope_even=float(sle)))

print("\n== 6. Vorticity: rms curl v_Lambda against rms k.pi_Lambda at k<=0.12, from the stress files (seed-sign mean, Lambda = 0.2), PM vs 7LPT; ratio in %")
def cplx(a): return a["re"] + 1j * a["im"] if a.dtype.names else (a[..., 0] + 1j * a[..., 1] if a.shape[-1] == 2 else a)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl_fit2 import load_stress2, load_ops2
_, nvec, K, _ = load_ops2(1)
KV = 2 * np.pi / 500 * nvec.astype(float); m_ = (K > 0) & (K <= 0.12)
for cfg in ("tl_full", "tl_div"):
    for model in ("pm", "lpt7"):
        acc = {}
        for s in (1, 2, 3, 4):
            for sg in (1, -1):
                a = load_stress2(cfg, model, s, sg)
                for (D, l), f in a.items():
                    if l != 0.2: continue
                    w = (np.abs(f["curlv"][m_]) ** 2).sum(1).mean(); kp = (np.abs((KV[m_] * f["P"][m_]).sum(1)) ** 2).mean()
                    acc.setdefault(D, []).append((w, kp))
        print(cfg, model, " ".join(f"D={D}: {100 * np.sqrt(np.mean([x[0] for x in acc[D]]) / np.mean([x[1] for x in acc[D]])):.2f}%" for D in sorted(acc)))
        out.setdefault("vort", []).append(dict(cfg=cfg, model=model, D=sorted(acc), ratio=[float(np.sqrt(np.mean([x[0] for x in acc[D]]) / np.mean([x[1] for x in acc[D]]))) for D in sorted(acc)]))
json.dump(out, open(os.path.join(R, "stage7_summary.json"), "w"), indent=0, default=float)
