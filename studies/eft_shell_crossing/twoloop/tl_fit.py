"""Experiment 3: field-level projection of the measured coarse-grained stress onto the two-loop operator basis of
arXiv:2509.05187, and the derived tests (twoloop/README.md §3).

    python tl_fit.py <config> [models=pm,lpt3,lpt5,lpt7] [kfit=0.15]

For each model, epoch D and cutoff Λ the six components of τ̂ = τ̂_kin + τ̂_grav are regressed jointly (all seeds, both
signs, Frobenius weights 1 on diagonal / 2 on off-diagonal components, modes k ≤ k_fit) on

    τ̂(k; D, s_L) = Σ_c  c̄_c Σ_{n ∈ orders(c)} D^{n+2} s_L^n C_c^{(n)}(k)  + ε                         (Eq. 2.50–2.51)

with the 18 coefficients of (2.51) [c̄ = c/k_NL², h⁻²Mpc²; lapd1 = c_{∂²δ,1}/k_NL⁴, h⁻⁴Mpc⁴]. Odd orders flip sign with
s_L, so the ±L pairs separate orders {1,3} from {2} exactly. A "free" variant gives every C-function its own coefficient
(tests of the fluid-element structure T4 and of time non-locality T3).  The 18 are mapped to the paper's 7 + 1
P(k)-level coefficients with (A.21–A.32).
"""
import os, sys, json, glob
import numpy as np, h5py

SCR = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
cplx = lambda x: x["r"] + 1j * x["i"] if x.dtype.names else x
L = 500.0; KF = 2 * np.pi / L
COMPW = np.array([1, 1, 1, 2, 2, 2.0])                      # Frobenius weights (xx, yy, zz, xy, xz, yz)

OPS = [("C1_r1", 1, "r1"), ("C1_d1", 1, "d1"),
       ("C2_r1", 2, "r1"), ("C2_r2", 2, "r2"), ("C2_d1", 2, "d1"), ("C2_d2", 2, "d2"),
       ("C2_r2_1", 2, "rr1"), ("C2_rd1", 2, "rd1"), ("C2_dd1", 2, "dd1"),
       ("C3_r1", 3, "r1"), ("C3_r2", 3, "r2"), ("C3_r3", 3, "r3"), ("C3_d1", 3, "d1"), ("C3_d2", 3, "d2"),
       ("C3_d3", 3, "d3"), ("C3_rr1", 3, "rr1"), ("C3_rr2", 3, "rr2"), ("C3_rd1", 3, "rd1"), ("C3_rd2", 3, "rd2"),
       ("C3_dd1", 3, "dd1"), ("C3_dd2", 3, "dd2"), ("C3_rrr1", 3, "rrr1"), ("C3_rrd1", 3, "rrd1"),
       ("C3_rdd1", 3, "rdd1"), ("C3_ddd1", 3, "ddd1"), ("C3_trr2r1", 3, "trr2r1"), ("Cd2_d1", 1, "lapd1")]
LABELS = ["r1", "d1", "r2", "d2", "rr1", "rd1", "dd1", "r3", "d3", "rr2", "rd2", "dd2", "rrr1", "rrd1", "rdd1",
          "ddd1", "trr2r1", "lapd1"]
PAPER = {"r1": "c_{r,1}", "d1": "c_{δ,1}", "r2": "c_{r,2}", "d2": "c_{δ,2}", "rr1": "c_{r²,1}", "rd1": "c_{rδ,1}",
         "dd1": "c_{δ²,1}", "r3": "c_{r,3}", "d3": "c_{δ,3}", "rr2": "c_{r²,2}", "rd2": "c_{rδ,2}", "dd2": "c_{δ²,2}",
         "rrr1": "c_{r³,1}", "rrd1": "c_{r²δ,1}", "rdd1": "c_{rδ²,1}", "ddd1": "c_{δ³,1}", "trr2r1": "c_{(r²)r,1}",
         "lapd1": "c_{∂²δ,1}/k_NL²"}
# D power of each function: D^{n+2}, except the higher-derivative term (D⁵, linear in δ: odd)
DPOW = {name: (5 if name == "Cd2_d1" else n + 2) for name, n, _ in OPS}
PAR = {name: (1 if name == "Cd2_d1" else n % 2) for name, n, _ in OPS}     # parity in s_L (1 = odd)

# ── (A.21–A.32): integrand degeneracies → the paper's seven one-loop-type coefficients (4.49) ──────────────────────────
GAMMA = ["d1", "d2", "d3", "rd1", "rd2", "dd1", "rdd1"]
_DEG = {   # K_X = Σ a_XΓ K_Γ over Γ' = GAMMA + rrr1, r3
    "dd2":    {"d3": 775/187, "rrr1": 60/187, "r3": 200/187, "rdd1": -401/187, "rd2": -130/187},
    "trr2r1": {"d3": 1085/374, "rrr1": 416/187, "r3": 140/187, "rdd1": -1459/374, "rd2": -91/187},
    "rr1":    {"dd1": -7/4, "d2": 7/4, "rd1": 1},
    "rr2":    {"d3": -1225/374, "rrr1": -156/187, "r3": -520/187, "rdd1": 1225/374, "rd2": 525/187},
    "r2":     {"dd1": 3/4, "d2": -3/4, "d3": 705/374, "rrr1": 160/187, "r3": -90/187, "rdd1": -705/374, "rd1": 1,
               "rd2": -35/187},
    "r1":     {"dd1": -3/4, "d1": 1, "d2": 7/4, "d3": -331/374, "rrr1": -160/187, "r3": -97/187, "rdd1": 705/374,
               "rd1": -1, "rd2": 35/187},
    "ddd1":   {"d3": 1085/374, "rrr1": 42/187, "r3": 140/187, "rdd1": -711/374, "rd2": -91/187},
    "rrd1":   {"d3": -1085/748, "rrr1": -21/187, "r3": -70/187, "rdd1": 1833/748, "rd2": 91/374},
}
# angular degeneracies (A.32): K_rrr1 = (13/25) K_rdd1;  K_r3 = (13/20)(33513/2366 K_rrr1 − 155/26 K_d3 + K_rd2)


def to_effective(c):
    """18 stress coefficients → the 7 one-loop-type coefficients of (4.49) (+ lapd1 unchanged)."""
    e = {g: c.get(g, 0.0) for g in GAMMA + ["rrr1", "r3"]}
    for X, row in _DEG.items():
        for g, a in row.items():
            e[g] = e.get(g, 0.0) + a * c.get(X, 0.0)
    # eliminate r3 then rrr1 (A.32)
    r3 = e.pop("r3")
    e["rrr1"] += r3 * (13 / 20) * 33513 / 2366; e["d3"] += r3 * (13 / 20) * (-155 / 26); e["rd2"] += r3 * 13 / 20
    rrr1 = e.pop("rrr1"); e["rdd1"] += rrr1 * 13 / 25
    e["lapd1"] = c.get("lapd1", 0.0)
    return e


# paper's UV coefficients κ_α: single-stream stress from modes above Λ, c̄_α = κ_α I(Λ)   (sign: c^τ = −c^{[L]})
KAPPA = {"d1": 61 / 70, "d2": 12409 / 5880, "rd1": 6997 / 3430, "dd1": 63149 / 41160, "d3": 167099 / 64680,
         "rd2": 27241 / 44100, "rdd1": -60145999 / 169785000}


def I_box(Lam, pk_k, pk_P, kuv, N=256):
    """I(Λ) = ∫_p P(p)/p² (1 − W_Λ²(p)) as the exact sum over the box modes 0 < |p| < k_uv (z = 0 / D = 1)."""
    n = np.arange(-N // 2 + 1, N // 2)
    n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
    n2 = n2[(n2 > 0) & (KF * np.sqrt(n2) < kuv)]
    u, c = np.unique(n2, return_counts=True); q = KF * np.sqrt(u)
    P = np.exp(np.interp(np.log(q), np.log(pk_k), np.log(pk_P)))
    return float((c * P / q ** 2 * (1 - np.exp(-q ** 2 / Lam ** 2))).sum() / L ** 3)


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_ops(cfg, seed):
    h = h5py.File(os.path.join(SCR, cfg, "tl", f"ops_L{seed}.h5"), "r")
    out = {}
    for g in h:
        if not g.startswith("L"):
            continue
        Lam = round(float(g[1:]), 3)
        out[Lam] = {k: cplx(h[g][k][()]).T for k in h[g] if k.startswith("C")}   # (M, 6)
        out[Lam]["delta"] = {n: cplx(h[g][f"delta{n}"][()]) for n in (1, 2, 3)}
    nvec = h["nvec"][()]
    nvec = nvec if nvec.shape[0] != 3 else nvec.T
    return out, KF * np.sqrt((nvec ** 2).sum(1)), cplx(h["dL"][()])


def load_stress(cfg, model, seed, sign):
    f = os.path.join(SCR, cfg, "tl", f"stress_{model}_L{seed}_s{sign:+d}.h5")
    if not os.path.exists(f):
        return None
    h = h5py.File(f, "r"); out = {}
    for gD in h:
        if not gD.startswith("D"):
            continue
        D = float(gD[1:])
        for gL in h[gD]:
            Lam = round(float(gL[1:]), 3); g = h[gD][gL]
            out[(D, Lam)] = dict(kin=cplx(g["kin"][()]).T, grav=cplx(g["grav"][()]).T, delta=cplx(g["delta"][()]),
                                 curlv=cplx(g["curlv"][()]).T, pi=cplx(g["pi"][()]).T)
    return out


# ── regression ──────────────────────────────────────────────────────────────────────────────────────────────────────
def design(ops, D, s, free=False):
    """Columns: one per coefficient label (shared) or per C-function (free); each column is (M,6) complex."""
    cols = {}
    for name, n, lab in OPS:
        key = name if free else lab
        v = D ** DPOW[name] * (s ** PAR[name]) * ops[name]
        cols[key] = cols.get(key, 0) + v
    return cols


def fit(ys, Xs, kmask, keys):
    """Weighted LS over (realizations, modes ≤ kfit, 6 components); real coefficients."""
    A = []; b = []
    w = np.sqrt(COMPW)[None, :]
    for y, X in zip(ys, Xs):
        b.append((y[kmask] * w).ravel())
        A.append(np.stack([(X[k][kmask] * w).ravel() for k in keys], 1))
    A = np.concatenate(A); b = np.concatenate(b)
    Ar = np.concatenate([A.real, A.imag]); br = np.concatenate([b.real, b.imag])
    p, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    return p


def run(cfg, models, kfit):
    seeds = sorted({int(os.path.basename(f).split("_L")[1].split(".")[0]) for f in glob.glob(os.path.join(SCR, cfg, "tl", "ops_L*.h5"))})
    OP = {}; K = None
    for s in seeds:
        OP[s], K, _ = load_ops(cfg, s)
    res = {}
    for model in models:
        data = {(s, sg): load_stress(cfg, model, s, sg) for s in seeds for sg in (1, -1)}
        data = {k: v for k, v in data.items() if v is not None}
        if not data:
            continue
        keys_DL = sorted(set.intersection(*[set(v.keys()) for v in data.values()]))
        for (D, Lam) in keys_DL:
            if Lam not in OP[seeds[0]]:
                continue
            kmask = (K <= min(kfit, Lam / 2)) & (K > 0)
            rec = {}
            for free in (False, True):
                ks = None; ys = []; Xs = []; tags = []
                for (s, sg), v in data.items():
                    X = design(OP[s][Lam], D, sg, free); ks = ks or list(X.keys())
                    ys.append(v[(D, Lam)]["kin"] + v[(D, Lam)]["grav"]); Xs.append(X); tags.append(s)
                p = fit(ys, Xs, kmask, ks)
                # jackknife over long seeds
                jk = []
                for s0 in seeds:
                    sel = [i for i, t in enumerate(tags) if t != s0]
                    if len(sel) and len(set(tags)) > 1:
                        jk.append(fit([ys[i] for i in sel], [Xs[i] for i in sel], kmask, ks))
                n = len(jk)
                err = np.sqrt((n - 1) / n * ((np.array(jk) - p) ** 2).sum(0)) if n > 1 else np.full_like(p, np.nan)
                rec["free" if free else "shared"] = dict(zip(ks, zip(p.tolist(), err.tolist())))
                if not free:
                    # residual (stochastic) spectrum of the trace and of the longitudinal projection k̂_ik̂_jτ_ij
                    resid = [y - sum(p[i] * X[k] for i, k in enumerate(ks)) for y, X in zip(ys, Xs)]
                    rec["resid"] = residual_stats(resid, ys, cfg, K, Lam)
                    rec["eff"] = {g: v for g, v in to_effective(dict(zip(ks, p))).items()}
                    if n > 1:
                        effj = np.array([[to_effective(dict(zip(ks, q)))[g] for g in GAMMA + ["lapd1"]] for q in jk])
                        e0 = np.array([rec["eff"][g] for g in GAMMA + ["lapd1"]])
                        rec["eff_err"] = dict(zip(GAMMA + ["lapd1"], np.sqrt((n - 1) / n * ((effj - e0) ** 2).sum(0)).tolist()))
            res[f"{model}|D{D:.3f}|L{Lam:.3f}"] = rec
            print(f"{model:5s} D={D:.2f} Λ={Lam:.2f}  c̄_δ1+c̄_r1 = {rec['shared']['d1'][0] + rec['shared']['r1'][0]:8.3f}"
                  f"  eff: " + " ".join(f"{g}={rec['eff'][g]:+.2f}" for g in GAMMA), flush=True)
    return res, K


def residual_stats(resid, ys, cfg, K, Lam):
    """Binned power of the residual trace and longitudinal part vs the total (all realizations)."""
    edges = KF * (np.arange(1, 25) - 0.5)
    nb = len(edges) - 1
    b = np.digitize(K, edges) - 1
    out = {"k": (0.5 * (edges[1:] + edges[:-1])).tolist()}
    for lab, arr in (("resid", resid), ("total", ys)):
        tr = sum(np.abs(a[:, 0] + a[:, 1] + a[:, 2]) ** 2 for a in arr)
        out[lab + "_trace"] = [float(tr[b == i].mean()) * L ** 3 / len(arr) if np.any(b == i) else np.nan for i in range(nb)]
    return out


if __name__ == "__main__":
    cfg = sys.argv[1]
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pm", "lpt3", "lpt5", "lpt7"]
    kfit = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    res, K = run(cfg, models, kfit)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"tl_fit_{cfg}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)
