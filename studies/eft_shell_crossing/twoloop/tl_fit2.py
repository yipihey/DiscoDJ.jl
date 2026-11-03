"""Experiment 3, paired estimator: short-mode stress and counterterm responses, field level.

    python tl_fit2.py <config> [models=pm,lpt3,lpt5,lpt7] [kfit=0.12]

Observables (sharp coarse-graining at Λ, full member minus the IR member whose ICs are cut at Λ — same long modes):
    Δτ̂_ij      = τ̂[full] − τ̂[IR]                         (6 components; tree-level coarse-graining of long modes cancels)
    Δδ_Λ       = δ_Λ[full] − δ_Λ[IR]                       (density response to the short modes)
    ΔP_Λ       = P_Λ[full] − P_Λ[IR],  P = [(1+δ)Π]_Λ       (momentum response; carries the generated vorticity)
Model, linear in the 18 coefficients c̄_α of Eq. (2.51) (twoloop/README.md):
    Δτ̂  = Σ_α c̄_α Σ_n D^{n+2} s^n C^{(n)}_α
    Δδ  = Σ_α c̄_α [D³ s δ̃¹_α + D⁴ δ̃²_α + D⁵ s δ̃³_α]                     (Eqs. 2.57, 2.59, 2.61; lapd1: D⁵ s k⁴δ̃₁/26)
    ΔP  = −Σ_α c̄_α [D² s π̃¹_α + D³ π̃²_α]                                (Eqs. 2.58, 2.60; P = π/(ρ̄ℋfD))
Four fits per (model, D, Λ): stress-only, density-only, momentum-only, joint.  Closure = agreement of the P(k)-relevant
combinations (c̄_δ1 + c̄_r1 and the 7+1 effective coefficients of 4.49) between the stress and response fits.
"""
import os, sys, json
import numpy as np, h5py
import tl_fit as T                                         # OPS, DPOW, PAR, to_effective, GAMMA, KAPPA

SCR = T.SCR; L = T.L; KF = T.KF
cplx = T.cplx
LAMBDAS = [0.1, 0.15, 0.2, 0.3]
IRCFG = {0.1: "tl_ir010", 0.15: "tl_ir015", 0.2: "tl_ir020", 0.3: "tl_ir030"}
LABELS = T.LABELS


def load_ops2(seed, filt="sharp"):
    h = h5py.File(os.path.join(SCR, "tl_full", "tl", f"ops_{filt}_L{seed}.h5"), "r")
    nvec = h["nvec"][()]; nvec = nvec if nvec.shape[0] != 3 else nvec.T
    out = {}
    for g in h:
        if not g.startswith("L"):
            continue
        Lam = round(float(g[1:]), 3); G = h[g]
        o = {k: cplx(G[k][()]).T for k in G if k.startswith("C")}
        o["resp"] = {lab: {k: (cplx(G["resp"][lab][k][()]).T if G["resp"][lab][k].ndim == 2 else cplx(G["resp"][lab][k][()]))
                           for k in G["resp"][lab]} for lab in G["resp"]}
        o["delta1"] = cplx(G["delta1"][()])
        out[Lam] = o
    return out, nvec, KF * np.sqrt((nvec ** 2).sum(1)), {"pk_k": h["pk_k"][()], "pk_P": h["pk_P"][()]} if "pk_k" in h else None


def load_stress2(cfg, model, seed, sign, filt="sharp"):
    f = os.path.join(SCR, cfg, "tl", f"stress_{filt}_{model}_L{seed}_s{sign:+d}.h5")
    if not os.path.exists(f):
        return None
    h = h5py.File(f, "r"); out = {}
    for gD in h:
        if not gD.startswith("D"):
            continue
        for gL in h[gD]:
            g = h[gD][gL]
            out[(round(float(gD[1:]), 3), round(float(gL[1:]), 3))] = dict(
                tau=cplx(g["kin"][()]).T + cplx(g["grav"][()]).T, delta=cplx(g["delta"][()]), P=cplx(g["pi"][()]).T,
                curlv=cplx(g["curlv"][()]).T, nmin=float(g.attrs.get("nmin", np.nan)))
    return out


def columns(O, D, s):
    """Per label: (stress (M,6), density (M,), momentum (M,3)) model columns."""
    cols = {}
    for lab in LABELS:
        tau = 0
        for name, n, l in T.OPS:
            if l == lab:
                tau = tau + D ** T.DPOW[name] * s ** T.PAR[name] * O[name]
        r = O["resp"][lab]
        if lab == "lapd1":
            dens = D ** 5 * s * r["d3"]; mom = 0 * r["p1"]
        else:
            dens = D ** 3 * s * r["d1"] + D ** 4 * r["d2"] + D ** 5 * s * r["d3"]
            mom = -(D ** 2 * s * r["p1"] + D ** 3 * r["p2"])
        cols[lab] = (tau, dens, mom)
    return cols


def lsq(blocks, labels):
    """blocks: list of (y (N,), X dict lab→(N,)) already whitened; returns real coefficients."""
    A = np.concatenate([np.stack([X[l] for l in labels], 1) for _, X in blocks])
    b = np.concatenate([y for y, _ in blocks])
    Ar = np.concatenate([A.real, A.imag]); br = np.concatenate([b.real, b.imag])
    p, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    sv = np.linalg.svd(Ar, compute_uv=False)
    return p, sv[0] / sv[-1]


def assemble(kind, data, OPS_, D, Lam, kmask, w):
    """Whitened (y, X) blocks for one observable over all realizations."""
    blocks = []
    for (s, sg), (full, ir) in data.items():
        cols = columns(OPS_[s][Lam], D, sg)
        f = full[(D, Lam)]; i_ = ir[(D, Lam)]
        if kind == "tau":
            y = ((f["tau"] - i_["tau"])[kmask] * np.sqrt(T.COMPW)[None, :]).ravel() / w
            X = {l: (cols[l][0][kmask] * np.sqrt(T.COMPW)[None, :]).ravel() / w for l in LABELS}
        elif kind == "dens":
            y = (f["delta"] - i_["delta"])[kmask] / w
            X = {l: cols[l][1][kmask] / w for l in LABELS}
        else:
            y = (f["P"] - i_["P"])[kmask].ravel() / w
            X = {l: cols[l][2][kmask].ravel() / w for l in LABELS}
        blocks.append((y, X))
    return blocks


def run(cfg, models, kfit):
    seeds = [1, 2, 3, 4]
    OPS_ = {}; pk = None
    for s in seeds:
        try:
            OPS_[s], nvec, K, pk_ = load_ops2(s); pk = pk or pk_
        except OSError:
            pass
    seeds = sorted(OPS_)
    res = {}
    for model in models:
        for Lam in LAMBDAS:
            data = {}
            for s in seeds:
                for sg in (1, -1):
                    fu = load_stress2(cfg, model, s, sg); ir = load_stress2(IRCFG[Lam], model, s, sg)
                    if fu is None or ir is None or Lam not in OPS_[s]:
                        continue
                    data[(s, sg)] = (fu, ir)
            if not data:
                continue
            Ds = sorted({D for (D, l) in next(iter(data.values()))[0] if l == Lam} &
                        {D for (D, l) in next(iter(data.values()))[1] if l == Lam})
            kmask = (K > 0) & (K <= min(kfit, 0.6 * Lam))
            for D in Ds:
                rec = {"nmin_full": min(v[0][(D, Lam)]["nmin"] for v in data.values())}
                fits = {}
                # whitening scale per observable: rms of the measured difference (first pass)
                for kind in ("tau", "dens", "mom"):
                    b0 = assemble(kind, data, OPS_, D, Lam, kmask, 1.0)
                    w = np.sqrt(np.mean(np.concatenate([np.abs(y) ** 2 for y, _ in b0]))) + 1e-30
                    blocks = assemble(kind, data, OPS_, D, Lam, kmask, w)
                    p, cond = lsq(blocks, LABELS)
                    resid = np.concatenate([y - sum(p[i] * X[l] for i, l in enumerate(LABELS)) for y, X in blocks])
                    tot = np.concatenate([y for y, _ in blocks])
                    fits[kind] = dict(c=dict(zip(LABELS, p.tolist())), cond=float(cond),
                                      frac_resid=float(np.sum(np.abs(resid) ** 2) / np.sum(np.abs(tot) ** 2)), w=float(w))
                    fits[kind]["blocks"] = blocks
                joint = [b for kind in ("tau", "dens", "mom") for b in fits[kind].pop("blocks")]
                p, cond = lsq(joint, LABELS)
                fits["joint"] = dict(c=dict(zip(LABELS, p.tolist())), cond=float(cond))
                for kind, f in fits.items():
                    f["eff"] = T.to_effective(f["c"]); f["lin"] = f["c"]["d1"] + f["c"]["r1"]
                rec["fits"] = fits
                if pk is not None:
                    rec["I_sharp"] = T.I_box(Lam, pk["pk_k"], pk["pk_P"], 1.6) if False else float(I_sharp(Lam, pk))
                res[f"{model}|L{Lam:.3f}|D{D:.3f}"] = rec
                print(f"{model:5s} Λ={Lam:.2f} D={D:.2f}  c̄_δ1+c̄_r1: stress {fits['tau']['lin']:7.2f}  dens {fits['dens']['lin']:7.2f}"
                      f"  mom {fits['mom']['lin']:7.2f}  joint {fits['joint']['lin']:7.2f}   resid(τ,δ,P) = "
                      f"{fits['tau']['frac_resid']:.3f} {fits['dens']['frac_resid']:.3f} {fits['mom']['frac_resid']:.3f}", flush=True)
    return res


def I_sharp(Lam, pk, kuv=1.6, N=256):
    n = np.arange(-N // 2 + 1, N // 2)
    n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
    n2 = n2[(n2 > 0) & (KF * np.sqrt(n2) < kuv) & (KF * np.sqrt(n2) >= Lam)]
    u, c = np.unique(n2, return_counts=True); q = KF * np.sqrt(u)
    P = np.exp(np.interp(np.log(q), np.log(pk["pk_k"]), np.log(pk["pk_P"])))
    return (c * P / q ** 2).sum() / L ** 3


if __name__ == "__main__":
    cfg = sys.argv[1]
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pm", "lpt3", "lpt5", "lpt7"]
    kfit = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
    res = run(cfg, models, kfit)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"tl_fit2_{cfg}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", out)
