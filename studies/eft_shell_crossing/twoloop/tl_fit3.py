"""Experiment 3 production fit: paired differences, explicit odd/even separation, multi-epoch windows (twoloop/README.md).

    python tl_fit3.py <A> <B|IR> [models=pm,lpt3,lpt5,lpt7] [kfit=0.12]

A − B: two members with identical long modes, coarse-grained identically with a sharp cutoff Λ.
   B = IR  : IR member with ICs cut at Λ → the stress of all modes above Λ (the running c̄_α(Λ))
   B = cfg : another UV cut (e.g. tl_full − tl_uv10) → the stress of a far-UV band only (analytic UV test, no IR needed)

±L pairs split every difference exactly into odd (orders 1, 3 and the odd stochastic part) and even (order 2 and the
pure short × short stochastic part).  Model (linear in the 18 c̄_α):
   stress   odd : Σ D^{n+2} C^{(n)}, n = 1, 3 (+ D⁵ δ^K∂²δ₁)      even : D⁴ C^{(2)}
   density  odd : D³ δ̃¹ + D⁵ δ̃³                                  even : D⁴ δ̃²                 (Eqs. 2.57, 2.59, 2.61)
   momentum odd : −D² π̃¹ (leading only)                           even : −D³ π̃²                (Eqs. 2.58, 2.60)
Fits use normal equations accumulated per (seed, D, observable), so seeds and epochs can be recombined cheaply: a window of
epochs with constant c̄ separates D³ from D⁵; delete-one-seed jackknife gives the errors.
"""
import os, sys, json
import numpy as np
import tl_fit as T
from tl_fit2 import load_ops2, load_stress2, IRCFG, LABELS, KF, L

LAMBDAS = [0.1, 0.15, 0.2, 0.3]
SEEDS = [1, 2, 3, 4]
W6 = np.sqrt(T.COMPW)[None, :]
LIN = ["r1", "d1"]
EVEN7 = ["r1", "r2", "d1", "d2", "rr1", "rd1", "dd1"]


def columns(O, D):
    """odd/even model columns, per label, for s_L = +1."""
    c = {}
    for lab in LABELS:
        to = 0; te = 0
        for name, n, l in T.OPS:
            if l != lab:
                continue
            v = D ** T.DPOW[name] * O[name]
            if T.PAR[name] == 1:
                to = to + v
            else:
                te = te + v
        r = O["resp"][lab]
        M = O["C1_r1"].shape[0]
        to = to if np.ndim(to) else np.zeros((M, 6), complex); te = te if np.ndim(te) else np.zeros((M, 6), complex)
        if lab == "lapd1":
            do = D ** 5 * r["d3"]; de = np.zeros(M, complex); po = np.zeros((M, 3), complex); pe = po
        else:
            do = D ** 3 * r["d1"] + D ** 5 * r["d3"]; de = D ** 4 * r["d2"]; po = -D ** 2 * r["p1"]; pe = -D ** 3 * r["p2"]
        c[lab] = dict(tau_odd=to, tau_even=te, dens_odd=do, dens_even=de, mom_odd=po, mom_even=pe)
    return c


class NE:
    """Normal-equation blocks keyed (seed, D, observable)."""
    def __init__(self):
        self.A = {}; self.b = {}; self.y2 = {}

    def add(self, key, y, X):
        Xr = np.concatenate([X.real, X.imag]); yr = np.concatenate([y.real, y.imag])
        self.A[key] = Xr.T @ Xr; self.b[key] = Xr.T @ yr; self.y2[key] = float(yr @ yr)

    def solve(self, sel, labels, weight):
        idx = [LABELS.index(l) for l in labels]
        A = 0; b = 0
        for k in self.A:
            if sel(k):
                w = weight(k)
                A = A + w * self.A[k][np.ix_(idx, idx)]; b = b + w * self.b[k][idx]
        s = np.sqrt(np.maximum(np.diag(A), 1e-300))                        # precondition (columns differ by orders of magnitude)
        Ah = A / np.outer(s, s); bh = b / s
        p = np.linalg.lstsq(Ah, bh, rcond=1e-10)[0] / s
        return p, float(np.linalg.cond(Ah))

    def chi2(self, key, p, labels):
        idx = [LABELS.index(l) for l in labels]
        return self.y2[key] - 2 * p @ self.b[key][idx] + p @ self.A[key][np.ix_(idx, idx)] @ p


def build(A, B, model, Lam, OP, K, kfit, nmin_floor=0.02):
    """Accumulate normal equations and per-k-bin residual inputs for A − B."""
    kmask = (K > 0) & (K <= min(kfit, 0.6 * Lam))
    cB = IRCFG[Lam] if B == "IR" else B
    ne = NE(); raw = {}; Ds = set()
    for s in SEEDS:
        if s not in OP or Lam not in OP[s]:
            continue
        mem = {}
        for sg in (1, -1):
            a = load_stress2(A, model, s, sg); b = load_stress2(cB, model, s, sg)
            if a is None or b is None:
                mem = None; break
            mem[sg] = (a, b)
        if mem is None:
            continue
        for key in mem[1][0]:
            D, l = key
            if l != Lam or key not in mem[1][1]:
                continue
            nm = min(mem[sg][i][key]["nmin"] for sg in (1, -1) for i in (0, 1))
            if not nm > nmin_floor:                                        # 1/n_Λ of the sharp field is singular
                continue
            Ds.add(D)
            dlt = {sg: {f: mem[sg][0][key][f] - mem[sg][1][key][f] for f in ("tau", "delta", "P")} for sg in (1, -1)}
            odd = {f: (dlt[1][f] - dlt[-1][f]) / 2 for f in dlt[1]}; even = {f: (dlt[1][f] + dlt[-1][f]) / 2 for f in dlt[1]}
            c = columns(OP[s][Lam], D)
            def stack(cn, m, w=None):
                return np.stack([((c[l][cn][m] * W6) if c[l][cn].ndim == 2 and c[l][cn].shape[1] == 6 else c[l][cn][m]).ravel() for l in LABELS], 1)
            ne.add((s, D, "tau_odd"), (odd["tau"][kmask] * W6).ravel(), stack("tau_odd", kmask))
            ne.add((s, D, "tau_even"), (even["tau"][kmask] * W6).ravel(), stack("tau_even", kmask))
            ne.add((s, D, "dens_odd"), odd["delta"][kmask].ravel(), stack("dens_odd", kmask))
            ne.add((s, D, "dens_even"), even["delta"][kmask].ravel(), stack("dens_even", kmask))
            ne.add((s, D, "mom_odd"), odd["P"][kmask].ravel(), stack("mom_odd", kmask))
            ne.add((s, D, "mom_even"), even["P"][kmask].ravel(), stack("mom_even", kmask))
            raw[(s, D)] = dict(odd=odd, even=even, cols=c)
    return ne, raw, sorted(Ds), kmask


def weights(ne, obs):
    """Inverse variance per (D, observable): residual after a per-block fit of the whole basis (noise-dominated blocks)."""
    w = {}
    for k in ne.A:
        if k[2] in obs:
            p, _ = ne.solve(lambda q: q == k, LABELS, lambda q: 1.0)
            w[k] = 1.0 / max(ne.chi2(k, p, LABELS), 1e-300)
    return w


def fit_set(ne, Ds, obs, labels, seeds):
    """Fit over epochs Ds and observables obs; delete-one-seed jackknife."""
    w = weights(ne, obs)
    sel = lambda ss: (lambda k: k[0] in ss and k[1] in Ds and k[2] in obs)
    p, cond = ne.solve(sel(seeds), labels, lambda k: w[k])
    if len(seeds) > 1:
        jk = [ne.solve(sel([s for s in seeds if s != s0]), labels, lambda k: w[k])[0] for s0 in seeds]
        n = len(jk); err = np.sqrt((n - 1) / n * ((np.array(jk) - p) ** 2).sum(0))
    else:
        jk = []; err = np.full_like(p, np.nan)
    return p, err, cond, jk


def eff_with_err(p, jk, labels):
    e = T.to_effective(dict(zip(labels, p)))
    keys = T.GAMMA + ["lapd1"]
    if jk:
        v = np.array([[T.to_effective(dict(zip(labels, q)))[g] for g in keys] for q in jk]); n = len(jk)
        err = np.sqrt((n - 1) / n * ((v - np.array([e[g] for g in keys])) ** 2).sum(0))
    else:
        err = np.full(len(keys), np.nan)
    return {g: [float(e[g]), float(err[i])] for i, g in enumerate(keys)}


def residual_spectra(raw, ne, labels, p, K, kmask):
    """Stochastic spectra after projection: trace and longitudinal stress, odd and even, per k bin (L³ normalised)."""
    return None


def I_band(pk, kmin, kmax, N=256, mod=None):
    n = np.arange(-N // 2 + 1, N // 2)
    n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
    n2 = n2[(n2 > 0) & (KF * np.sqrt(n2) < kmax) & (KF * np.sqrt(n2) >= kmin)]
    u, c = np.unique(n2, return_counts=True); q = KF * np.sqrt(u)
    P = np.exp(np.interp(np.log(q), np.log(pk["pk_k"]), np.log(pk["pk_P"])))
    if mod:
        P0 = np.exp(np.interp(np.log(mod[0]), np.log(pk["pk_k"]), np.log(pk["pk_P"])))
        P = np.where(q > mod[0], P0 * (q / mod[0]) ** mod[1], P)
    return float((c * P / q ** 2).sum() / L ** 3)


def run(A, B, models, kfit):
    OP = {}; K = None; pk = None
    for s in SEEDS:
        try:
            OP[s], nvec, K, pk_ = load_ops2(s); pk = pk or pk_
        except (OSError, KeyError):
            pass
    if not OP:
        raise SystemExit("no operators yet")
    KUV = {"tl_full": 1.6, "tl_uv10": 1.0, "tl_uv06": 0.6, "tl_div": 1.6, "tl_div10": 1.0}
    mod = (0.3, -1.0) if "div" in A else None
    res = {}
    for model in models:
        for Lam in LAMBDAS:
            ne, raw, Ds, kmask = build(A, B, model, Lam, OP, K, kfit)
            if not Ds:
                continue
            seeds = sorted({k[0] for k in ne.A})
            Ival = I_band(pk, Lam if B == "IR" else KUV[B], KUV[A], mod=mod)
            rec = dict(I=Ival, seeds=seeds, Ds=Ds, nmodes=int(kmask.sum()), per_epoch={}, window={})
            for D in Ds:
                r = {}
                # the leading coefficient, three ways (closure of stress, density, momentum)
                for tag, obs in (("stress", ("tau_odd",)), ("dens", ("dens_odd",)), ("mom", ("mom_odd",))):
                    p, e, cond, jk = fit_set(ne, [D], obs, LIN, seeds)
                    # at linear order only c_r1 + c_d1 enters the density/longitudinal momentum
                    pl = fit_lin_sum(ne, [D], obs, seeds)
                    r[tag] = dict(r1=[p[0], e[0]], d1=[p[1], e[1]], sum=pl)
                res_frac = {}
                for ob in ("tau_odd", "tau_even", "dens_odd", "dens_even", "mom_even"):
                    p, *_ = ne.solve(lambda k: k[1] == D and k[2] == ob, LABELS, lambda k: 1.0)
                    tot = sum(ne.y2[k] for k in ne.A if k[1] == D and k[2] == ob)
                    chi = sum(ne.chi2(k, p, LABELS) for k in ne.A if k[1] == D and k[2] == ob)
                    res_frac[ob] = chi / max(tot, 1e-300)
                r["resid_frac"] = res_frac
                rec["per_epoch"][f"{D:.3f}"] = r
            # windows of epochs with constant c̄: stress, density, joint; all 18 coefficients and the effective 7 + 1
            for wname, dmax in (("D<=0.3", 0.3), ("D<=0.5", 0.5), ("D<=0.8", 0.8), ("all", 9.0)):
                dw = [D for D in Ds if D <= dmax]
                if len(dw) < 2:
                    continue
                rw = {}
                for tag, obs in (("stress", ("tau_odd", "tau_even")), ("dens", ("dens_odd", "dens_even")),
                                 ("joint", ("tau_odd", "tau_even", "dens_odd", "dens_even", "mom_even"))):
                    p, e, cond, jk = fit_set(ne, dw, obs, LABELS, seeds)
                    rw[tag] = dict(c={l: [float(p[i]), float(e[i])] for i, l in enumerate(LABELS)}, cond=cond,
                                   eff=eff_with_err(p, jk, LABELS))
                rec["window"][wname] = rw
            res[f"{model}|L{Lam:.3f}"] = rec
            report(model, Lam, rec)
    return res


def fit_lin_sum(ne, Ds, obs, seeds):
    """Single coefficient c̄_r1 + c̄_d1: one-parameter fit using the sum column r1 + d1 scaled to c_r1 = c_d1 = c/2."""
    w = weights(ne, obs); i1, i2 = LABELS.index("r1"), LABELS.index("d1")
    def one(ss):
        a = 0; b = 0
        for k in ne.A:
            if k[0] in ss and k[1] in Ds and k[2] in obs:
                M = ne.A[k]; a += w[k] * (M[i1, i1] + 2 * M[i1, i2] + M[i2, i2]) / 4; b += w[k] * (ne.b[k][i1] + ne.b[k][i2]) / 2
        return b / max(a, 1e-300)
    p = one(seeds)
    if len(seeds) > 1:
        jk = np.array([one([s for s in seeds if s != s0]) for s0 in seeds]); n = len(seeds)
        return [float(p), float(np.sqrt((n - 1) / n * ((jk - p) ** 2).sum()))]
    return [float(p), float("nan")]


def report(model, Lam, rec):
    print(f"\n{model}  Λ = {Lam}  I = {rec['I']:.3f}  (61/70)I = {61/70*rec['I']:.3f}  seeds {rec['seeds']}  modes {rec['nmodes']}", flush=True)
    for D, r in rec["per_epoch"].items():
        print(f"  D={D}  c̄_lin stress {r['stress']['sum'][0]:7.2f}±{r['stress']['sum'][1]:5.2f}  dens {r['dens']['sum'][0]:7.2f}±{r['dens']['sum'][1]:5.2f}"
              f"  mom {r['mom']['sum'][0]:7.2f}±{r['mom']['sum'][1]:5.2f}  | r1 {r['stress']['r1'][0]:6.2f} d1 {r['stress']['d1'][0]:6.2f}"
              f" | resid τo {r['resid_frac']['tau_odd']:.3f} τe {r['resid_frac']['tau_even']:.3f} δo {r['resid_frac']['dens_odd']:.3f}", flush=True)
    for w, rw in rec["window"].items():
        for tag in ("stress", "dens", "joint"):
            e = rw[tag]["eff"]
            print(f"  {w:7s} {tag:6s} eff/I: " + " ".join(f"{g}={e[g][0]/rec['I']:+.2f}±{e[g][1]/rec['I']:.2f}" for g in T.GAMMA) + f"  cond {rw[tag]['cond']:.1e}", flush=True)
    print("  paper κ     : " + " ".join(f"{g}={T.KAPPA[g]:+.2f}" for g in T.GAMMA), flush=True)


if __name__ == "__main__":
    A, B = sys.argv[1], sys.argv[2]
    models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["lpt3", "lpt5", "lpt7", "pm"]
    kfit = float(sys.argv[4]) if len(sys.argv) > 4 else 0.12
    res = run(A, B, models, kfit)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"tl_fit3_{A}_minus_{B}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", out)
