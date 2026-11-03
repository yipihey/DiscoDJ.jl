#!/usr/bin/env python3
"""Does a change of the UV cutoff change the variance at IR scales?  Seed-matched comparison of two spectra sharing their long modes -> results/sigmaM_pair_<A>_<B>.json

    python sigmaM_pair.py <A> <B>

At fixed Eulerian radius and from the same seeds and signs:
   total    = Var_A/Var_B - 1,   added = D^2 (sig_lin,A^2 - sig_lin,B^2)/Var_B (linear power of the extra modes through the window),   response = total - added.
Spherical collapse predicts total for each spectrum with contraction (R_L = R mu^{1/3}) and without.  Errors: jackknife over seeds, after the +- pair mean."""
import glob, json, os, re, sys
import h5py, numpy as np
A, B = sys.argv[1], sys.argv[2]
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing"); here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SPEC = {"tl_full": (1.6, None), "tl_uv10": (1.0, None), "tl_uv06": (0.6, None), "tl_div": (1.6, (0.3, -1.0)), "tl_div10": (1.0, (0.3, -1.0)), "tl_ir010": (0.1, None), "tl_ir015": (0.15, None), "tl_ir020": (0.2, None), "tl_ir030": (0.3, None)}
with h5py.File(os.path.join(D0, "tl_full", "tl", "ops_sharp_L1.h5")) as h:
    pk, pP = h["pk_k"][()], h["pk_P"][()]
n = np.arange(-256, 257); n2a = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel(); KF = 2 * np.pi / 500.
tophat = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
grid = np.exp(np.linspace(np.log(1.5), np.log(400), 400))
def siglin(cfg):
    ku, mod = SPEC[cfg]; n2 = n2a[(n2a > 0) & (KF * np.sqrt(n2a) < ku)]; u, c = np.unique(n2, return_counts=True); q = KF * np.sqrt(u); P = np.exp(np.interp(np.log(q), np.log(pk), np.log(pP)))
    if mod:
        P0 = np.exp(np.interp(np.log(mod[0]), np.log(pk), np.log(pP))); P = np.where(q > mod[0], P0 * (q / mod[0]) ** mod[1], P)
    s = np.array([np.sqrt((c * P * tophat(q * R) ** 2).sum() / 500. ** 3) for R in grid])
    return lambda R: np.exp(np.interp(np.log(R), np.log(grid), np.log(s)))
SA, SB = siglin(A), siglin(B)
th = np.linspace(1e-3, 2 * np.pi - 1e-3, 40000); et = np.linspace(1e-3, 40, 40000)
dL = np.concatenate([(-3 / 5 * (3 / 4 * (np.sinh(et) - et)) ** (2 / 3))[::-1], 3 / 5 * (3 / 4 * (th - np.sin(th))) ** (2 / 3)])
lm = np.log(np.concatenate([(9 / 2 * (np.sinh(et) - et) ** 2 / (np.cosh(et) - 1) ** 3)[::-1], 9 / 2 * (th - np.sin(th)) ** 2 / (1 - np.cos(th)) ** 3]))
o = np.argsort(dL); dL, lm = dL[o], np.maximum.accumulate(lm[o]); LV = np.log(178.0)
def sc_var(S, R, D, contraction):
    x = np.linspace(-6, LV, 8000); mu = np.exp(x); Rl = R * mu ** (1 / 3) if contraction else np.full_like(mu, R)
    nu = np.interp(x, lm, dL) / (D * S(Rl)); w = np.exp(-nu ** 2 / 2) * np.abs(np.gradient(nu, x)) / mu; w[x >= LV - 1e-6] = 0
    m = (w * mu).sum() / w.sum(); return (w * (mu / m - 1) ** 2).sum() / w.sum()
def load(cfg):
    out = {}
    for f in glob.glob(os.path.join(D0, "sigmaM", f"{cfg}_pm_L*_s*_D*.h5")):
        m = re.search(r"_L(\d+)_s([+-]\d)_D([0-9.]+)\.h5$", f)
        with h5py.File(f) as h:
            out[(int(m[1]), int(m[2]), float(m[3]))] = h["mom"][()].T[:, 1]; RE = h["RE"][()]
    return out, RE
VA, RE = load(A); VB, _ = load(B)
Ds = sorted({k[2] for k in VA} & {k[2] for k in VB}); seeds = sorted({k[0] for k in VA if (k in VB)})
res = dict(A=A, B=B, RE=RE.tolist(), epochs={})
print(f"{A} vs {B}, seeds {seeds}.  total = Var_A/Var_B-1, added = linear power of the extra modes, response = total-added (seed-matched).  SC_c: with contraction; SC_n: without.")
print("D    R    | total ± | added | response ± | SC_c total, SC_c response | SC_n total, SC_n response | Var_B")
RS = [r for r in (5, 6.7, 10, 15, 20, 30, 50, 80)]
for D in Ds:
    a = np.array([[0.5 * (VA[(s, 1, D)] + VA[(s, -1, D)]) for s in seeds]])[0]; b = np.array([0.5 * (VB[(s, 1, D)] + VB[(s, -1, D)]) for s in seeds])
    def stat(idx): return a[idx].mean(0) / b[idx].mean(0) - 1
    allv = list(range(len(seeds))); tot = stat(allv); J = np.array([stat([i for i in allv if i != k]) for k in allv]); k = len(seeds); et_ = np.sqrt((k - 1) / k * ((J - J.mean(0)) ** 2).sum(0))
    add = D ** 2 * (SA(RE) ** 2 - SB(RE) ** 2) / b.mean(0)
    rows = []
    for R in RS:
        i = int(np.argmin(abs(RE - R))); Ri = RE[i]
        sa, sb = sc_var(SA, Ri, D, True), sc_var(SB, Ri, D, True); na, nb = sc_var(SA, Ri, D, False), sc_var(SB, Ri, D, False)
        ad_c = D ** 2 * (SA(Ri) ** 2 - SB(Ri) ** 2) / sb; ad_n = D ** 2 * (SA(Ri) ** 2 - SB(Ri) ** 2) / nb
        print(f"{D:.1f} {Ri:5.1f} | {tot[i]:+.4f} {et_[i]:.4f} | {add[i]:+.4f} | {tot[i] - add[i]:+.4f} {et_[i]:.4f} | {sa / sb - 1:+.4f} {sa / sb - 1 - ad_c:+.4f} | {na / nb - 1:+.4f} {na / nb - 1 - ad_n:+.4f} | {b.mean(0)[i]:.4f}")
        rows.append(dict(R=float(Ri), total=float(tot[i]), err=float(et_[i]), added=float(add[i]), response=float(tot[i] - add[i]), sc_c_total=sa / sb - 1, sc_c_resp=sa / sb - 1 - ad_c, sc_n_total=na / nb - 1, sc_n_resp=na / nb - 1 - ad_n, var_b=float(b.mean(0)[i])))
    res["epochs"][f"{D:.3f}"] = rows
json.dump(res, open(os.path.join(here, f"sigmaM_pair_{A}_{B}.json"), "w"), indent=0, default=float)
