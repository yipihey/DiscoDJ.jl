"""Reconstruction of Carrasco, Hertzberg & Senatore (2012) Fig. 3 from their equations, and the same
running prediction for this study's setup (EdS, Planck18 shape, sharp k-splits).

    python3 chs/chs_fig3.py            → chs/chs_running.json  (+ prints a comparison with the digitized curves)

Ingredients (all from the paper):
  * Gaussian smoothing W_Λ(k) = exp(−k²/2Λ²) of the fields (Eq. 14); loops use P_{11,l}(k,Λ) = W_Λ² P_lin (Eq. 49).
    Λ is a wavenumber; the real-space Gaussian width is 1/Λ (Λ = 1/3 h/Mpc ↔ σ_x = 3 Mpc/h).
  * One-loop P22, P13 in the App. C approximation (EdS kernels × ΛCDM growth, Eq. 82; percent-level
    agreement with their exact time dependence, their Fig. 7).
  * Counterterm (Eq. 83):  P_13,c = −(2/9) c²_comb(a0) D(a0)² / (ℋ0² D'(a0)² a0²) k² P_11,l
                                   = −(2/9) c² k² P_11,l / (ℋ0² f0²)        (a0 = 1, D(a0) = 1, f = dlnD/dlna).
  * c²(Λ = ∞) from matching P11 + P22 + P13 + P13,c to the non-linear CAMB (halofit) spectrum at k_ren.
  * Running (Eq. 62):  c²(Λ) = c²(∞) + lim_{k→0} [P13(k,∞) − P13(k,Λ)] / (−(2/9) k² P_11,l(k,Λ)/(ℋ0² f0²)),
    evaluated at k_ext = 0.01 h/Mpc as in their figure.
Cosmology (their footnote 8 / Sec. 5): Ω_m = 0.25, Ω_Λ = 0.75, Ω_b = 0.04, h = 0.7, n_s = 1, σ8 = 0.8, z = 0.

Response convention of this study:  Δδ(k) = R δ_L,  R = −C2 k².  The counterterm of Eq. 83 corresponds to
R = P_13,c /(2P) = −c² k² /(9 ℋ0² f0²), i.e.   C2 = c² / (9 ℋ0² f0²)   (a length², ℋ0 = 100 km/s/(Mpc/h)).
"""
import os, json
import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = os.path.dirname(os.path.abspath(__file__))
C_LIGHT = 299792.458                     # km/s
H0H = 100.0                              # ℋ0 = H0 in km/s/(Mpc/h)


# ── linear / non-linear spectra ─────────────────────────────────────────────────────────────────
def chs_spectra(kmax=60.0):
    import camb
    h = 0.7; Om, Ob = 0.25, 0.04
    p = camb.CAMBparams()
    p.set_cosmology(H0=100 * h, ombh2=Ob * h ** 2, omch2=(Om - Ob) * h ** 2, mnu=0.0, omk=0.0)
    p.InitPower.set_params(ns=1.0, As=2e-9)
    p.set_matter_power(redshifts=[0.0], kmax=kmax * h * 1.05)
    p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p)
    s8 = r.get_sigma8_0()
    p.InitPower.set_params(ns=1.0, As=2e-9 * (0.8 / s8) ** 2)
    p.set_matter_power(redshifts=[0.0], kmax=kmax * h * 1.05)
    p.NonLinear = camb.model.NonLinear_both
    p.NonLinearModel.set_params(halofit_version="original")      # CAMB of mid-2012: Smith et al. (2003)
    r = camb.get_results(p)
    kh, z, Pl = r.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    kh2, z2, Pnl = r.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    return kh, Pl[0], Pnl[0], r.get_sigma8_0()


def growth_rate_lcdm(Om=0.25, a=1.0):
    """f = dlnD/dlna for flat ΛCDM (integral solution of the growing mode)."""
    from scipy.integrate import quad
    E = lambda x: np.sqrt(Om / x ** 3 + 1 - Om)
    I = lambda x: quad(lambda y: 1 / (y * E(y)) ** 3, 0, x)[0]
    D = lambda x: 2.5 * Om * E(x) * I(x)
    eps = 1e-4
    return (np.log(D(a * (1 + eps))) - np.log(D(a * (1 - eps)))) / (np.log(1 + eps) - np.log(1 - eps))


class Spectrum:
    def __init__(self, k, P):
        self.lk, self.lP = np.log(k), np.log(P)
        self.n_lo = (self.lP[1] - self.lP[0]) / (self.lk[1] - self.lk[0])
        self.n_hi = (self.lP[-1] - self.lP[-2]) / (self.lk[-1] - self.lk[-2])

    def __call__(self, k):
        lk = np.log(np.asarray(k, float))
        out = np.interp(lk, self.lk, self.lP)
        out = np.where(lk < self.lk[0], self.lP[0] + self.n_lo * (lk - self.lk[0]), out)
        out = np.where(lk > self.lk[-1], self.lP[-1] + self.n_hi * (lk - self.lk[-1]), out)
        return np.exp(out)


# ── one-loop kernels (App. C, Eq. 82) ─────────────────────────────────────────────────────────────
def _p13_bracket(r):
    r = np.asarray(r, float)
    out = np.empty_like(r)
    mid = (r > 0.02) & (r < 40)
    x = r[mid]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[mid] = (3 / x ** 3 * (x ** 2 - 1) ** 3 * (7 * x ** 2 + 2) * np.log(np.abs((x + 1) / (1 - x)))
                    - 42 * x ** 4 + 100 * x ** 2 + 12 / x ** 2 - 158)
    lo = r <= 0.02; x = r[lo]           # small-r series: −168 + 928/5 r² − 4512/35 r⁴
    out[lo] = -168 + 928 / 5 * x ** 2 - 4512 / 35 * x ** 4
    hi = r >= 40; x = r[hi]             # large-r series (checked against 50-digit evaluation):
    out[hi] = -488 / 5 + 96 / 5 / x ** 2 - 160 / 21 / x ** 4      # −488/5 + 96/(5r²) − 160/(21r⁴)
    near = np.abs(r - 1) < 1e-6
    out[near] = -88.0                    # finite limit at r = 1 (log term vanishes with (r²−1)³)
    return out


def p13(k, P, W2=lambda q: 1.0, rmin=1e-5, rmax_k=200.0, n=20000):
    """P̃13(k) = k³/(1008π²) P(k) ∫ dr B(r) P(kr) W²(kr)   (sourcing spectrum smoothed by W²)."""
    lr = np.linspace(np.log(rmin), np.log(rmax_k / k), n)
    r = np.exp(lr)
    # refine around r = 1
    r = np.unique(np.concatenate([r, 1 + np.linspace(-0.05, 0.05, 4001)]))
    r = r[(r > 0) & (np.abs(r - 1) > 1e-9)]
    f = _p13_bracket(r) * P(k * r) * W2(k * r)
    return k ** 3 / (1008 * np.pi ** 2) * P(k) * np.trapz(f, r)


def p22(k, P, W2=lambda q: 1.0, nr=3000, nx=200, rmax_k=200.0):
    lr = np.linspace(np.log(1e-4), np.log(rmax_k / k), nr); r = np.exp(lr)
    r = np.unique(np.concatenate([r, 1 + np.linspace(-0.1, 0.1, 801)]))
    x, w = leggauss(nx)
    R, X = np.meshgrid(r, x, indexing="ij")
    d2 = R ** 2 - 2 * R * X + 1
    d2 = np.maximum(d2, 1e-12)
    num = (-10 * R * X ** 2 + 3 * R + 7 * X) ** 2
    f = num / d2 ** 2 * P(k * R) * W2(k * R) * P(k * np.sqrt(d2)) * W2(k * np.sqrt(d2))
    inner = (f * w[None, :]).sum(1)
    return k ** 3 / (392 * np.pi ** 2) * np.trapz(inner, r)


def gaussW2(L):
    return (lambda q: np.exp(-np.asarray(q) ** 2 / L ** 2)) if np.isfinite(L) else (lambda q: 1.0)


def sharpW2(ks, kuv=np.inf):
    return lambda q: ((np.asarray(q) < ks) | (np.asarray(q) >= kuv)).astype(float)


def main():
    out = {}
    # ── (A) CHS cosmology: reconstruct Fig. 3 ────────────────────────────────────────────────────
    k, Pl, Pnl, s8 = chs_spectra()
    P = Spectrum(k, Pl); PN = Spectrum(k, Pnl)
    f0 = growth_rate_lcdm(0.25)
    ct = lambda kk: -(2 / 9) * kk ** 2 * P(kk) / (H0H ** 2 * f0 ** 2)          # P13,c per unit c² (km/s)⁻²
    c2inf = {}
    for kren in (0.10, 0.16, 0.18):
        resid = PN(kren) - P(kren) - p22(kren, P) - p13(kren, P)
        c2inf[kren] = resid / ct(kren) / C_LIGHT ** 2
        print(f"k_ren = {kren}:  P22/P = {p22(kren, P)/P(kren):.4f}  P13/P = {p13(kren, P)/P(kren):.4f}"
              f"  (Pnl/P − 1) = {PN(kren)/P(kren)-1:.4f}   c²(∞) = {c2inf[kren]*1e6:.3f} × 10⁻⁶ c²")
    kext = 0.01
    Ls = np.concatenate([np.linspace(0.1, 1.0, 46), [1.5, 2.0, 3.0, 5.0]])
    p13inf = p13(kext, P)
    run = []
    for L in Ls:
        d = (p13inf - p13(kext, P, gaussW2(L))) / (ct(kext) * np.exp(-kext ** 2 / L ** 2)) / C_LIGHT ** 2
        run.append(d)
    run = np.array(run)
    # UV-limit check: Δc² = (61/105)(9/2) ℋ0² f0² Δσ_v²,  Δσ_v² = (1/6π²)∫ P (1 − W²)
    q = np.exp(np.linspace(np.log(1e-4), np.log(60), 20000))
    uv = [(61 / 105) * 4.5 * H0H ** 2 * f0 ** 2 * np.trapz(P(q) * (1 - np.exp(-q ** 2 / L ** 2)), q) / (6 * np.pi ** 2) / C_LIGHT ** 2 for L in Ls]
    out["chs"] = {"cosmology": {"Om": 0.25, "Ob": 0.04, "h": 0.7, "ns": 1.0, "sigma8": s8, "f0": f0,
                                "spectra": "CAMB linear + halofit (original, Smith et al. 2003)"},
                  "Lambda": Ls.tolist(), "running": run.tolist(), "running_uv_limit": list(map(float, uv)),
                  "c2_inf": {str(k_): float(v) for k_, v in c2inf.items()}, "k_ext": kext,
                  "C2_per_c2": float(C_LIGHT ** 2 / (9 * H0H ** 2 * f0 ** 2)),
                  "note": "c² in units of c_light²; C2 = c²/(9 ℋ0² f0²) [(Mpc/h)²] is the long-mode density response"}
    dig = json.load(open(os.path.join(HERE, "chs2012_fig3_digitized.json")))
    for key, kren in (("kren_0.10", 0.10), ("kren_0.18", 0.18)):
        pts = np.array(dig[key]); pred = c2inf[kren] + np.interp(pts[:, 0], Ls, run)
        print(f"{key}: reconstruction / digitized  (Λ = 0.1 … 1): median {np.median(pred / pts[:, 1]):.3f},"
              f" range {np.min(pred / pts[:, 1]):.3f} – {np.max(pred / pts[:, 1]):.3f}")
        dg = np.array(dig[key]); dshape = dg[:, 1] - np.interp(1.0, dg[:, 0], dg[:, 1])
        pshape = np.interp(dg[:, 0], Ls, run) - np.interp(1.0, Ls, run)
        print(f"   running shape c²(Λ) − c²(1): digitized {dshape[0]*1e6:.3f} at Λ = {dg[0,0]:.3f}, reconstructed {pshape[0]*1e6:.3f}"
              f";  at Λ = 1/3: digitized {np.interp(1/3, dg[:,0], dshape)*1e6:.3f}, reconstructed {np.interp(1/3, dg[:,0], pshape)*1e6:.3f}")

    # ── (B) this study's setup: EdS (f = 1), Planck18 EH shape, sharp splits inside the band < k_uv ───
    ours = json.load(open(os.path.join(HERE, "planck18_pk.json")))
    Po = Spectrum(np.array(ours["k"]), np.array(ours["Pk"]))
    kuv = 1.2
    ks = np.array([0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.1])
    # one-loop (EdS) response of the long-mode density to the band [ks, kuv):  C2 = −[P13(full) − P13(IR)]/(2k²P)
    def C2_band(lo, hi, kk=kext):
        W2 = lambda qq: ((np.asarray(qq) >= lo) & (np.asarray(qq) < hi)).astype(float)
        return -p13(kk, Po, W2) / (2 * kk ** 2 * Po(kk))
    C2pt = np.array([C2_band(s, kuv) for s in ks])
    C2pt_k04 = np.array([C2_band(s, kuv, 0.04) if s > 0.06 else np.nan for s in ks])
    q = np.exp(np.linspace(np.log(1e-4), np.log(kuv), 20000))
    sv2 = lambda lo, hi: np.trapz(np.where((q >= lo) & (q < hi), Po(q), 0.0), q) / (6 * np.pi ** 2)
    # Gaussian-Λ equivalent of a sharp split: equal removed σ_v² (the only Λ dependence of Eq. 62 as k_ext → 0)
    def lam_eq(s, PP, upper):
        target = np.trapz(np.where((q >= s) & (q < upper), PP(q), 0.0), q)
        from scipy.optimize import brentq
        g = lambda L: np.trapz(PP(q) * (1 - np.exp(-q ** 2 / L ** 2)) * (q < upper), q) - target
        return brentq(g, 0.01, 50)
    out["ours"] = {"k_split": ks.tolist(), "k_uv": kuv, "C2_pt_1loop_kext0.01": C2pt.tolist(),
                   "C2_pt_1loop_k0.04": C2pt_k04.tolist(),
                   "C2_pt_uv_limit": [(61 / 210) * sv2(s, kuv) for s in ks],
                   "kinematic_sigma_v2_over_2": [0.5 * sv2(s, kuv) for s in ks],
                   "Lambda_eq": [lam_eq(s, Po, kuv) for s in ks],
                   "c2_per_C2_EdS": float(9 * H0H ** 2 / C_LIGHT ** 2),
                   "note": "EdS, D = 1; C2 in (Mpc/h)²; c²_eff = 9 ℋ0² C2 (f = 1)"}
    # the Gaussian Λ ↔ sharp k_split dictionary in the CHS cosmology too (no k_uv)
    q2 = np.exp(np.linspace(np.log(1e-4), np.log(60), 20000))
    def ks_of_lambda(L, PP):
        tot = np.trapz(PP(q2) * (1 - np.exp(-q2 ** 2 / L ** 2)), q2)
        cum = np.concatenate([[0], np.cumsum(0.5 * (PP(q2)[1:] + PP(q2)[:-1]) * np.diff(q2))])
        above = cum[-1] - cum
        return float(np.interp(tot, above[::-1], q2[::-1]))
    out["dictionary"] = {"Lambda": [0.1, 1 / 6, 0.2, 0.25, 1 / 3, 0.5, 1.0],
                         "k_split_equiv_chs": [ks_of_lambda(L, P) for L in [0.1, 1 / 6, 0.2, 0.25, 1 / 3, 0.5, 1.0]],
                         "k_split_equiv_ours": [ks_of_lambda(L, Po) for L in [0.1, 1 / 6, 0.2, 0.25, 1 / 3, 0.5, 1.0]],
                         "gaussian_width_Mpc_h": [1 / L for L in [0.1, 1 / 6, 0.2, 0.25, 1 / 3, 0.5, 1.0]]}
    print("Λ → sharp-split equivalent (equal removed σ_v²):", json.dumps(out["dictionary"], indent=None))
    print("ours: k_split, Λ_eq, one-loop C2 (k→0), UV-limit, kinematic σv²/2:")
    for i, s in enumerate(ks):
        print(f"  {s:5.2f}  {out['ours']['Lambda_eq'][i]:6.3f}  {C2pt[i]:7.3f}  {out['ours']['C2_pt_uv_limit'][i]:7.3f}  {out['ours']['kinematic_sigma_v2_over_2'][i]:7.3f}")
    json.dump(out, open(os.path.join(HERE, "chs_running.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
