"""ΛCDM counterterm calibration: c_s(z) of the one-loop EFT and of the sheet forward model, and the zero-parameter
1 % test of the matter power spectrum.

    python scripts/lcdm_cs.py                     (reads $EFTSC_SCRATCH/lcdm_*/ and results/lcdm_full_pt.h5)

Conventions (as in the EdS study): δ_k are discrete Fourier coefficients (rfft/N³) of the stored mode set; δ_L is the
fixed-amplitude linear field at D = 1 (identical in every model, sign s_L).  Per k bin (width k_f) and epoch D:

    R_X(k) = Σ Re(δ_X s_L δ_L*) / (D Σ|δ_L|²) − 1        (propagator response, odd in δ_L)
    Q_X(k) = Σ |δ_X|² / (D² Σ|δ_L|²) − 1                 (P_X/P_L − 1; fixed amplitudes: Σ|δ_L|² ∝ Σ P_L exactly)

Perturbation theory is evaluated on the same box: P22 from the realization's own δ⁽²⁾ (lcdm_pt.jl), P13 and the
propagator from the exact discrete sum over the box modes 0 < |q| < 1.6 h/Mpc (fixed amplitudes ⇒ the realized
P(q) is the input spectrum); EdS kernels × D_ΛCDM.  BAO IR resummation (wiggle/no-wiggle, Σ²) is applied as a
continuum correction.  The counterterm is ΔP/P = −2 C₂ k² (C₂ a comoving length², h⁻²Mpc²), propagator −C₂ k².
"""
import os, sys, json, glob
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "chs"))
import chs_fig3 as C                                        # one-loop kernels (verified in the EdS study)

SCR = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
RES = os.path.join(HERE, "..", "results")
L = 500.0; KF = 2 * np.pi / L; KCUT = 1.6; S8_FID = 0.8105
cplx = lambda x: x["r"] + 1j * x["i"] if x.dtype.names else x


# ── data ──────────────────────────────────────────────────────────────────────────────────────────
def load(name, model):
    """{(seed, sign): dict(dens (M,E) or (M,n,E), epochs)} for model 'pm' or 'lpt'."""
    out = {}
    for f in sorted(glob.glob(os.path.join(SCR, name, f"{model}_IR_L*_S0_s*.h5"))):
        h = h5py.File(f, "r")
        r = dict(dens=cplx(h["dens"][()]), epochs=h["epochs"][()], nvec=h["nvec"][()], inv=h["inv"][()])
        if "dens_x" in h:      # exact direct-sum density (k ≤ k_exact) substituted into the CIC modes
            idx = {tuple(v): i for i, v in enumerate(r["nvec"].T)}
            j = np.array([idx[tuple(v)] for v in h["nvec_x"][()].T])
            r["densx"] = r["dens"].copy(); r["densx"][j] = cplx(h["dens_x"][()])
        out[(int(h.attrs["seed_L"]), int(h.attrs["sign"]))] = r
    return out


PT = h5py.File(os.path.join(RES, "lcdm_full_pt.h5"), "r")
NVEC = PT["nvec"][()]; KM = KF * np.sqrt((NVEC.astype(float) ** 2).sum(0))
DL = {int(s): cplx(PT["dL"][()])[i] for i, s in enumerate(PT["seeds"][()])}
D2 = {int(s): cplx(PT["d2"][()])[i] for i, s in enumerate(PT["seeds"][()])}
PK = C.Spectrum(PT["pk_k"][()], PT["pk_P"][()])            # z = 0, σ8 = 0.8105
EDGES = KF * (np.arange(1, 33) - 0.5); KB = 0.5 * (EDGES[1:] + EDGES[:-1])
BIN = np.digitize(KM, EDGES) - 1; NB = len(KB)


def binsum(v, _=None):
    m = BIN >= 0
    return np.bincount(BIN[m], weights=np.real(v)[m], minlength=NB + 1)[:NB]


def stats(runs, sel, D_of_e, order=None, field="dens"):
    """Per long seed: binned sums (Sxx, Sxy, Syy) over its ±L members → R, Q with jackknife over seeds."""
    seeds = sorted({s for s, _ in runs}); E = len(D_of_e)
    Sxx = np.zeros((len(seeds), E, NB)); Sxy = np.zeros_like(Sxx); Syy = np.zeros_like(Sxx)
    for i, s in enumerate(seeds):
        for sg in (1, -1):
            if (s, sg) not in runs:
                continue
            d = runs[(s, sg)].get(field, runs[(s, sg)]["dens"]); x = sg * DL[s]
            for e in range(E):
                y = d[:, e] if order is None else d[:, order - 1, e]
                Sxx[i, e] += binsum(np.abs(x) ** 2, 0); Sxy[i, e] += binsum(y * np.conj(x), 0)
                Syy[i, e] += binsum(np.abs(y) ** 2, 0)
    D = np.asarray(D_of_e)[None, :, None]

    def est(ix):
        a, b, c = Sxx[ix].sum(0), Sxy[ix].sum(0), Syy[ix].sum(0)
        return b / (D[0] * a) - 1, c / (D[0] ** 2 * a) - 1
    R, Q = est(slice(None))
    jk = [est([j for j in range(len(seeds)) if j != i]) for i in range(len(seeds))]
    n = len(seeds)
    if n >= 2:
        eR = np.sqrt((n - 1) / n * sum((r - R) ** 2 for r, _ in jk)); eQ = np.sqrt((n - 1) / n * sum((q - Q) ** 2 for _, q in jk))
    else:                                                   # one seed (preliminary): mode-count weights only
        cnt = np.bincount(BIN[BIN >= 0], minlength=NB + 1)[:NB].astype(float)
        eR = np.broadcast_to(1e-3 / np.sqrt(np.maximum(cnt, 1)), R.shape).copy(); eQ = 2 * eR
    return dict(R=R, Q=Q, eR=eR, eQ=eQ, seeds=seeds, Sxx=Sxx, Syy=Syy, Sxy=Sxy)


# ── perturbation theory on the box ────────────────────────────────────────────────────────────────
def box_q_multiplicity(N=256):
    n = np.arange(-N // 2 + 1, N // 2)
    n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
    n2 = n2[(n2 > 0) & (KF * np.sqrt(n2) < KCUT)]
    u, c = np.unique(n2, return_counts=True)
    return KF * np.sqrt(u), c


QB, MULT = box_q_multiplicity()


def r1_box(k):
    """One-loop propagator R1(k) = P13/(2P) as the exact sum over box modes (z = 0, σ8 fid)."""
    r = QB[None, :] / np.atleast_1d(k)[:, None]
    return (MULT * PK(QB) * C._p13_bracket(r.ravel()).reshape(r.shape) / (1008 * r ** 2)).sum(1) / L ** 3


def bin_avg(f_of_k):
    """P_L-weighted average over the modes of each bin (fixed amplitudes: weight ∝ P_L(k_mode))."""
    w = PK(KM)
    u, inv = np.unique((NVEC.astype(int) ** 2).sum(0), return_inverse=True)
    v = f_of_k(KF * np.sqrt(u))[inv]                       # evaluate once per distinct |k|
    m = BIN >= 0
    return (np.bincount(BIN[m], weights=(w * v)[m], minlength=NB + 1)[:NB]
            / np.bincount(BIN[m], weights=w[m], minlength=NB + 1)[:NB])


def q22_box(seeds, s8fac=1.0):
    """Σ|δ⁽²⁾|² / Σ|δ_L|² per bin (z = 0 units; × D² gives P22/P_L)."""
    a = sum(binsum(np.abs(D2[s]) ** 2, 0) for s in seeds); b = sum(binsum(np.abs(DL[s]) ** 2, 0) for s in seeds)
    return s8fac * a / b


# continuum one-loop with the box's sourcing window (q in [k_f, 1.6)), wiggle/no-wiggle split and Σ²
def nowiggle(k):
    """Broadband: Gaussian smoothing (σ = 0.25 in ln k) of P / P_EH,nw, × P_EH,nw (EH98 no-wiggle fit)."""
    h, Om, Ob, ns = 0.67742, 0.259622 + 0.0488911, 0.0488911, 0.96822
    def tnw(kk):
        omh2 = Om * h * h; fb = Ob / Om; th = 2.72548 / 2.7
        s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * (Ob * h * h) ** 0.75)
        ag = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb ** 2
        q = kk * th ** 2 / (Om * h)  # kk in h/Mpc
        gam = Om * h * (ag + (1 - ag) / (1 + (0.43 * kk * h * s) ** 4))
        q = kk * th ** 2 / gam
        L0 = np.log(2 * np.e + 1.8 * q); C0 = 14.2 + 731 / (1 + 62.5 * q)
        return L0 / (L0 + C0 * q * q)
    lk = np.linspace(np.log(1e-4), np.log(50), 4000); kk = np.exp(lk)
    base = kk ** ns * tnw(kk) ** 2
    ratio = PK(kk) / base
    sig = 0.25 / (lk[1] - lk[0]); g = np.exp(-0.5 * (np.arange(-4 * sig, 4 * sig + 1) / sig) ** 2); g /= g.sum()
    pad = len(g) // 2
    rs = np.convolve(np.pad(ratio, pad, mode="edge"), g, mode="valid")
    return np.exp(np.interp(np.log(k), lk, np.log(rs * base)))


W2BOX = lambda q: ((np.asarray(q) >= KF) & (np.asarray(q) < KCUT)).astype(float)


def sigma2_bao(rbao=105.0, ks=0.2):
    q = np.linspace(KF, ks, 4000); x = q * rbao
    j0 = np.sin(x) / x; j2 = (3 / x ** 2 - 1) * np.sin(x) / x - 3 * np.cos(x) / x ** 2
    return np.trapz(nowiggle(q) * (1 - j0 + 2 * j2), q) / (6 * np.pi ** 2)


def continuum_tables(kk):
    Pnw = C.Spectrum(np.exp(np.linspace(np.log(1e-4), np.log(50), 4000)),
                     nowiggle(np.exp(np.linspace(np.log(1e-4), np.log(50), 4000))))
    t = {}
    for lab, P in (("L", PK), ("nw", Pnw)):
        t["p22" + lab] = np.array([C.p22(k, P, W2BOX) for k in kk])
        t["p13" + lab] = np.array([C.p13(k, P, W2BOX) for k in kk])
    t["PL"] = PK(kk); t["Pnw"] = Pnw(kk); t["Pw"] = t["PL"] - t["Pnw"]
    return t


# ── fits ──────────────────────────────────────────────────────────────────────────────────────────
def fit_k2(k, y, err, kmax, k4=True):
    m = (k <= kmax) & np.isfinite(y) & (err > 0)
    A = np.vstack([-k[m] ** 2] + ([k[m] ** 4] if k4 else [])).T
    w = 1 / err[m]
    p, *_ = np.linalg.lstsq(A * w[:, None], y[m] * w, rcond=None)
    cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None]))
    return p, np.sqrt(np.diag(cov))


def main():
    out = {}
    full_pm = load("lcdm_full", "pm"); full_lpt = load("lcdm_full", "lpt")
    any_run = next(iter(full_pm.values()))
    Ds = any_run["epochs"]; E = len(Ds)
    zs = [19.0, 9.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0]
    seeds = sorted({s for s, _ in full_pm})
    pm = stats(full_pm, None, Ds, field="densx")            # exact density: PM vs analytic PT
    pmc = stats(full_pm, None, Ds)                          # CIC: PM vs sheet (same estimator on both sides)
    SV2 = (MULT * PK(QB) / QB ** 2).sum() / L ** 3 / 3       # 1-D displacement variance of the box (z = 0)
    out["sigma_v2_box_z0"] = SV2

    # PT on the box
    r1 = bin_avg(r1_box)                                    # z = 0 propagator one-loop (bin averaged)
    q22 = q22_box(seeds)
    kk = KB; ct = continuum_tables(kk)
    Sig2 = sigma2_bao()
    out["Sigma2_bao_z0"] = Sig2
    # diagnostics: box vs continuum one-loop
    out["box_vs_cont"] = dict(k=kk.tolist(), q22_box=q22.tolist(), q22_cont=(ct["p22L"] / ct["PL"]).tolist(),
                              r1_box=r1.tolist(), r1_cont=(ct["p13L"] / (2 * ct["PL"])).tolist())
    ep = {}
    for e, D in enumerate(Ds):
        D2_ = D ** 2
        xv = kk ** 2 * SV2 * D2_ / 2
        R1L = np.exp(-xv) * (1 + D2_ * r1 + xv) - 1          # IR-resummed one-loop propagator
        Q1L = D2_ * (q22 + 2 * r1)
        x = kk ** 2 * Sig2 * D2_
        P1L = D2_ * (ct["p22L"] + ct["p13L"]); P1nw = D2_ * (ct["p22nw"] + ct["p13nw"])
        dIR = (ct["Pw"] * (np.exp(-x) * (1 + x) - 1) + (np.exp(-x) - 1) * (P1L - P1nw)) / ct["PL"]
        ctshape = (ct["Pnw"] + np.exp(-x) * ct["Pw"]) / ct["PL"]
        yR = (pm["R"][e] - R1L) / np.exp(-xv); eR = np.maximum(pm["eR"][e], 1e-6) / np.exp(-xv)
        yQ = (pm["Q"][e] - Q1L - dIR) / ctshape; eQ = np.maximum(pm["eQ"][e], 1e-6) / ctshape
        rec = dict(z=zs[e], D=float(D), inv_pm=float(np.mean([r["inv"][e] for r in full_pm.values()])))
        for lab, y, er, fac in (("prop", yR, eR, 1.0), ("pk", yQ, eQ, 2.0)):
            fits = {}
            for kmax in (0.05, 0.075, 0.1, 0.15, 0.2):
                p, sp = fit_k2(kk, y / fac, er / fac, kmax)
                p1, sp1 = fit_k2(kk, y / fac, er / fac, kmax, k4=False)
                fits[str(kmax)] = dict(C2=p[0], eC2=sp[0], C4=p[1], C2_only=p1[0], eC2_only=sp1[0])
            rec["C2_" + lab] = fits
        # zero-parameter prediction of P(k) with the propagator C2 (k4 fit, kmax = 0.1)
        C2p = rec["C2_prop"]["0.1"]["C2_only"]                 # primary: pure k² fit, k ≤ 0.1
        for lab, C2use in (("pred_prop", C2p), ("no_ct", 0.0)):
            Qp = Q1L + dIR - 2 * C2use * kk ** 2 * ctshape
            res = (pm["Q"][e] - Qp) / (1 + pm["Q"][e])
            rec[lab] = dict(C2=C2use, res=res.tolist(), err=(pm["eQ"][e] / (1 + pm["Q"][e])).tolist())
        rec["lin_res"] = (pm["Q"][e] / (1 + pm["Q"][e])).tolist()
        rec["R_pm"] = pm["R"][e].tolist(); rec["R_1L"] = R1L.tolist(); rec["Q_pm"] = pm["Q"][e].tolist()
        rec["Q_1L"] = Q1L.tolist(); rec["dIR"] = dIR.tolist()
        ep[str(zs[e])] = rec
    out["k"] = kk.tolist(); out["epochs"] = ep

    # sheet forward model: LPT_n on Λ-cut ICs vs PM (same realizations, field level)
    sheet = {}
    for name in ["lcdm_full"] + [f"lcdm_cut{c:03d}" for c in (10, 15, 20, 30, 50, 100)]:
        runs = full_lpt if name == "lcdm_full" else load(name, "lpt")
        if not runs:
            continue
        Lam = 1.6 if name == "lcdm_full" else int(name[-3:]) / 100
        sheet[name] = {"Lambda": Lam}
        for n in (1, 2, 3, 5, 7):
            runs = {key: v for key, v in runs.items() if key in full_pm}
            st = stats(runs, None, Ds, order=n)
            per = {}
            for e, D in enumerate(Ds):
                # field-level: P_PM / P_sheet − 1 per bin, cross-correlation r, response difference
                Sss = 0; Spp = 0; cross = 0
                for key in runs:
                    a = full_pm[key]["dens"][:, e]; b = runs[key]["dens"][:, n - 1, e]
                    Sss = Sss + binsum(np.abs(b) ** 2); Spp = Spp + binsum(np.abs(a) ** 2)
                    cross = cross + binsum(a * np.conj(b))
                rr = cross / np.sqrt(Spp * Sss)
                dR = pmc["R"][e] - st["R"][e]
                eR = np.sqrt(pmc["eR"][e] ** 2 + st["eR"][e] ** 2) + 1e-6
                okm = KB < 0.8 * Lam
                fits = {}
                for kmax in (0.05, 0.1, 0.15, 0.2):
                    if np.sum(okm & (KB <= kmax)) < 3:
                        continue
                    p, sp = fit_k2(KB[okm], dR[okm], eR[okm], kmax, k4=False)
                    fits[str(kmax)] = dict(C2=p[0], eC2=sp[0])
                ratio = Spp / Sss - 1                       # P_PM / P_sheet − 1 (raw sheet)
                per[str(zs[e])] = dict(ratio=ratio.tolist(), r=rr.tolist(), dR=dR.tolist(), C2=fits,
                                       inv=float(np.mean([r["inv"][n - 1, e] for r in runs.values()])))
            sheet[name][str(n)] = per
    out["sheet"] = sheet

    # σ8 variants (same seeds 1, 2): C2 from the propagator, PT rescaled by (σ8/σ8_fid)²
    s8 = {}
    for name, sig8 in (("lcdm_s8lo", 0.70), ("lcdm_s8hi", 0.92)):
        runs = load(name, "pm")
        if not runs:
            continue
        f = (sig8 / S8_FID) ** 2
        st = stats(runs, None, Ds, field="densx")
        SV2 = (MULT * PK(QB) / QB ** 2).sum() / L ** 3 / 3
        # the ICs are the fiducial ones × σ8/σ8_fid: rescale the regressor (DL) via D_eff
        rec = {}
        for e, D in enumerate(Ds):
            Deff = D * sig8 / S8_FID
            R = (st["R"][e] + 1) * D / Deff - 1           # stats() divided by D with the fiducial δ_L
            Q = (st["Q"][e] + 1) * D ** 2 / Deff ** 2 - 1
            xv = KB ** 2 * SV2 * Deff ** 2 / 2
            y = (R - (np.exp(-xv) * (1 + Deff ** 2 * r1 + xv) - 1)) / np.exp(-xv)
            p, sp = fit_k2(KB, y, (st["eR"][e] + 1e-6) / np.exp(-xv), 0.1, k4=False)
            rec[str(zs[e])] = dict(D=float(D), Deff=float(Deff), C2=p[0], eC2=sp[0], R=R.tolist(), Q=Q.tolist())
        s8[name] = dict(sigma8=sig8, epochs=rec)
    out["sigma8"] = s8

    with open(os.path.join(RES, "lcdm_cs_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    report(out)
    report_sheet(out)


def report(out):
    kk = np.array(out["k"])
    print(f"Σ²_BAO(z=0) = {out['Sigma2_bao_z0']:.2f} (Mpc/h)²")
    bc = out["box_vs_cont"]
    print("box vs continuum one-loop (z=0):  k   P22/P box  cont   R1 box  cont")
    for i in (1, 3, 7, 11, 15, 23):
        print(f"   {kk[i]:.3f}  {bc['q22_box'][i]:8.4f} {bc['q22_cont'][i]:8.4f}  {bc['r1_box'][i]:8.4f} {bc['r1_cont'][i]:8.4f}")
    print("\nC2 [h⁻²Mpc²] (fit −C2 k² + C4 k⁴ to PM − one-loop; ± jackknife), propagator vs P(k):")
    print("   (pure k² fits; last column: k²+k⁴ fit to the propagator, k ≤ 0.2)")
    print("   z     D    inv(PM)   prop k≤0.075   prop k≤0.1    P(k) k≤0.1    P(k) k≤0.15   prop k²+k⁴ ≤0.2")
    for z, r in out["epochs"].items():
        a, b, c, d, f = r["C2_prop"]["0.075"], r["C2_prop"]["0.1"], r["C2_pk"]["0.1"], r["C2_pk"]["0.15"], r["C2_prop"]["0.2"]
        print(f"  {float(z):4.2f} {r['D']:.3f}  {r['inv_pm']:.1e}  {a['C2_only']:6.3f}±{a['eC2_only']:.3f}  {b['C2_only']:6.3f}±{b['eC2_only']:.3f}"
              f"  {c['C2_only']:6.3f}±{c['eC2_only']:.3f}  {d['C2_only']:6.3f}±{d['eC2_only']:.3f}   {f['C2']:6.3f}±{f['eC2']:.3f}")
    print("\nZero-parameter prediction (one-loop + IR + C2 from the propagator): (P_PM − P_pred)/P_PM [%]")
    sel = [np.argmin(abs(kk - x)) for x in (0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.28)]
    print("   z     " + "  ".join(f"k={kk[i]:.3f}" for i in sel) + "   | no counterterm at same k")
    for z, r in out["epochs"].items():
        p = np.array(r["pred_prop"]["res"]) * 100; n0 = np.array(r["no_ct"]["res"]) * 100
        print(f"  {float(z):4.2f}  " + "  ".join(f"{p[i]:+7.2f}" for i in sel) + "   | " + " ".join(f"{n0[i]:+6.2f}" for i in sel))




def k_fail(k, res, tol=0.01, kmin=0.02):
    """Smallest k (> kmin) beyond which |res| exceeds tol in two consecutive bins (NaN-safe)."""
    bad = (np.abs(np.nan_to_num(res, nan=1.0)) > tol) & (k > kmin)
    for i in range(len(k) - 1):
        if bad[i] and bad[i + 1]:
            return k[i]
    return np.inf


def report_sheet(out):
    kk = np.array(out["k"])
    print("\nSheet forward model (nLPT on Λ-cut ICs) vs PM, field level (CIC both):")
    print("  Λ     n   z    inv(sheet)  C2_sheet(k≤0.1)  k_1% raw  k_1% +ct   1−r(k≈0.1)  1−r(k≈0.2)")
    for name, sh in out["sheet"].items():
        Lam = sh["Lambda"]
        for n in ("1", "2", "3", "5", "7"):
            if n not in sh:
                continue
            for z in ("1.0", "0.5", "0.0"):
                r = sh[n][z]
                rat = np.array(r["ratio"]); rr = np.array(r["r"])
                c2 = r["C2"].get("0.1", r["C2"].get("0.05", {"C2": np.nan, "eC2": np.nan}))
                corr = (1 + rat) / (1 - 2 * c2["C2"] * kk ** 2) - 1 if np.isfinite(c2["C2"]) else rat
                lim = kk < 0.8 * Lam
                kr = k_fail(kk[lim], rat[lim]); kc = k_fail(kk[lim], corr[lim])
                i1, i2 = np.argmin(abs(kk - 0.1)), np.argmin(abs(kk - 0.2))
                print(f"  {Lam:4.2f}  {n}  {float(z):3.1f}  {r['inv']:.2e}   {c2['C2']:+7.3f}±{c2.get('eC2', np.nan):.3f}"
                      f"   {kr:6.3f}    {kc:6.3f}    {1 - rr[i1]:.1e}     {1 - rr[i2]:.1e}")


if __name__ == "__main__":
    main()
