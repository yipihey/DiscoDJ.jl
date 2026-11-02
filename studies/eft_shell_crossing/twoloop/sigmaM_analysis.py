#!/usr/bin/env python3
"""Fixed-mass sigmas from the sigmaM.jl outputs -> results/sigmaM_<cfg>.json (run with a Python that has h5py, numpy; matplotlib for figures).

    python sigmaM_analysis.py <cfg> [figdir]

M is labelled by R0, the radius holding M at mean density: M = rhobar V(R0).  With lnV = 3 ln(R/R0):
  s_mass(M)  std of lnV around centres drawn from the density   P(R_M <= R) = P_m(mu_R >= (R0/R)^3)
  s_rand(M)  the same around uniformly random centres
  s_lag(M)   std of ln(image volume) of a fixed Lagrangian sphere of radius R0 (signed P1 tetrahedra: the volume enclosed by the image of its surface)
  s_lagu(M)  the same, unsigned (total sheet volume, counts folds)
  s_R(R0)    std of delta_R in random spheres of fixed Eulerian radius R0
  s_lin(R0)  D * linear sigma at radius R0: sum over the box's own modes, k < k_uv, of the fixed-amplitude spectrum
Errors: delete-one jackknife over seeds, after averaging the +- pair of each seed.
Where more than 2 % of the mass has R_M below the smallest radius (or more than 2 % of points above the largest), that M is not quoted.
"""
import glob, json, os, re, sys
import h5py
import numpy as np

cfg = sys.argv[1]
figdir = sys.argv[2] if len(sys.argv) > 2 else None
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
KUV = {"tl_full": 1.6, "tl_uv10": 1.0, "tl_uv06": 0.6, "tl_ir010": 0.1, "tl_ir015": 0.15, "tl_ir020": 0.2, "tl_ir030": 0.3}[cfg]


def tophat(x):
    x = np.maximum(x, 1e-4)
    return 3 * (np.sin(x) - x * np.cos(x)) / x ** 3


with h5py.File(os.path.join(D0, "tl_full", "tl", "ops_sharp_L1.h5")) as h:
    pk, pP = h["pk_k"][()], h["pk_P"][()]
L = 500.0
KF = 2 * np.pi / L
n = np.arange(-256, 257)
n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
n2 = n2[(n2 > 0) & (KF * np.sqrt(n2) < KUV)]
u2, cnt2 = np.unique(n2, return_counts=True)
qk = KF * np.sqrt(u2)
qP = np.exp(np.interp(np.log(qk), np.log(pk), np.log(pP)))


def sig_lin(R):
    return float(np.sqrt((cnt2 * qP * tophat(qk * R) ** 2).sum() / L ** 3))


QS = (np.arange(800) + 0.5) / 800


def fixed_mass(T, lb, RE, R0):
    """Quantiles of lnV at fixed M from the table T[j, b] = P(log mu_{R_j} >= lb[b]).
    F(R, M) = q is a contour: log mu_q(R) is smooth in ln R, so solve log mu_q(R) = 3 ln(R0/R) by interpolating along R for each q.  Reading the CDF at grid radii
    instead would give each radius interval a uniform distribution, an artificial width ~ dlnV/sqrt(12) when the distribution is narrower than the grid."""
    x = np.full(len(QS), np.nan); lnR = np.log(RE); g = np.empty((len(RE), len(QS)))
    for j in range(len(RE)):
        g[j] = np.interp(QS, T[j, ::-1], lb[::-1])                  # log mu_q(R_j): the level exceeded with probability q
    g = g - 3 * (np.log(R0) - lnR)[:, None]                         # log(mass inside R / M) on the contour: increases with R
    for i in range(len(QS)):
        c = g[:, i]
        k = np.argmax(c >= 0)
        if c[0] >= 0 or c.max() < 0:
            continue
        t = -c[k - 1] / (c[k] - c[k - 1])
        x[i] = 3 * (lnR[k - 1] + t * (lnR[k] - lnR[k - 1]) - np.log(R0))
    return x


def one(f):
    with h5py.File(f) as h:
        RE, RL = h["RE"][()], h["RL"][()]
        C, W = h["count"][()].T, h["massw"][()].T          # (radius, bin)
        mom = h["mom"][()].T; lag = h["lag"][()].T; lb = h["lbins"][()]; D = float(h.attrs["D"])
    def table(H):
        H = H / H.sum(1, keepdims=True)
        return np.concatenate([H[:, ::-1].cumsum(1)[:, ::-1], np.zeros((H.shape[0], 1))], axis=1)   # P(log mu >= lb)
    Fc, Fm = table(C), table(W)
    out = dict(D=D, RE=RE, RL=RL, s_R=np.sqrt(mom[:, 1]), xibar=mom[:, 2], s_Rm=np.sqrt(np.maximum(mom[:, 3] - mom[:, 2] ** 2, 0)))
    for tag, T in (("rand", Fc), ("mass", Fm)):
        ms, ss, tr = [], [], []
        for R0 in RL:
            x = fixed_mass(T, lb, RE, R0)
            ok = np.isfinite(x)
            ms.append(x[ok].mean() if ok.any() else np.nan); ss.append(x[ok].std() if ok.any() else np.nan); tr.append(1 - ok.mean())
        out["m_" + tag] = np.array(ms); out["s_" + tag] = np.array(ss); out["t_" + tag] = np.array(tr)
    out["m_lag"], out["s_lag"], out["m_lagu"], out["s_lagu"], out["ok_lag"] = lag[:, 0], lag[:, 1], lag[:, 2], lag[:, 3], lag[:, 4]
    return out


files = sorted(glob.glob(os.path.join(D0, "sigmaM", f"{cfg}_pm_L*_s*_D*.h5")))
R = {}
for f in files:
    m = re.search(r"_L(\d+)_s([+-]\d)_D([0-9.]+)\.h5$", f)
    R[(int(m[1]), int(m[2]), float(m[3]))] = one(f)
seeds = sorted({k[0] for k in R}); Ds = sorted({k[2] for k in R})
print(f"{cfg}: {len(R)} snapshots, seeds {seeds}, D {Ds}")
FIELDS = ("s_R", "xibar", "s_Rm", "m_rand", "s_rand", "t_rand", "m_mass", "s_mass", "t_mass", "m_lag", "s_lag", "m_lagu", "s_lagu", "ok_lag")


def agg(D, fld):
    """Mean over seeds of the +- pair mean, with jackknife error over the seeds that have both signs."""
    ss = [s for s in seeds if (s, 1, D) in R and (s, -1, D) in R]
    if not ss:                                  # partial output while the run is in progress: not for quoting
        ss = [s for s in seeds if (s, 1, D) in R]
        a = np.array([R[(s, 1, D)][fld] for s in ss])
    else:
        a = np.array([0.5 * (R[(s, 1, D)][fld] + R[(s, -1, D)][fld]) for s in ss])
    m = a.mean(0)
    if len(ss) < 2:
        return m, m * np.nan
    J = np.array([np.delete(a, i, 0).mean(0) for i in range(len(ss))]); k = len(ss)
    return m, np.sqrt((k - 1) / k * ((J - J.mean(0)) ** 2).sum(0))


s0 = next(iter(R.values())); RE, RL = s0["RE"], s0["RL"]
slinL = np.array([sig_lin(r) for r in RL]); slinE = np.array([sig_lin(r) for r in RE])
res = dict(cfg=cfg, RL=RL.tolist(), RE=RE.tolist(), sig_lin_RL=slinL.tolist(), sig_lin_RE=slinE.tolist(), epochs={})
for D in Ds:
    e = {}
    for fl in FIELDS:
        m, s = agg(D, fl); e[fl] = m.tolist(); e["e_" + fl] = s.tolist()
    res["epochs"][f"{D:.3f}"] = e
json.dump(res, open(os.path.join(here, f"sigmaM_{cfg}.json"), "w"), indent=0, default=float)

idx = [i for i, r in enumerate(RL) if any(abs(r - x) < 0.5 * (RL[1] / RL[0] - 1) * x for x in (6, 8, 10, 15, 20, 30, 50, 80))]
print("\nAll sigmas at the same M. R0 = radius holding M at mean density. Dsig_lin is the linear sigma at R0.  Masked '--' where > 2 % of the distribution lies outside the radius range.")
print("D    R0   | Dsig_lin | sigma(R0) | s_mass ± | s_rand ± | s_lag ± | s_lag,unsigned | <lnV>_mass <lnV>_rand <lnV>_lag")
for D in Ds:
    e = res["epochs"][f"{D:.3f}"]
    S = lambda k: np.array(e[k])
    sR = np.interp(np.log(RL), np.log(RE), S("s_R"))
    for i in idx:
        mask = lambda t: "--" if S("t_" + t)[i] > 0.02 else None
        fm = lambda v, er, t: (mask(t) or f"{v:.3f}±{er:.3f}")
        print(f"{D:.1f} {RL[i]:5.1f} | {D * slinL[i]:.3f} | {sR[i]:.3f} | {fm(S('s_mass')[i], S('e_s_mass')[i], 'mass')} | {fm(S('s_rand')[i], S('e_s_rand')[i], 'rand')} | {S('s_lag')[i]:.3f}±{S('e_s_lag')[i]:.3f} | {S('s_lagu')[i]:.3f} | {S('m_mass')[i]:+.3f} {S('m_rand')[i]:+.3f} {S('m_lag')[i]:+.3f}")

print("\nContraction test: is sigma closer to linear theory at fixed mass than at fixed Eulerian radius?  Ratios to Dsig_lin(R0).")
print("D    R0  | sigma(R0)/lin | s_mass/lin | s_rand/lin | s_lag/lin | contraction: R_L/R_E from 1+xibar | sig_lin(R_L)/sig_lin(R_E)")
for D in Ds:
    e = res["epochs"][f"{D:.3f}"]
    S = lambda k: np.array(e[k])
    sR = np.interp(np.log(RL), np.log(RE), S("s_R")); xi = np.interp(np.log(RL), np.log(RE), S("xibar"))
    for i in idx:
        rl = RL[i] * (1 + xi[i]) ** (1 / 3)
        lin = D * slinL[i]
        print(f"{D:.1f} {RL[i]:5.1f} | {sR[i] / lin:.3f} | {'--' if S('t_mass')[i] > .02 else format(S('s_mass')[i] / lin, '.3f')} | {'--' if S('t_rand')[i] > .02 else format(S('s_rand')[i] / lin, '.3f')} | {S('s_lag')[i] / lin:.3f} | {rl / RL[i]:.3f} | {sig_lin(rl) / slinL[i]:.3f}")

if figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # default validated categorical order (dataviz references/palette.md), fixed per entity across every panel
    COL = {"lin": "#52514e", "R": "#2a78d6", "mass": "#eb6834", "rand": "#1baf7a", "lag": "#eda100"}
    LAB = {"lin": r"linear, $D\,\sigma_{\rm lin}(M)$", "R": r"random spheres, fixed radius $R(M)$: $\sigma(\delta_R)$", "mass": r"mass-drawn centres, fixed $M$: $\sigma(\ln V)$",
           "rand": r"random centres, fixed $M$: $\sigma(\ln V)$", "lag": r"Lagrangian patch, fixed $M$: $\sigma(\ln V)$"}
    show = [D for D in (0.3, 0.5, 0.8, 1.0) if D in Ds]
    MASS = 4 * np.pi / 3 * RL ** 3 * 2.775e11       # h⁻¹ M☉, EdS (Ω_m = 1): M = ρ̄ V(R0)
    fig, axs = plt.subplots(2, len(show), figsize=(3.4 * len(show), 6.2), sharex=True, sharey="row", gridspec_kw=dict(height_ratios=[3, 2], hspace=0.06, wspace=0.05), facecolor="white")
    for c, D in enumerate(show):
        e = res["epochs"][f"{D:.3f}"]
        S = lambda k: np.array(e[k])
        lin = D * slinL
        sR = np.interp(np.log(RL), np.log(RE), S("s_R"))
        cur = {"lin": lin, "R": sR, "mass": np.where(S("t_mass") > .02, np.nan, S("s_mass")), "rand": np.where(S("t_rand") > .02, np.nan, S("s_rand")), "lag": S("s_lag")}
        for k, y in cur.items():
            axs[0, c].plot(MASS, y, color=COL[k], lw=2, ls="--" if k == "lin" else "-", label=LAB[k], zorder=3 if k == "mass" else 2)
            if k != "lin":
                axs[1, c].plot(MASS, y / lin, color=COL[k], lw=2)
        axs[0, c].set_xscale("log"); axs[0, c].set_yscale("log"); axs[1, c].set_xscale("log")
        axs[1, c].axhline(1, color=COL["lin"], lw=1.2, ls="--")
        axs[0, c].set_title(f"D = {D:g}", fontsize=11, loc="left", color="#1b1b1a")
        axs[1, c].set_xlabel(r"$M$ [$h^{-1}M_\\odot$]", fontsize=10)
        for a in axs[:, c]:
            a.grid(True, which="major", color="#e4e3df", lw=0.7); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(labelsize=8.5, color="#8a8a85")
    axs[0, 0].set_ylabel(r"$\sigma$", fontsize=11); axs[1, 0].set_ylabel(r"$\sigma$ / linear", fontsize=11)
    axs[1, 0].set_ylim(0.6, 1.6)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(" ", fontsize=20)
    os.makedirs(figdir, exist_ok=True)
    axs[0, 0].set_ylim(bottom=max(1e-3, np.nanmin(show[0] * slinL) * 0.8))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"figM1_sigma_fixed_mass_{cfg}.{ext}"), dpi=170, bbox_inches="tight")
    print("wrote figure")
