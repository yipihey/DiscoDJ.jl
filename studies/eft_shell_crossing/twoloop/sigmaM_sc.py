#!/usr/bin/env python3
"""Spherical-collapse predictions for the fixed-mass sigmas, and what separates mass-drawn spheres from Lagrangian patches (results/sigmaM_sc_<cfg>.json, figures/figM2_*).

Spherical collapse (EdS, exact parametric): V/V_L = 1/(1 + delta_NL(delta_L)), frozen at 1/Delta_vir = 1/178 after collapse.  delta_L = D sigma_lin(R_L) nu, nu Gaussian.
  Lagrangian patch, fixed M:     lnV = g(delta_L) with P Gaussian at R_L(M)                                  [no free parameter except the freeze]
  random centres, fixed M:       the same reweighted by V (a random point lies in a patch with probability proportional to its volume)
  mass-weighted spheres, fixed M: the Lagrangian PDF itself
  random spheres, fixed R:       with contraction: mu at R_E comes from delta_L at R_L = R_E mu^{1/3}, P(mu) from the Gaussian at that R_L, normalised; without contraction: sigma_lin(R_E)
Measured here from the same histograms: s_muw = fixed M with centres weighted by the sphere's own mass mu_R instead of rho(x); s_lnmu = sigma(ln mu_R) at fixed radius (same variable as ln V).
"""
import glob, json, os, re, sys
import h5py, numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:]
cfg = sys.argv[1]; figdir = sys.argv[2] if len(sys.argv) > 2 else None
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
r = json.load(open(os.path.join(here, f"sigmaM_{cfg}.json")))
RL = np.array(r["RL"]); RE = np.array(r["RE"]); QS = (np.arange(800) + 0.5) / 800
with h5py.File(os.path.join(D0, "tl_full", "tl", "ops_sharp_L1.h5")) as h:
    pk, pP = h["pk_k"][()], h["pk_P"][()]
KUV = {"tl_full": 1.6, "tl_ir020": 0.2, "tl_uv06": 0.6, "tl_uv10": 1.0, "tl_div": 1.6, "tl_div10": 1.0}[cfg]
MOD = (0.3, -1.0) if "div" in cfg else None
n = np.arange(-256, 257); n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel(); KF = 2 * np.pi / 500.
n2 = n2[(n2 > 0) & (KF * np.sqrt(n2) < KUV)]; u2, c2 = np.unique(n2, return_counts=True); qk = KF * np.sqrt(u2); qP = np.exp(np.interp(np.log(qk), np.log(pk), np.log(pP)))
if MOD:
    qP = np.where(qk > MOD[0], np.exp(np.interp(np.log(MOD[0]), np.log(pk), np.log(pP))) * (qk / MOD[0]) ** MOD[1], qP)
tophat = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
grid = np.exp(np.linspace(np.log(1.5), np.log(400), 400)); sl = np.array([np.sqrt((c2 * qP * tophat(qk * R) ** 2).sum() / 500. ** 3) for R in grid])
sig_lin = lambda R: np.exp(np.interp(np.log(R), np.log(grid), np.log(sl)))

# exact EdS spherical collapse, ln(V/V_L) against delta_L
th = np.linspace(1e-3, 2 * np.pi - 1e-3, 40000)
dL_o = 3 / 5 * (3 / 4 * (th - np.sin(th))) ** (2 / 3); mu_o = 9 / 2 * (th - np.sin(th)) ** 2 / (1 - np.cos(th)) ** 3
et = np.linspace(1e-3, 40, 40000)
dL_u = -3 / 5 * (3 / 4 * (np.sinh(et) - et)) ** (2 / 3); mu_u = 9 / 2 * (np.sinh(et) - et) ** 2 / (np.cosh(et) - 1) ** 3
dL = np.concatenate([dL_u[::-1], dL_o]); lnmu = np.log(np.concatenate([mu_u[::-1], mu_o]))
o = np.argsort(dL); dL, lnmu = dL[o], np.maximum.accumulate(lnmu[o]); LVIR = np.log(178.0)
def g_lnV(d):
    """ln(V/V_L) of a sphere with linear overdensity d, frozen at virial density after that point on its way to collapse."""
    return -np.minimum(np.interp(d, dL, lnmu), LVIR)
print(f"check: delta_c = {dL_o[-1]:.3f}; small delta_L: lnV = -delta_L? at 0.01: {g_lnV(0.01):.5f}, at -0.01: {g_lnV(-0.01):.5f}")

def moments(x, w):
    w = w / w.sum(); m = (w * x).sum(); return m, np.sqrt((w * (x - m) ** 2).sum())

def sc(R, D):
    s = D * sig_lin(R); nu = np.linspace(-8, 8, 16001); p = np.exp(-nu ** 2 / 2); lv = g_lnV(s * nu)
    return dict(lag=moments(lv, p), rand=moments(lv, p * np.exp(lv)))

def sc_fixedR(R, D, contraction=True):
    """sigma(delta_R), sigma(ln mu_R) at fixed Eulerian radius from spherical collapse, delta_L at the Lagrangian radius of each density."""
    lm = np.linspace(-6, LVIR, 8000); mu = np.exp(lm); Rl = R * mu ** (1 / 3) if contraction else np.full_like(mu, R)
    s = D * sig_lin(Rl); d = np.interp(lm, lnmu, dL); nu = d / s
    # volume-weighted P(ln mu) d ln mu: Gaussian in nu, Jacobian d nu / d ln mu, one factor 1/mu from the mass-to-volume weight
    dnu = np.abs(np.gradient(nu, lm)); w = np.exp(-nu ** 2 / 2) * dnu / mu
    w[lm >= LVIR - 1e-6] = 0
    m0 = (w * mu).sum() / w.sum(); sd = np.sqrt((w * (mu / m0 - 1) ** 2).sum() / w.sum()); sl_ = moments(lm, w)[1]
    return sd, sl_

def fixed_mass(T, lb, R0):
    lnR = np.log(RE); g = np.array([np.interp(QS, T[j, ::-1], lb[::-1]) for j in range(len(RE))]) - 3 * (np.log(R0) - lnR)[:, None]
    x = np.full(len(QS), np.nan)
    for i in range(len(QS)):
        c = g[:, i]; k = np.argmax(c >= 0)
        if c[0] >= 0 or c.max() < 0: continue
        t = -c[k - 1] / (c[k] - c[k - 1]); x[i] = 3 * (lnR[k - 1] + t * (lnR[k] - lnR[k - 1]) - np.log(R0))
    return x

# reweighted measurements, per seed and sign
data = {}
for f in sorted(glob.glob(os.path.join(D0, "sigmaM", f"{cfg}_pm_L*_s*_D*.h5"))):
    m = re.search(r"_L(\d+)_s([+-]\d)_D([0-9.]+)\.h5$", f); s, sg, D = int(m[1]), int(m[2]), float(m[3])
    with h5py.File(f) as h:
        C = h["count"][()].T; lb = h["lbins"][()]
    mid = 0.5 * (lb[1:] + lb[:-1]); W = C * np.exp(mid)
    Ht = W / W.sum(1, keepdims=True); T = np.concatenate([Ht[:, ::-1].cumsum(1)[:, ::-1], np.zeros((Ht.shape[0], 1))], axis=1)
    sm = []; un = []
    for R0 in RL:
        x = fixed_mass(T, lb, R0); ok = np.isfinite(x); sm.append(x[ok].std()); un.append(1 - ok.mean())
    Pc = C / C.sum(1, keepdims=True); me = (Pc * mid).sum(1); slm = np.sqrt((Pc * (mid - me[:, None]) ** 2).sum(1))
    data.setdefault(D, {}).setdefault(s, []).append((np.array(sm), np.array(un), slm))
def agg(D, i):
    a = np.array([np.mean([x[i] for x in data[D][s]], axis=0) for s in sorted(data[D])]); k = len(a); m = a.mean(0)
    J = np.array([np.delete(a, q, 0).mean(0) for q in range(k)]); return m, np.sqrt((k - 1) / k * ((J - J.mean(0)) ** 2).sum(0))

idx = [int(np.argmin(abs(RL - x))) for x in (10.5, 14.8, 20.7, 29, 51)]
print("\nD  R0  M | lin | LAGRANGIAN patch meas, SC | RANDOM centres meas, SC | MASS-DRAWN meas (rho weight) | mu_R weight meas | SC mass-weight = patch; fraction unresolved")
out = dict(cfg=cfg, RL=RL.tolist(), RE=RE.tolist(), epochs={})
for D in sorted(data):
    e = r["epochs"][f"{D:.3f}"]; smu, emu = agg(D, 0); uu, _ = agg(D, 1); slm, _ = agg(D, 2)
    pred = [sc(R, D) for R in RL]; pR = [sc_fixedR(R, D) for R in RL]; pR0 = [sc_fixedR(R, D, False) for R in RL]
    sR = np.interp(np.log(RL), np.log(RE), np.array(e["s_R"])); sLM = np.interp(np.log(RL), np.log(RE), slm)
    pRE = [sc_fixedR(R, D) for R in RE]; pRE0 = [sc_fixedR(R, D, False) for R in RE]
    out["epochs"][f"{D:.3f}"] = dict(s_muw=smu.tolist(), e_muw=emu.tolist(), unres_muw=uu.tolist(), s_lnmu=sLM.tolist(), s_R=sR.tolist(), sc_lag=[p["lag"][1] for p in pred], sc_lag_mean=[p["lag"][0] for p in pred], sc_rand=[p["rand"][1] for p in pred],
                                     s_lnmu_RE=slm.tolist(), sc_fixRE_d=[p[0] for p in pRE], sc_fixRE_lnmu=[p[1] for p in pRE], sc_nocontrE_d=[p[0] for p in pRE0], sc_fixR_d=[p[0] for p in pR], sc_fixR_lnmu=[p[1] for p in pR], sc_nocontr_d=[p[0] for p in pR0], sc_nocontr_lnmu=[p[1] for p in pR0])
    for i in idx:
        print(f"{D:.1f} {RL[i]:5.1f} {4 * np.pi / 3 * RL[i] ** 3 * 2.775e11:.1e} | {D * sig_lin(RL[i]):.3f} | {e['s_lag'][i]:.3f} {pred[i]['lag'][1]:.3f} | {e['s_rand'][i]:.3f} {pred[i]['rand'][1]:.3f} | {e['s_mass'][i]:.3f} | {smu[i]:.3f}±{emu[i]:.3f} | u={uu[i]:.3f}")
print("\nFixed Eulerian radius: measured sigma(delta_R), SC with contraction, SC without; measured sigma(ln mu_R), SC with, SC without.  Ratio of contraction effect = with/without")
for D in sorted(data):
    q = out["epochs"][f"{D:.3f}"]
    for i in idx:
        print(f"{D:.1f} {RL[i]:5.1f} | {q['s_R'][i]:.3f} {q['sc_fixR_d'][i]:.3f} {q['sc_nocontr_d'][i]:.3f} | {q['s_lnmu'][i]:.3f} {q['sc_fixR_lnmu'][i]:.3f} {q['sc_nocontr_lnmu'][i]:.3f} | {q['sc_fixR_lnmu'][i] / q['sc_nocontr_lnmu'][i]:.3f}")
json.dump(out, open(os.path.join(here, f"sigmaM_sc_{cfg}.json"), "w"), indent=0, default=float)

if figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # dataviz default categorical slots in fixed order; each entity keeps its colour from figM1
    COL = {"R": "#2a78d6", "mass": "#eb6834", "rand": "#1baf7a", "lag": "#eda100", "muw": "#e87ba4", "ink": "#52514e"}
    MASS = 4 * np.pi / 3 * RL ** 3 * 2.775e11
    show = [D for D in (0.5, 0.6, 0.8, 1.0) if f"{D:.3f}" in out["epochs"]]
    fig, axs = plt.subplots(1, len(show), figsize=(3.5 * len(show), 4.2), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.05), facecolor="white")
    for ax, D in zip(axs, show):
        e = r["epochs"][f"{D:.3f}"]; q = out["epochs"][f"{D:.3f}"]
        lin = D * sig_lin(RL)
        A = lambda k: np.array(q[k]); B = lambda k: np.array(e[k])
        cut = lambda y, t: np.where(np.array(t) > 0.005, np.nan, y)
        lag = np.where(B("ok_lag") < 0.999, np.nan, B("s_lag"))
        ax.axhline(1, color=COL["ink"], lw=1.2, ls="--", label="linear")
        ax.plot(MASS, A("s_R") / lin, color=COL["R"], lw=2, label=r"fixed radius $\sigma(\delta_R)$")
        ax.plot(MASS, A("sc_fixR_d") / lin, color=COL["R"], lw=1.5, ls=(0, (4, 2)), label="  SC with contraction")
        ax.plot(MASS, A("sc_nocontr_d") / lin, color=COL["R"], lw=1.5, ls=":", label="  SC, no contraction")
        ax.plot(MASS, lag / lin, color=COL["lag"], lw=2, label=r"Lagrangian patch $\sigma(\ln V)$")
        ax.plot(MASS, A("sc_lag") / lin, color=COL["lag"], lw=1.5, ls=(0, (4, 2)), label="  SC")
        ax.plot(MASS, cut(B("s_rand"), B("t_rand")) / lin, color=COL["rand"], lw=2, label=r"random centres, fixed $M$")
        ax.plot(MASS, A("sc_rand") / lin, color=COL["rand"], lw=1.5, ls=(0, (4, 2)), label="  SC")
        ax.plot(MASS, cut(B("s_mass"), B("t_mass")) / lin, color=COL["mass"], lw=2, label=r"$\rho$-drawn centres, fixed $M$")
        ax.plot(MASS, cut(A("s_muw"), A("unres_muw")) / lin, color=COL["muw"], lw=2, label=r"$\mu_R$-weighted centres, fixed $M$")
        ax.set_xscale("log"); ax.set_ylim(0.75, 1.6); ax.set_xlim(8e14, 7e17)
        ax.set_title(f"D = {D:g}", fontsize=11, loc="left", color="#1b1b1a"); ax.set_xlabel(r"$M$ [$h^{-1}M_\odot$]", fontsize=10)
        ax.grid(True, which="major", color="#e4e3df", lw=0.7); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].set_ylabel(r"$\sigma$ / linear $\sigma$ at the same mass", fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, 1.13))
    os.makedirs(figdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"figM2_sigma_spherical_collapse_{cfg}.{ext}"), dpi=170, bbox_inches="tight")
    print("wrote figure")
