"""Collect steps 2-5 (results/steps/steps_N*_l*_z*.npz) into convergence bookkeeping, the
marginal / copula / displacement-mapping attribution, tables and figures.

    python steps_collect.py        (also: python analysis.py collect)

Outputs: results/steps/summary.json, results/steps_tables.md, figures/steps_*.{png,pdf}.

EVERY statistic is CONDITIONAL on the single-stream mask (Lagrangian M, Eulerian E, both runs)
and is not an unconditional two-point statistic.

Convergence (per statistic, per z, R, weighting, run), decided before looking at LPT-vs-N-body
differences:
  particle number  S(N_hi, level 0) vs S(N_lo, level 0), N_hi = the largest N available
  sheet refinement S(N_ref, level 1) vs S(N_ref, level 0), N_ref = the largest N with level 1
  tolerance        tol = max(TOL_REL |S| + TOL_ABS, 3 x tie-seed floor of step 1c)
  PRE-SET run-level test (reported, 'run-level'): a bin passes if each run's statistic passes.
  It turned out to be vacuous: the LPT - N-body differences (dxi_y ~ 1e-3, 1 - rho ~ 1e-4) are
  far below its tolerance.  After seeing the level-0 z = 0 runs (N = 64, 128, 256; before any
  refinement result) convergence was therefore moved to the DIFFERENCES themselves (used for
  every 'converged' flag and grey band):
    xi       D = xi[NB] - xi[LPT]:  |D_a - D_b| <= TOL_D_REL |D_hi| + noise
    r(k)     1 - r(k):              |d(1-r)|    <= TOL_D_REL (1-r)_hi + noise
             noise = max(TOL_D_ABS, largest resolution change of D at r >= NOISE_R, resp.
             of 1 - r at k <= 2 pi / NOISE_R), where D has no signal (measured, per statistic)
    scalars  1 - rho_S, 1 - eta^2, 1 - R2_iso: relative TOL_D_REL (+ TOL_D_ABS);
             non-monotone part: absolute NONMONO_ABS (it is estimator-limited, see report)
  A separation bin (or k bin) is CONVERGED only if both comparisons pass; the
  converged range is the contiguous range adjoining the large-scale end (xi: r >= r_conv; r(k):
  k <= k_conv).  Scalars (Spearman, eta^2, R2_iso) use TOL_SCALAR.  Headline numbers are from the
  N_hi, level-0 run; refinement at N_hi is assumed no worse than at N_ref (stated in the report).

Attribution (Eulerian mask E, R fixed; non-unique, both orderings are reported):
  delta space      T = xi_d[NB] - xi_d[LPT]
      ordering A (marginal first)  Marg_A = xi_d[LPT ranks, NB marginal] - xi_d[LPT]
                                   Cop_A  = xi_d[NB] - xi_d[LPT ranks, NB marginal]
      ordering B (copula first)    Cop_B  = xi_d[NB ranks, LPT marginal] - xi_d[LPT]
                                   Marg_B = xi_d[NB] - xi_d[NB ranks, LPT marginal]
  score space      D = xi_y^E[NB] - xi_y^E[LPT]; Z_ab = Lagrangian score of run a carried by the
                   displacement map of run b
      ordering 1 (Lagrangian copula first)  Lag_1 = xi_Z[nl] - xi_Z[ll], Map_1 = xi_Z[nn] - xi_Z[nl]
      ordering 2 (mapping first)            Map_2 = xi_Z[ln] - xi_Z[ll], Lag_2 = xi_Z[nn] - xi_Z[ln]
      residual                              D - (xi_Z[nn] - xi_Z[ll])  (Eulerian re-smoothing and
                                            re-ranking not carried by label transport)
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CFG, OUT, FIG

STEPS = os.path.join(OUT, "steps")
S1 = os.path.join(OUT, "step1")
L = CFG["box_L"]
TOL_REL, TOL_ABS, TOL_SCALAR, TOL_RK = 0.02, 0.002, 0.005, 0.01      # pre-set, run-level (vacuous)
TOL_D_REL, TOL_D_ABS, NONMONO_ABS = 0.25, 1e-5, 1e-6                # on the differences (used)
NOISE_R = 70.0     # Mpc/h: D carries no signal beyond (5 R_F); its resolution scatter there = noise floor
INK, GRID = "#0b0b0b", "#e4e3df"
C_NB, C_2, C_4 = "#eb6834", "#2a78d6", "#e87ba4"
C_PAIR = {"2lpt": C_2, "4lpt": C_4}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
                     "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight"})
MASK_NOTE = ("Conditional on the single-stream mask (both runs); not an unconditional "
             "two-point statistic.")


def load_all():
    out = {}
    for p in sorted(glob.glob(os.path.join(STEPS, "steps_N*_l*_z*.npz"))):
        r = json.loads(str(np.load(p)["result"]))
        out[(r["N"], r["level"], float(r["z"]))] = r
    return out


def seed_floor(n, level, z, model, lpt, R, w):
    """Tie-seed floor of step 1c for this run (level-0 files carry 3 seeds)."""
    p = os.path.join(S1, f"c_N{n}_l0_z{z:g}.json")
    if not os.path.exists(p):
        return 0.0
    runs = json.load(open(p))["runs"]
    k = f"{model}|mask:{lpt}|R{R:g}"
    v = runs.get(k, {}).get(w, {}).get("seed_floor")
    return float(v) if v is not None else 0.0


def arr(x):
    return np.array(x, dtype=float)


def conv_mask(hi, lo, floor):
    tol = np.maximum(TOL_REL * np.abs(hi) + TOL_ABS, 3 * floor)
    return np.abs(hi - lo) <= tol


def adjoining(ok, from_end, skip=None):
    """Contiguous True run adjoining one end (large scales); bins flagged in `skip` (empty bins)
    neither stop the run nor count as converged."""
    out = np.zeros_like(ok)
    idx = range(len(ok) - 1, -1, -1) if from_end else range(len(ok))
    for i in idx:
        if skip is not None and skip[i]:
            continue
        if not ok[i]:
            break
        out[i] = True
    return out


def plan(D):
    Ns = sorted({k[0] for k in D})
    zs = sorted({k[2] for k in D})
    n_hi = max(n for (n, l, _) in D if l == 0)
    lo = [n for n in Ns if n < n_hi and any(k[:2] == (n, 0) for k in D)]
    n_lo = max(lo) if lo else None
    ref = [n for (n, l, _) in D if l == 1]
    n_ref = max(ref) if ref else None
    return Ns, zs, n_hi, n_lo, n_ref


def get(D, n, l, z, lpt, R, w):
    try:
        return D[(n, l, z)]["pairs"][lpt]["R"][f"{R:g}"][w]
    except KeyError:
        return None


def xi_conv_run(D, pl, z, lpt, R, w, key_lpt, key_nb, sub=None):
    """PRE-SET run-level test: converged-bin mask for a xi statistic (both runs, both comparisons)."""
    _, _, n_hi, n_lo, n_ref = pl
    ok = None
    for key, model in ((key_lpt, lpt), (key_nb, "nbody")):
        def val(n, l):
            g = get(D, n, l, z, lpt, R, w)
            if g is None:
                return None
            v = g[key] if sub is None else g[sub][key]
            return arr(v["xi"])
        hi = val(n_hi, 0)
        fl = seed_floor(n_hi, 0, z, model, lpt, R, w)
        m = np.isfinite(hi)
        for a, b in ((val(n_hi, 0), val(n_lo, 0) if n_lo else None),
                     (val(n_ref, 1) if n_ref else None, val(n_ref, 0) if n_ref else None)):
            if a is None or b is None:
                m &= False
                continue
            m &= conv_mask(a, b, fl) & np.isfinite(b)
        ok = m if ok is None else ok & m
    return adjoining(ok, from_end=True)


def xi_conv(D, pl, z, lpt, R, w, key_lpt, key_nb, sub=None):
    """Converged-bin mask on the difference D = xi[NB] - xi[LPT] (particle and refinement)."""
    _, _, n_hi, n_lo, n_ref = pl

    def dif(n, l):
        g = get(D, n, l, z, lpt, R, w) if n else None
        if g is None:
            return None
        h = g if sub is None else g[sub]
        return arr(h[key_nb]["xi"]) - arr(h[key_lpt]["xi"])
    hi = dif(n_hi, 0)
    g0 = get(D, n_hi, 0, z, lpt, R, w)
    r = arr((g0 if sub is None else g0[sub])[key_lpt]["r"])
    pairs = [(hi, dif(n_lo, 0)), (dif(n_ref, 1), dif(n_ref, 0))]
    if any(a is None or b is None for a, b in pairs):
        return np.zeros(hi.shape, bool)
    # numerical scatter of D: largest resolution change at r >= NOISE_R, where D carries no signal
    big = np.isfinite(r) & (r >= NOISE_R)
    noise = max([TOL_D_ABS] + [float(np.nanmax(np.abs(a - b)[big])) for a, b in pairs if np.any(big & np.isfinite(a - b))])
    ok = np.isfinite(hi)
    for a, b in pairs:
        # a bin that cannot be compared (absent at the lower resolution) is not converged
        ok &= np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= TOL_D_REL * np.abs(hi) + noise)
    return adjoining(ok, from_end=True, skip=~np.isfinite(hi))


def scalar_conv(D, pl, z, lpt, R, w, key):
    _, _, n_hi, n_lo, n_ref = pl
    vals = {}
    for n, l in ((n_hi, 0), (n_lo, 0), (n_ref, 1), (n_ref, 0)):
        g = get(D, n, l, z, lpt, R, w) if n else None
        vals[(n, l)] = g[key] if g else None
    hi = vals[(n_hi, 0)]
    dp = abs(hi - vals[(n_lo, 0)]) if vals[(n_lo, 0)] is not None else np.inf
    dr = abs(vals[(n_ref, 1)] - vals[(n_ref, 0)]) if None not in (vals[(n_ref, 1)], vals[(n_ref, 0)]) else np.inf
    # departures are 1 - value (value -> 1); the non-monotone part is itself a departure
    tol = NONMONO_ABS if key == "nonmonotone" else TOL_D_REL * abs(1 - hi) + TOL_D_ABS
    return dict(value=hi, departure=(hi if key == "nonmonotone" else 1 - hi), d_particle=dp, d_refine=dr,
                tol=tol, converged=bool(max(dp, dr) <= tol), converged_preset=bool(max(dp, dr) <= TOL_SCALAR))


def rk_conv(D, pl, z, lpt, R, w):
    _, _, n_hi, n_lo, n_ref = pl
    def rk(n, l):
        g = get(D, n, l, z, lpt, R, w) if n else None
        return (arr(g["spectra"]["k"]), arr(g["spectra"]["r"])) if g else (None, None)
    k_hi, r_hi = rk(n_hi, 0)
    fin = np.isfinite(r_hi) & np.isfinite(k_hi)
    diffs = []
    for (ka, ra), (kb, rb) in ((rk(n_hi, 0), rk(n_lo, 0)), (rk(n_ref, 1), rk(n_ref, 0))):
        if ra is None or rb is None:
            return k_hi, r_hi, np.zeros(k_hi.shape, bool)
        fa, fb = np.isfinite(ra) & np.isfinite(ka), np.isfinite(rb) & np.isfinite(kb)
        ra_i = np.interp(k_hi, ka[fa], ra[fa], left=np.nan, right=np.nan)
        rb_i = np.interp(k_hi, kb[fb], rb[fb], left=np.nan, right=np.nan)
        diffs.append(np.abs(ra_i - rb_i))
    small_k = fin & (k_hi <= 2 * np.pi / NOISE_R)
    noise = max([TOL_D_ABS] + [float(np.nanmax(d[small_k])) for d in diffs if np.any(small_k & np.isfinite(d))])
    ok = fin.copy()
    for d in diffs:
        ok &= np.isfinite(d) & (d <= TOL_D_REL * np.abs(1 - r_hi) + noise)
    return k_hi, r_hi, adjoining(ok, from_end=False, skip=~fin)


def attribution(g):
    xd = {k: arr(v["xi"]) for k, v in g["xi_delta"].items()}
    xZ = {k: arr(v["xi"]) for k, v in g["xi_Z"].items()}
    r = arr(g["xi_delta"]["lpt"]["r"])
    T = xd["nbody"] - xd["lpt"]
    out = dict(r=r, T=T,
               Marg_A=xd["remap_lpt_to_nb"] - xd["lpt"], Cop_A=xd["nbody"] - xd["remap_lpt_to_nb"],
               Cop_B=xd["remap_nb_to_lpt"] - xd["lpt"], Marg_B=xd["nbody"] - xd["remap_nb_to_lpt"])
    Dy = arr(g["xiE_nb"]["xi"]) - arr(g["xiE_lpt"]["xi"])
    out.update(Dy=Dy, Lag_1=xZ["nl"] - xZ["ll"], Map_1=xZ["nn"] - xZ["nl"],
               Map_2=xZ["ln"] - xZ["ll"], Lag_2=xZ["nn"] - xZ["ln"],
               Resid=Dy - (xZ["nn"] - xZ["ll"]))
    return out


def shade(ax, r, conv):
    """Grey band over the unconverged (small-scale) range."""
    if not np.any(conv):
        ax.axvspan(np.nanmin(r), np.nanmax(r), color="#bdbcb8", alpha=0.35, lw=0)
        return
    rc = np.nanmin(r[conv])
    ax.axvspan(np.nanmin(r) * 0.9, rc, color="#bdbcb8", alpha=0.35, lw=0)


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def main():
    D = load_all()
    if not D:
        print("no steps results"); return
    pl = plan(D)
    Ns, zs, n_hi, n_lo, n_ref = pl
    Rs = CFG["R_list"]; lpts = CFG["lpt_orders"]
    summ = dict(plan=dict(N=Ns, z=zs, N_hi=n_hi, N_lo=n_lo, N_ref=n_ref,
                          tol=dict(rel=TOL_REL, abs=TOL_ABS, scalar=TOL_SCALAR, rk=TOL_RK)),
                mask_note=MASK_NOTE, lagr={}, eul={}, scalars={}, attrib={})
    lines = ["# Steps 2-5 tables (generated by steps_collect.py)", "",
             f"> {MASK_NOTE}", "",
             f"Headline run: N = {n_hi}, sheet level 0. Particle-number convergence: N = {n_hi} vs {n_lo} "
             f"(level 0). Sheet-refinement convergence: N = {n_ref}, level 1 vs 0. "
             f"Convergence acts on the LPT - N-body DIFFERENCES: |dD| <= {TOL_D_REL}|D| + {TOL_D_ABS:g} "
             f"(xi: D = xi_NB - xi_LPT; r(k): 1 - r; scalars: 1 - value; non-monotone part: <= {NONMONO_ABS:g} absolute). "
             f"The pre-set run-level tolerances (xi: {TOL_REL}|S| + {TOL_ABS}; scalars {TOL_SCALAR}) exceed the "
             "differences themselves and are shown for reference only; the change was made after seeing the "
             "level-0 z = 0 runs and before any refinement result.", ""]

    # ---------------- steps 2 + 3 tables ----------------
    for w in ("mass", "vol"):
        lines += [f"## Step 3: element-paired ranks ({w}-weighted), N = {n_hi}", "",
                  "| z | pair | R | 1 - Spearman | 1 - eta^2 | 1 - R2_iso | non-monotone (eta^2 - R2_iso) | converged (differences) | converged (pre-set) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for z in zs:
            for lpt in lpts:
                for R in Rs:
                    if get(D, n_hi, 0, z, lpt, R, w) is None:
                        continue
                    s = {k: scalar_conv(D, pl, z, lpt, R, w, k) for k in ("spearman", "eta2", "r2_iso", "nonmonotone")}
                    summ["scalars"][f"{w}|z{z:g}|{lpt}|R{R:g}"] = s
                    c = [k for k, v in s.items() if not v["converged"]]
                    c0 = all(v["converged_preset"] for v in s.values())
                    lines.append(f"| {z:g} | {lpt} vs N-body | {R:g} | {1 - s['spearman']['value']:.2e} | "
                                 f"{1 - s['eta2']['value']:.2e} | {1 - s['r2_iso']['value']:.2e} | "
                                 f"{s['nonmonotone']['value']:.1e} | {'yes' if not c else 'NO (' + ', '.join(c) + ')'} | "
                                 f"{'yes' if c0 else 'NO'} |")
        lines.append("")

    for w in ("mass", "vol"):
        lines += [f"## Step 2: Lagrangian copula xi_y ({w}-weighted), N = {n_hi}", "",
                  "| z | pair | R | r_conv [Mpc/h] | max abs(xi_NB - xi_LPT) (r >= r_conv) at r | k_conv [h/Mpc] | max 1 - r(k) (k <= k_conv) | r_conv pre-set run-level |",
                  "|---|---|---|---|---|---|---|---|"]
        for z in zs:
            for lpt in lpts:
                for R in Rs:
                    g = get(D, n_hi, 0, z, lpt, R, w)
                    if g is None:
                        continue
                    r = arr(g["xi_lpt"]["r"]); dx = arr(g["xi_nb"]["xi"]) - arr(g["xi_lpt"]["xi"])
                    cv = xi_conv(D, pl, z, lpt, R, w, "xi_lpt", "xi_nb")
                    k, rk, kc = rk_conv(D, pl, z, lpt, R, w)
                    rc = float(np.nanmin(r[cv])) if cv.any() else np.inf
                    kcv = float(np.nanmax(k[kc])) if kc.any() else 0.0
                    if cv.any():
                        i = np.nanargmax(np.where(cv, np.abs(dx), -1))
                        md, mr = float(dx[i]), float(r[i])
                    else:
                        md, mr = np.nan, np.nan
                    mrk = float(np.nanmax(1 - rk[kc])) if kc.any() else np.nan
                    cv0 = xi_conv_run(D, pl, z, lpt, R, w, "xi_lpt", "xi_nb")
                    rc0 = float(np.nanmin(r[cv0])) if cv0.any() else np.inf
                    summ["lagr"][f"{w}|z{z:g}|{lpt}|R{R:g}"] = dict(r_conv=rc, max_dxi=md, at_r=mr, k_conv=kcv,
                                                                   max_1_minus_rk=mrk, r_conv_preset=rc0)
                    lines.append(f"| {z:g} | {lpt} | {R:g} | {rc:.3g} | {md:+.2e} at {mr:.3g} | {kcv:.3g} | {mrk:.2e} | {rc0:.3g} |")
        lines.append("")

    # ---------------- step 4 + 5 tables ----------------
    for w in ("mass", "vol"):
        lines += [f"## Steps 4-5: Eulerian (mask E) xi and attribution ({w}-weighted), N = {n_hi}", "",
                  "Values at the smallest converged separation r_c and averaged over r >= r_c "
                  "(pair-weighted bins). T = xi_delta[NB] - xi_delta[LPT].", "",
                  "| z | pair | R | r_c | T(r_c) | Marg_A | Cop_A | Cop_B | Marg_B | Dy(r_c) | Lag_1 | Map_1 | Map_2 | Lag_2 | Resid |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for z in zs:
            for lpt in lpts:
                for R in Rs:
                    g = get(D, n_hi, 0, z, lpt, R, w)
                    if g is None or "xi_delta" not in g:
                        continue
                    a = attribution(g)
                    cvd = xi_conv(D, pl, z, lpt, R, w, "lpt", "nbody", sub="xi_delta")
                    cvy = xi_conv(D, pl, z, lpt, R, w, "xiE_lpt", "xiE_nb")
                    cv = cvd & cvy
                    summ["attrib"][f"{w}|z{z:g}|{lpt}|R{R:g}"] = {k: v for k, v in a.items()} | dict(converged=cv)
                    if not cv.any():
                        lines.append(f"| {z:g} | {lpt} | {R:g} | none | " + " | ".join(["-"] * 11) + " |")
                        continue
                    i = int(np.nanargmin(np.where(cv, a["r"], np.inf)))
                    f = lambda k: f"{a[k][i]:+.2e}"
                    lines.append(f"| {z:g} | {lpt} | {R:g} | {a['r'][i]:.3g} | {f('T')} | {f('Marg_A')} | {f('Cop_A')} | "
                                 f"{f('Cop_B')} | {f('Marg_B')} | {f('Dy')} | {f('Lag_1')} | {f('Map_1')} | {f('Map_2')} | "
                                 f"{f('Lag_2')} | {f('Resid')} |")
        lines.append("")

    # ---------------- figures ----------------
    for w in ("mass", "vol"):
        for z in zs:
            # step 2: xi_y and its difference
            fig, ax = plt.subplots(2, len(Rs), figsize=(2.6 * len(Rs), 4.6), sharex=True, squeeze=False)
            for j, R in enumerate(Rs):
                for lpt in lpts:
                    g = get(D, n_hi, 0, z, lpt, R, w)
                    if g is None:
                        continue
                    r = arr(g["xi_lpt"]["r"])
                    if lpt == lpts[0]:
                        ax[0, j].loglog(r, np.abs(arr(g["xi_nb"]["xi"])), color=C_NB, label="N-body")
                    ax[0, j].loglog(r, np.abs(arr(g["xi_lpt"]["xi"])), color=C_PAIR[lpt], ls="--", label=lpt.upper())
                    ax[1, j].semilogx(r, arr(g["xi_nb"]["xi"]) - arr(g["xi_lpt"]["xi"]), color=C_PAIR[lpt], label=f"NB - {lpt.upper()}")
                    shade(ax[1, j], r, xi_conv(D, pl, z, lpt, R, w, "xi_lpt", "xi_nb"))
                shade(ax[0, j], r, xi_conv(D, pl, z, lpts[-1], R, w, "xi_lpt", "xi_nb"))
                ax[0, j].set_title(f"R = {R:g} Mpc/h" if R > 0 else "R = 0 (element)")
                ax[1, j].axhline(0, color=INK, lw=0.6)
                ax[1, j].set_xlabel("|Δq| [Mpc/h]")
            ax[0, 0].set_ylabel("|ξ_y| (Lagrangian)"); ax[1, 0].set_ylabel("Δξ_y")
            ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
            fig.suptitle(f"Step 2: Lagrangian copula ξ_y, {w}-weighted, z = {z:g}, N = {n_hi}. "
                         f"Grey: unconverged. {MASK_NOTE}", fontsize=8)
            savefig(fig, f"steps_2_xi_{w}_z{z:g}")

            # step 2: r(k)
            fig, ax = plt.subplots(1, len(Rs), figsize=(2.6 * len(Rs), 2.6), sharey=True, squeeze=False)
            for j, R in enumerate(Rs):
                for lpt in lpts:
                    if get(D, n_hi, 0, z, lpt, R, w) is None:
                        continue
                    k, rk, kc = rk_conv(D, pl, z, lpt, R, w)
                    ax[0, j].semilogx(k, 1 - rk, color=C_PAIR[lpt], label=lpt.upper())
                    if kc.any():
                        ax[0, j].axvspan(np.nanmax(k[kc]), np.nanmax(k) * 1.05, color="#bdbcb8", alpha=0.35, lw=0)
                    else:
                        ax[0, j].axvspan(np.nanmin(k), np.nanmax(k), color="#bdbcb8", alpha=0.35, lw=0)
                ax[0, j].set_yscale("symlog", linthresh=1e-4)
                ax[0, j].set_title(f"R = {R:g}"); ax[0, j].set_xlabel("k_q [h/Mpc]")
            ax[0, 0].set_ylabel("1 - r(k)"); ax[0, 0].legend(fontsize=7)
            fig.suptitle(f"Step 2: cross-correlation of LPT and N-body scores, {w}-weighted, z = {z:g}. "
                         f"Grey: unconverged. {MASK_NOTE}", fontsize=8)
            savefig(fig, f"steps_2_rk_{w}_z{z:g}")

            # steps 4-5: attribution
            fig, ax = plt.subplots(2, len(Rs), figsize=(2.6 * len(Rs), 4.8), sharex=True, squeeze=False)
            for j, R in enumerate(Rs):
                g = get(D, n_hi, 0, z, lpts[-1], R, w)
                g2 = get(D, n_hi, 0, z, lpts[0], R, w)
                for lpt, gg in ((lpts[0], g2), (lpts[-1], g)):
                    if gg is None or "xi_delta" not in gg:
                        continue
                    a = attribution(gg)
                    ls = "-" if lpt == lpts[-1] else ":"
                    ax[0, j].semilogx(a["r"], a["T"], color=INK, ls=ls, label=f"total ({lpt})")
                    ax[0, j].semilogx(a["r"], a["Marg_A"], color=C_2, ls=ls, label="marginal (A)")
                    ax[0, j].semilogx(a["r"], a["Marg_B"], color=C_2, ls=ls, alpha=0.45, label="marginal (B)")
                    ax[0, j].semilogx(a["r"], a["Cop_A"], color=C_NB, ls=ls, label="copula (A)")
                    ax[0, j].semilogx(a["r"], a["Cop_B"], color=C_NB, ls=ls, alpha=0.45, label="copula (B)")
                    ax[1, j].semilogx(a["r"], a["Dy"], color=INK, ls=ls, label=f"Δξ_y^E ({lpt})")
                    ax[1, j].semilogx(a["r"], a["Lag_1"], color=C_4, ls=ls, label="Lagr. copula (1)")
                    ax[1, j].semilogx(a["r"], a["Lag_2"], color=C_4, ls=ls, alpha=0.45, label="Lagr. copula (2)")
                    ax[1, j].semilogx(a["r"], a["Map_1"], color="#1baf7a", ls=ls, label="mapping (1)")
                    ax[1, j].semilogx(a["r"], a["Map_2"], color="#1baf7a", ls=ls, alpha=0.45, label="mapping (2)")
                    ax[1, j].semilogx(a["r"], a["Resid"], color="#8a8984", ls=ls, label="residual")
                    cv = xi_conv(D, pl, z, lpt, R, w, "lpt", "nbody", sub="xi_delta") & xi_conv(D, pl, z, lpt, R, w, "xiE_lpt", "xiE_nb")
                    if lpt == lpts[-1]:
                        shade(ax[0, j], a["r"], cv); shade(ax[1, j], a["r"], cv)
                for a_ in ax[:, j]:
                    a_.axhline(0, color=INK, lw=0.6)
                ax[0, j].set_title(f"R = {R:g}"); ax[1, j].set_xlabel("|Δx| [Mpc/h]")
            ax[0, 0].set_ylabel("Δξ_δ (Eulerian, E)"); ax[1, 0].set_ylabel("Δξ_y (Eulerian, E)")
            ax[0, -1].legend(fontsize=6, loc="upper right"); ax[1, -1].legend(fontsize=6, loc="upper right")
            fig.suptitle(f"Steps 4-5: attribution, {w}-weighted, z = {z:g}, N = {n_hi} (solid {lpts[-1].upper()}, "
                         f"dotted {lpts[0].upper()}); orderings A/B and 1/2 differ = non-uniqueness. "
                         f"Grey: unconverged. {MASK_NOTE}", fontsize=7)
            savefig(fig, f"steps_45_attrib_{w}_z{z:g}")

        # step 3 vs R
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        for z, ls in zip(zs, ("-", "--", ":")):
            for lpt in lpts:
                for n in sorted({k[0] for k in D if k[1] == 0}):
                    vals = [get(D, n, 0, z, lpt, R, w) for R in Rs]
                    if any(v is None for v in vals):
                        continue
                    alpha = 1.0 if n == n_hi else 0.35
                    lab = f"{lpt.upper()} z={z:g}" if n == n_hi else None
                    ax[0].plot(Rs, [1 - v["spearman"] for v in vals], color=C_PAIR[lpt], ls=ls, alpha=alpha, marker="o", ms=3, label=lab)
                    ax[1].plot(Rs, [max(v["nonmonotone"], 1e-8) for v in vals], color=C_PAIR[lpt], ls=ls, alpha=alpha, marker="o", ms=3)
        ax[0].set_yscale("log"); ax[1].set_yscale("log")
        ax[0].set_xlabel("R [Mpc/h]"); ax[1].set_xlabel("R [Mpc/h]")
        ax[0].set_ylabel("1 - Spearman"); ax[1].set_ylabel("η² - R²_iso (non-monotone)")
        ax[0].legend(fontsize=7)
        fig.suptitle(f"Step 3: element-paired ranks, {w}-weighted (faint: lower N). {MASK_NOTE}", fontsize=8)
        savefig(fig, f"steps_3_ranks_{w}")

    with open(os.path.join(STEPS, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    with open(os.path.join(OUT, "steps_tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
