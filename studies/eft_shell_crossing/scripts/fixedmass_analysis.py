#!/usr/bin/env python3
"""Fixed-mass sheet statistics of the fiducial pair -> results/fixedmass_summary.json (src/fixedmass.jl for definitions).

    python3 scripts/fixedmass_analysis.py [scratch]          (needs h5py, numpy; works on whichever fm_* files exist, including .partial)

Checks (budget of the spheres, exact kinematics, no fit):
  d<δ²>/dD = 2<δ μ̇>,  μ̇ = -flux;     d²<δ²>/dD² = 2<μ̇²> + 2<δ μ̈>,  μ̈ = stress - acc;     mean μ = 1;     counts = ng³
Science:
  variance response UV/IR - 1, F(R, μ) = P(μ_R >= μ) and its instantaneous ∂_D F, ∂_R F; ΔF = F_UV - F_IR;
  conditional budget E[X | μ] of stress, bulk, disp, grav, acc; 3LPT and 7LPT against PM.
7LPT stops at D = 0.7 for the UV member by design (lpt_max_D): the map has diverged beyond.
"""
import glob, json, os, sys
import h5py, numpy as np
D0 = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TERMS = ("count", "mu", "shell", "flux", "flux2", "stress", "bulk", "disp", "grav", "acc")
CH = ("mu", "shell", "flux", "stress", "bulk", "disp", "grav", "acc")

def load(pat):
    fs = sorted(glob.glob(os.path.join(D0, "fm_fiducial", pat)))
    if not fs:
        return {}
    out = {}
    with h5py.File(fs[0]) as h:
        for k in h:
            if not k.startswith("e") or "D" not in h[k].attrs:
                continue
            g = h[k]; o = int(g.attrs["order"]); D = float(g.attrs["D"])
            H = np.moveaxis(g["hist"][()], (0, 1, 2), (2, 1, 0)); M = np.moveaxis(g["tmom"][()], (0, 1, 2), (2, 1, 0))
            if o == 0:     # PM: a = (3/2D²) g − (3/2D) v, so ∇·(ρa) = (3/2D²) grav − (3/2D) flux
                H[:, :, TERMS.index("acc")] = 1.5 / D ** 2 * H[:, :, TERMS.index("grav")] - 1.5 / D * H[:, :, TERMS.index("flux")]
                M[:, CH.index("acc"), :2] = 1.5 / D ** 2 * M[:, CH.index("grav"), :2] - 1.5 / D * M[:, CH.index("flux"), :2]
            out[(o, round(D, 4))] = dict(R=g["R"][()], edges=g["edges"][()].T, hist=H, mom=g["mom"][()].T, tmom=M, scal={a: float(g.attrs[a]) for a in g.attrs if a not in ("D", "order", "stride")})
    return out

RUN = {"lptIR": load("fm_lpt_IR_L1_S0_s+1.h5*"), "lptUV": load("fm_lpt_UV_L1_S1001_s+1.h5*"), "pmIR": load("fm_pm_IR_L1_S0_s+1.h5*"), "pmUV": load("fm_pm_UV_L1_S1001_s+1.h5*")}
print({k: len(v) for k, v in RUN.items()})
get = lambda mem, model, D: RUN[("pm" if model == 0 else "lpt") + mem].get((model, round(D, 4)))
out = {"available": {k: sorted(f"{a[0]}:{a[1]}" for a in v) for k, v in RUN.items()}}
Dall = sorted({k[1] for v in RUN.values() for k in v})
RIDX = (5, 12, 20, 28, 36)
s0 = next(iter(next(v for v in RUN.values() if v).values()))
Rg = s0["R"]; print("radii used:", np.round(Rg[list(RIDX)], 1).tolist(), "of", len(Rg))

print("\n== Budget: mean mu - 1, counts, multistream fractions")
for mem in ("IR", "UV"):
    for m in (3, 7, 0):
        for D in Dall:
            r = get(mem, m, D)
            if r is None: continue
            mm = np.abs(r["tmom"][:, 0, 0] - 1).max()
            c = r["hist"][:, :, 0].sum(1).ptp()
            out.setdefault("sheet", []).append(dict(member=mem, model=m, D=D, ms_mass=r["scal"].get("ms_mass_frac"), ms_vol=r["scal"].get("ms_volume_frac"), mean_mu_err=float(mm), count_ptp=float(c)))
            if D in (0.1, 0.3, 0.5, 1.0):
                print(f" {mem} m={m} D={D:.3f} |<mu>-1|max {mm:.1e} countspread {c:.0f} ms mass {r['scal'].get('ms_mass_frac', float('nan')):.3f} ms vol {r['scal'].get('ms_volume_frac', float('nan')):.3f} <tr sigma> {r['scal'].get('mean_trsigma_mass', float('nan')):.3f}")

print("\n== Closure of d<δ²>/dD = 2<δ μ̇> by central difference (rel. error at chosen R; coarse epochs limit it)")
for mem in ("IR", "UV"):
    for m in (3, 7, 0):
        ds = [D for D in Dall if get(mem, m, D) is not None]
        for a, b, c in zip(ds[:-2], ds[1:-1], ds[2:]):
            ra, rb, rc = get(mem, m, a), get(mem, m, b), get(mem, m, c)
            h1, h2 = b - a, c - b          # three-point derivative on the unevenly spaced epochs
            fd = (-h2 / (h1 * (h1 + h2))) * ra["mom"][:, 1] + ((h2 - h1) / (h1 * h2)) * rb["mom"][:, 1] + (h1 / (h2 * (h1 + h2))) * rc["mom"][:, 1]
            pred = -2 * rb["tmom"][:, CH.index("flux"), 1]
            err = np.abs(fd - pred) / np.abs(pred)
            out.setdefault("closure1", []).append(dict(member=mem, model=m, D=b, err=err[list(RIDX)].tolist()))
            if b in (0.15, 0.3, 0.5):
                print(f" {mem} m={m} D={b:.3f} rel.err " + " ".join(f"{x:.2e}" for x in err[list(RIDX)]))

print("\n== Second-order budget: d²<δ²>/dD² = 2<μ̇²> + 2<δ(stress − acc)>, three-point on the uneven epochs; rel. error")
for mem in ("IR", "UV"):
    for m in (3, 0):
        ds = [D for D in Dall if get(mem, m, D) is not None]
        for a, b, c in zip(ds[:-2], ds[1:-1], ds[2:]):
            ra, rb, rc = get(mem, m, a), get(mem, m, b), get(mem, m, c)
            h1, h2 = b - a, c - b
            d2 = 2 * (h2 * ra["mom"][:, 1] - (h1 + h2) * rb["mom"][:, 1] + h1 * rc["mom"][:, 1]) / (h1 * h2 * (h1 + h2))
            t = rb["tmom"]
            pred = 2 * t[:, CH.index("flux"), 2] + 2 * (t[:, CH.index("stress"), 1] - t[:, CH.index("acc"), 1])
            err = np.abs(d2 - pred) / np.abs(pred)
            out.setdefault("closure2", []).append(dict(member=mem, model=m, D=b, err=err[list(RIDX)].tolist()))
            if b in (0.15, 0.25, 0.4, 0.7):
                print(f" {mem} m={m} D={b:.3f} rel.err " + " ".join(f"{x:.2e}" for x in err[list(RIDX)]))

print("\n== Variance response <δ²>_UV/<δ²>_IR - 1 by radius")
for D in (0.2, 0.3, 0.5, 0.7, 1.0):
    for m in (3, 7, 0):
        u, i = get("UV", m, D), get("IR", m, D)
        if u is None or i is None: continue
        v = u["mom"][:, 1] / i["mom"][:, 1] - 1
        print(f" D={D:.1f} m={m} " + " ".join(f"{x:+.4f}" for x in v[list(RIDX)]))
        out.setdefault("var_response", []).append(dict(D=D, model=m, R=Rg.tolist(), resp=v.tolist()))

def ccdf(r, j):
    c = r["hist"][j, ::-1, 0].cumsum()[::-1] / r["hist"][j, :, 0].sum()
    return r["edges"][j], c

def at_mu(r, j, mus):
    e, c = ccdf(r, j)
    c = np.append(c, 0.0)
    return np.interp(np.log(mus), e, c)      # edges are log μ

print("edges at R index 20, D first lpt:", np.round(s0["edges"][20][[0, 128, 256]], 3).tolist())
print("\n== F(R, mu) = P(mu_R >= mu): DF = F_UV - F_IR, for mu = .5, 1, 2, 4, 8")
MU = np.array([0.5, 1, 2, 4, 8.0])
for D in (0.3, 0.5, 0.7, 1.0):
    for m in (3, 7, 0):
        u, i = get("UV", m, D), get("IR", m, D)
        if u is None or i is None: continue
        for j in RIDX[:3]:
            a, b = at_mu(u, j, MU), at_mu(i, j, MU)
            print(f" D={D:.1f} m={m} R={Rg[j]:5.1f} F_UV " + " ".join(f"{x:.4f}" for x in a) + " | DF " + " ".join(f"{x:+.4f}" for x in a - b))
            out.setdefault("F", []).append(dict(D=D, model=m, R=float(Rg[j]), mu=MU.tolist(), F_uv=a.tolist(), dF=(a - b).tolist()))

def cond(r, j, t):
    c = r["hist"][j, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return r["hist"][j, :, TERMS.index(t)] / c

print("\n== Instantaneous dF/dD (from flux) vs three-point difference of F, at fixed mu; PM UV, R=10")
j = RIDX[1]
ds = [D for D in Dall if get("UV", 0, D) is not None]
for a, b, c in zip(ds[:-2], ds[1:-1], ds[2:]):
    if b not in (0.2, 0.25, 0.4, 0.5, 0.7): continue
    ra, rb, rc = get("UV", 0, a), get("UV", 0, b), get("UV", 0, c)
    h1, h2 = b - a, c - b
    Fa, Fb, Fc = at_mu(ra, j, MU[:3]), at_mu(rb, j, MU[:3]), at_mu(rc, j, MU[:3])
    fd = (-h2 / (h1 * (h1 + h2))) * Fa + ((h2 - h1) / (h1 * h2)) * Fb + (h1 / (h2 * (h1 + h2))) * Fc
    e = rb["edges"][j]; c0 = rb["hist"][j, :, 0]; dl = e[1] - e[0]; mid = 0.5 * (e[1:] + e[:-1])
    p = c0 / c0.sum() / dl / np.exp(mid)                                # p(μ)
    with np.errstate(invalid="ignore", divide="ignore"):
        md = -rb["hist"][j, :, TERMS.index("flux")] / c0
    inst = np.interp(np.log(MU[:3]), mid, np.nan_to_num(p * md))
    print(f" D={b:.3f} mu=.5,1,2 FD " + " ".join(f"{x:+.3f}" for x in fd) + " | inst " + " ".join(f"{x:+.3f}" for x in inst))
    out.setdefault("dFdD", []).append(dict(D=b, R=float(Rg[j]), mu=MU[:3].tolist(), fd=fd.tolist(), inst=inst.tolist()))

print("\n== Conditional budget, PM UV, E[X | mu]: dmu/dD, stress, of which bulk and disp, acc; mu-ddot = stress - acc")
for D in (0.3, 0.5, 0.7, 1.0):
    r = get("UV", 0, D)
    if r is None: continue
    for j in RIDX[:2]:
        c = r["hist"][j, :, 0]; e = r["edges"][j]; mid = np.exp(0.5 * (e[1:] + e[:-1]))
        for lo, hi in ((0.3, 0.7), (0.7, 1.5), (2, 4), (8, 32)):
            s = (mid >= lo) & (mid < hi) & (c > 0)
            if s.sum() == 0 or c[s].sum() < 100: continue
            w = c[s] / c[s].sum()
            vals = {t: float((w * cond(r, j, t)[s]).sum()) for t in ("flux", "stress", "bulk", "disp", "grav", "acc")}
            print(f" D={D:.1f} R={Rg[j]:5.1f} mu {lo}-{hi}: mudot {-vals['flux']:+.3f} stress {vals['stress']:+.3f} bulk {vals['bulk']:+.3f} disp {vals['disp']:+.3f} remainder {vals['stress'] - vals['bulk'] - vals['disp']:+.3f} acc {vals['acc']:+.3f} | mu-ddot {vals['stress'] - vals['acc']:+.3f} disp/|stress-acc| {abs(vals['disp'] / (vals['stress'] - vals['acc'])) if vals['stress'] != vals['acc'] else 0:.2f}")
            out.setdefault("budget", []).append(dict(D=D, R=float(Rg[j]), mu=[lo, hi], **vals))

with open(os.path.join(here, "fixedmass_summary.json"), "w") as f:
    json.dump(out, f, indent=0, default=float)
