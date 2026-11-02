#!/usr/bin/env python3
"""Sphere-variance response with the added band power separated, from the per-point sub-lattice -> results/fixedmass_response.json

    δ_UV = δ_IR + Δ  ⇒  ⟨δ_UV²⟩ − ⟨δ_IR²⟩ = ⟨Δ²⟩ + 2⟨Δ δ_IR⟩          (same sphere centres in both members)

⟨Δ²⟩: the added power (linear limit: D²σ²_band(R), band k_split ≤ k < k_uv, top-hat window).
2⟨Δ δ_IR⟩: coherent change of the existing structure.  Response ≡ 2⟨Δ δ_IR⟩/⟨δ_IR²⟩.  At linear order the band is statistically independent of
the IR modes, so this vanishes unless modes couple.  One UV/IR pair; errors: delete-one jackknife over 4³ sub-volumes (spatial scatter, not cosmic variance).
"""
import json, os, sys
import h5py, numpy as np
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing") + "/fm_fiducial/"
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def load(f):
    out = {}
    with h5py.File(D0 + f) as h:
        for k in h:
            if not k.startswith("e") or not isinstance(h[k], h5py.Group) or "D" not in h[k].attrs:
                continue
            g = h[k]
            out[(int(g.attrs["order"]), round(float(g.attrs["D"]), 4))] = g["sub"][0], g
            s = g["sub"]; R = g["sub_R"][()]
            out[(int(g.attrs["order"]), round(float(g.attrs["D"]), 4))] = (s[:, 0, :].copy(), R)   # (points, nsub): channel "mu"
    return out

LPT = {"IR": load("fm_lpt_IR_L1_S0_s+1.h5"), "UV": load("fm_lpt_UV_L1_S1001_s+1.h5")}
PM = {"IR": load("fm_pm_IR_L1_S0_s+1.h5"), "UV": load("fm_pm_UV_L1_S1001_s+1.h5")}
get = lambda mem, m, D: (PM if m == 0 else LPT)[mem].get((m, round(D, 4)))
Dall = sorted({k[1] for k in PM["UV"]})
s0 = get("UV", 0, Dall[0]); Rs = s0[1]; N = s0[0].shape[0]; Nc = round(N ** (1 / 3))
print("sub radii:", np.round(Rs, 2).tolist(), "points", N, "per side", Nc)

# band linear variance from the study's spectrum, with the sharp band of config/fiducial.toml
import re
cfg = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "fiducial.toml")).read()
ks, kuv = (float(re.search(rf"^{n}\s*=\s*([0-9.]+)", cfg, re.M).group(1)) for n in ("k_split", "k_uv"))
with h5py.File(D0 + "../tl_full/tl/ops_sharp_L1.h5") as h:
    pk, pP = h["pk_k"][()], h["pk_P"][()]
def sig2(R, a, b):
    k = np.exp(np.linspace(np.log(a), np.log(b), 4000)); P = np.exp(np.interp(np.log(k), np.log(pk), np.log(pP)))
    x = k * R; W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.trapz(P * W ** 2 * k ** 3 / (2 * np.pi ** 2), np.log(k))

block = (np.arange(N) // Nc ** 2 * 4 // Nc) * 16 + ((np.arange(N) // Nc) % Nc * 4 // Nc) * 4 + (np.arange(N) % Nc) * 4 // Nc
def jk(f):
    full = f(np.ones(N, bool)); parts = np.array([f(block != b) for b in range(64)])
    return float(full), float(np.sqrt(63 / 64 * ((parts - parts.mean()) ** 2).sum()))

out = dict(k_split=ks, k_uv=kuv, rows=[])
print(f"band {ks} ≤ k < {kuv}")
print("R     D    m | <dIR2>     <D2>/<dIR2> ±     <D2>/(D²σ²band) | response=2<D dIR>/<dIR2> ±     | corr(D,dIR) | total UV/IR-1 | linear-only predicted UV/IR-1")
idx = [i for i, r in enumerate(Rs) if r >= 3.5 and r <= 25]
for i in idx:
    R = Rs[i]; sb = sig2(R, ks, kuv); sl = sig2(R, 1e-4, ks)
    for D in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        for m in (3, 7, 0):
            u, r = get("UV", m, D), get("IR", m, D)
            if u is None or r is None: continue
            du = u[0][:, i].astype(np.float64) - 1; di = r[0][:, i].astype(np.float64) - 1
            d = du - du.mean(); e = di - di.mean(); dl = d - e
            v = (e ** 2).mean(); a = (dl ** 2).mean()
            add = jk(lambda s: (dl[s] ** 2).mean() / (e[s] ** 2).mean())
            resp = jk(lambda s: 2 * (dl[s] * e[s]).mean() / (e[s] ** 2).mean())
            corr = (dl * e).mean() / np.sqrt(a * v)
            tot = (d ** 2).mean() / v - 1
            lin = sb / sl
            print(f"{R:5.2f} {D:.1f} {m} | {v:9.4f}  {add[0]:7.3f} {add[1]:.3f}  {a / (D ** 2 * sb):7.3f} | {resp[0]:+8.4f} {resp[1]:.4f} | {corr:+.3f} | {tot:+.3f} | {lin:.3f}")
            out["rows"].append(dict(R=float(R), D=D, model=m, var_ir=v, add=add, add_over_linear=a / (D ** 2 * sb), response=resp, corr=float(corr), total=float(tot), linear_ratio=lin))
json.dump(out, open(os.path.join(here, "fixedmass_response.json"), "w"), indent=0, default=float)
