"""Displacement snapshots psi(q) = x(q) - q (float64, (3,n,n,n)) for N-body and LPT.

  python snapshots.py nbody N        # PM run from 2LPT ICs at a_initial, saves every z in z_list
  python snapshots.py lpt N          # 1LPT, 2LPT, 4LPT (nlpt engine, de-aliased) at every z
  python snapshots.py za N --seed S  # Zel'dovich only, extra phase seeds (test 1d band)
"""
import argparse, json, time
import numpy as np

from common import CFG, a_of_z, snap_path, SCRATCH
import core
import nlpt
from cosmo import Cosmology

L = CFG["box_L"]


def linear_field(cosmo, n, seed):
    if seed == CFG["phase_seed"]:
        import config as C0
        ph = core.master_phases(C0.N_MASTER, C0.SEED, C0.PHASES)
        return core.delta_k(cosmo, core.subgrid_modes(ph, n), n, L, CFG["R_F"])
    ph = core.master_phases(n, seed)          # independent realisation directly at n
    return core.delta_k(cosmo, ph, n, L, CFG["R_F"])


def minimg(x, q):
    d = x - q
    d -= L * np.round(d / L)
    return d


def run_nbody(n):
    cosmo = Cosmology()
    dk = linear_field(cosmo, n, CFG["phase_seed"])
    psi1, psi2 = core.lpt2(dk, n, L)
    ai = CFG["a_initial"]
    x, p = core.lpt_state(cosmo, psi1, psi2, ai, L)
    del psi1, psi2
    q = core.lattice(n, L)
    a_snaps = sorted(a_of_z(z) for z in CFG["z_list"])
    zs = {round(a_of_z(z), 12): z for z in CFG["z_list"]}
    np.save(snap_path("nbody", n, zs[round(ai, 12)]), minimg(x, q))
    # log-spaced KDK steps with every snapshot a as a node
    a_s = np.unique(np.concatenate([np.geomspace(ai, 1.0, CFG["nbody"]["steps"] + 1), a_snaps]))
    pm = core.PM(cosmo, L, CFG["nbody"]["mesh_factor"] * n)
    acc = pm.accel(x, a_s[0])
    t0 = time.time()
    for s in range(len(a_s) - 1):
        a0, a1 = a_s[s], a_s[s + 1]
        am = 0.5 * (a0 + a1)
        p += cosmo.kick_factor(a0, am) * acc
        x += cosmo.drift_factor(a0, a1) * p
        np.mod(x, L, out=x)
        acc = pm.accel(x, a1)
        p += cosmo.kick_factor(am, a1) * acc
        key = round(a1, 12)
        if key in zs and a1 > ai:
            np.save(snap_path("nbody", n, zs[key]), minimg(x, q))
            print(f"nbody N={n} z={zs[key]:g} saved ({time.time()-t0:.0f}s, {s+1} steps)", flush=True)


def run_lpt(n, models=("1lpt", "2lpt", "4lpt"), seed=None):
    seed = CFG["phase_seed"] if seed is None else seed
    cosmo = Cosmology()
    dk = linear_field(cosmo, n, seed)
    order = max(int(m[0]) for m in models)
    shapes = nlpt.compute_shapes(dk, n, L, dealias=True, n_order=max(order, 2))
    mods = nlpt.models(cosmo)
    for m in models:
        for z in CFG["z_list"]:
            a = a_of_z(z)
            psi = np.zeros((3, n, n, n))
            for g, key in mods[m]:
                psi += g(a) * shapes[key].astype(np.float64)
            np.save(snap_path(m, n, z, seed), psi)
        print(f"{m} N={n} seed={seed} saved", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["nbody", "lpt", "za"])
    ap.add_argument("N", type=int)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.kind == "nbody":
        run_nbody(a.N)
    elif a.kind == "lpt":
        run_lpt(a.N)
    else:
        run_lpt(a.N, models=("1lpt",), seed=a.seed)
