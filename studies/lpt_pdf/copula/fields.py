"""Per-element products of one snapshot, computed by DiscoDJNative's periodic sheet kernels
(native/DiscoDJNative/src/field/sheet_periodic.jl) through sheet_products.jl.

Element  = Lagrangian cube of the (refined) lattice, 6 tetrahedra.
Mask     = single-stream elements: no inverted tetrahedron AND exact stream count 1 at the
           element centroid (number of sheet tetrahedra containing it).
Values   = R = 0: element stream density Vq / V_e;  R > 0: top-hat smoothed sheet density
           (all streams) at the element centroid.
"""
import os, subprocess, time
import h5py
import numpy as np

from common import CFG, snap_path, SCRATCH

L = CFG["box_L"]
NG = CFG["sheet_mesh"]
HERE = os.path.dirname(os.path.abspath(__file__))
DJN = os.path.normpath(os.path.join(HERE, "..", "..", "..", "native", "DiscoDJNative"))
JULIA = os.environ.get("JULIA", "/opt/jl/bin/julia")
JENV = dict(os.environ, JULIA_DEPOT_PATH=os.environ.get("JULIA_DEPOT_PATH", "/opt/jdepot"),
            JULIA_PKG_SERVER=os.environ.get("JULIA_PKG_SERVER", ""))


def product_path(model, n, z, level, seed=None, tag="full"):
    s = "" if seed in (None, CFG["phase_seed"]) else f"_seed{seed}"
    return os.path.join(SCRATCH, f"prod_{tag}_{model}_N{n}_z{z:g}_l{level}{s}.h5")


def make_product(model, n, z, level, seed=None, with_R=True):
    tag = "full" if with_R else "geom"
    out = product_path(model, n, z, level, seed, tag)
    if os.path.exists(out):
        return out
    Rs = ",".join(f"{R:g}" for R in CFG["R_list"] if R > 0) if with_R else ""
    cmd = [JULIA, "-t", str(os.cpu_count()), f"--project={DJN}", os.path.join(HERE, "sheet_products.jl"),
           snap_path(model, n, z, seed), str(level), str(NG), str(L), Rs, out + ".tmp"]
    t0 = time.time()
    r = subprocess.run(cmd, env=JENV, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sheet_products failed ({model} N{n} z{z} l{level}):\n{r.stdout}\n{r.stderr}")
    os.replace(out + ".tmp", out)
    print(f"  [julia] {os.path.basename(out)} {time.time() - t0:.0f}s", flush=True)
    return out


class Snapshot:
    """Sheet products of one (model, N, z, refinement, seed)."""

    def __init__(self, model, n, z, level, seed=None, with_R=True):
        self.model, self.n0, self.z, self.level = model, n, z, level
        self.path = make_product(model, n, z, level, seed, with_R)
        with h5py.File(self.path, "r") as f:
            self.V = f["V"][...]
            nflip = f["nflip"][...]
            self.nstream = f["nstream"][...]
            self.attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
        self.n = self.V.shape[0]
        self.Vq = (L / self.n)**3
        self.nflip = int(np.count_nonzero(nflip))
        self.ms = (nflip > 0) | (self.nstream != 1)       # not single-stream
        self.mesh_mean = self.attrs.get("mesh_mean_density")

    def values(self, R):
        if R == 0:
            return self.Vq / self.V
        with h5py.File(self.path, "r") as f:
            return f[f"rhoR_{float(R)}"][...]


def cleanup(n, z, level):
    """Delete the (large) product files of one (N, z, level) once all tests used them."""
    import glob
    for p in glob.glob(os.path.join(SCRATCH, f"prod_*_N{n}_z{z:g}_l{level}*.h5")):
        os.remove(p)
