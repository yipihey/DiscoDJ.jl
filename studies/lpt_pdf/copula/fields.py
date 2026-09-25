"""Per-element products of one snapshot: element volumes, flip / multistream flags and
smoothed sheet densities at the element centroids."""
import os, time
import numpy as np

from common import CFG, snap_path, SCRATCH
import sheet
from cosmo import W_TH
import core

L = CFG["box_L"]
NG = CFG["sheet_mesh"]


def load_psi(model, n, z, level, seed=None):
    psi = np.load(snap_path(model, n, z, seed))
    return sheet.refine(psi, level)


class Snapshot:
    """Geometry of one (model, N, z, refinement, seed) sheet; smoothed fields on demand."""

    def __init__(self, model, n, z, level, seed=None, need_mesh=True):
        t0 = time.time()
        self.model, self.n0, self.z, self.level = model, n, z, level
        self.psi = load_psi(model, n, z, level, seed)
        self.n = self.psi.shape[1]
        self.Vq = (L / self.n)**3
        self.V, flip = sheet.elements(self.psi, L)
        self.nflip = int(flip.sum())
        self.ms = sheet.multistream_elements(self.psi, L, flip, NG)
        del flip
        self._rhok = None
        self.need_mesh = need_mesh
        self.t_geom = time.time() - t0

    def _mesh(self):
        if self._rhok is None:
            t0 = time.time()
            rho = sheet.sheet_density(self.psi, L, NG)
            self.mesh_mean = float(rho.mean())
            self._rhok = core.rfftn(rho)
            self.t_mesh = time.time() - t0
        return self._rhok

    def values(self, R):
        """rho_R at each element: R = 0 -> element stream density Vq / V_e;
        R > 0 -> top-hat (radius R) smoothed sheet density at the element centroid."""
        if R == 0:
            return self.Vq / self.V
        rk = self._mesh()
        kx, ky, kz = core.kgrid(NG, L, np.float64)
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        f = core.irfftn(rk * W_TH(k * R), NG)
        return sheet.sample_centroids(self.psi, L, f)

    def drop_psi(self):
        self.psi = None
