"""Fixed-amplitude top-hat-filtered ICs, 2LPT, PM N-body, density PDFs and
shell-crossing diagnostics.  numpy/scipy.fft + numba; lengths in Mpc/h,
time in 1/H0.
"""
import os
import numpy as np
import scipy.fft as sfft
import numba as nb

from cosmo import Cosmology, W_TH

NTHREADS = int(os.environ.get("NTHREADS", os.cpu_count()))
nb.set_num_threads(NTHREADS)


def rfftn(x):
    return sfft.rfftn(x, workers=NTHREADS)


def irfftn(x, n):
    return sfft.irfftn(x, s=(n, n, n), workers=NTHREADS)


def kgrid(n, L, dtype=np.float32):
    """Compact broadcastable k-vectors for an rfft grid of size n^3."""
    kf = 2 * np.pi / L
    k = (np.fft.fftfreq(n, 1.0 / n) * kf).astype(dtype)
    kz = (np.fft.rfftfreq(n, 1.0 / n) * kf).astype(dtype)
    return k[:, None, None], k[None, :, None], kz[None, None, :]


# ---------------------------------------------------------------------------
# Initial conditions: fixed amplitudes (Angulo & Pontzen 2016), phases shared
# across all resolutions via one master grid.
# ---------------------------------------------------------------------------
def master_phases(nmaster, seed, path=None):
    """Unit-modulus Hermitian phase field e^{i theta_k} on an nmaster^3 rfft grid.

    Built from real white noise so Hermitian symmetry is exact; every
    lower-resolution run takes the sub-cube of these modes, so all
    resolutions share identical phases for every mode they represent."""
    if path and os.path.exists(path):
        return np.load(path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((nmaster,) * 3, dtype=np.float32)
    wk = rfftn(w).astype(np.complex64)
    del w
    amp = np.abs(wk)
    amp[amp == 0] = 1
    wk /= amp
    wk[0, 0, 0] = 0
    if path:
        np.save(path, wk)
    return wk


def subgrid_modes(master, n):
    """Extract modes |i|,|j| < n/2, 0 <= l < n/2 of the master rfft grid into
    an n^3 rfft grid.  Nyquist planes of the small grid are zeroed."""
    nm = master.shape[0]
    h = n // 2
    idx = np.r_[0:h, nm - h + 1:nm]          # frequencies -h+1 .. h-1
    out = np.zeros((n, n, h + 1), dtype=np.complex64)
    sub = np.asarray(master[idx][:, idx][:, :, :h])
    dst = np.r_[0:h, n - h + 1:n]
    out[np.ix_(dst, dst, np.arange(h))] = sub
    return out


def delta_k(cosmo, phases_n, n, L, R_f, paired=False):
    """Fixed-amplitude linear density (a=1) in DFT convention:
    delta_hat = N^3 sqrt(P(k)/V) W_TH(k R_f) e^{i theta}."""
    kx, ky, kz = kgrid(n, L, np.float64)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    amp = n**3 * np.sqrt(cosmo.Pk(k) / L**3) * W_TH(k * R_f)
    d = (amp * phases_n).astype(np.complex64)
    d[0, 0, 0] = 0
    return -d if paired else d


# ---------------------------------------------------------------------------
# 2LPT
# ---------------------------------------------------------------------------
def lpt2(dk, n, L):
    """Return psi1, psi2 (each (3,n,n,n) float32) with
    x = q + D1 psi1 + D2 psi2  (D2 ~ -3/7 D1^2)."""
    kx, ky, kz = kgrid(n, L)
    ks = (kx, ky, kz)
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1
    phi1 = -dk / k2                      # lap phi1 = delta
    phi1[0, 0, 0] = 0
    psi1 = np.empty((3, n, n, n), np.float32)
    for i in range(3):
        psi1[i] = irfftn(-1j * ks[i] * phi1, n)      # psi1 = -grad phi1
    # second-order source S = sum_{i>j} (phi_ii phi_jj - phi_ij^2)
    d = {}
    for i in range(3):
        d[i, i] = irfftn(-ks[i] * ks[i] * phi1, n).astype(np.float32)
    S = d[0, 0] * d[1, 1] + d[0, 0] * d[2, 2] + d[1, 1] * d[2, 2]
    del d
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        dij = irfftn(-ks[i] * ks[j] * phi1, n).astype(np.float32)
        S -= dij * dij
    Sk = rfftn(S)
    del S
    phi2 = -Sk / k2
    phi2[0, 0, 0] = 0
    psi2 = np.empty((3, n, n, n), np.float32)
    for i in range(3):
        psi2[i] = irfftn(1j * ks[i] * phi2, n)       # psi2 = +grad phi2
    return psi1, psi2


def lattice_1d(n, L):
    return np.arange(n, dtype=np.float64) * (L / n)


def lpt_state(cosmo, psi1, psi2, a, L, order=2, dtype=np.float64, momenta=True):
    """Positions (3,n,n,n, wrapped into [0,L)) and momenta p = a^2 dx/dt.
    Built one component at a time to keep the peak memory low."""
    n = psi1.shape[1]
    D1, D2 = cosmo.D1(a), cosmo.D2(a)
    f1, f2 = cosmo.f1(a), cosmo.f2(a)
    q = lattice_1d(n, L)
    shp = [(n, 1, 1), (1, n, 1), (1, 1, n)]
    x = np.empty((3, n, n, n), dtype)
    p = np.empty((3, n, n, n), dtype) if momenta else None
    for i in range(3):
        xi = q.reshape(shp[i]) + D1 * psi1[i].astype(np.float64)
        if order >= 2:
            xi += D2 * psi2[i]
        x[i] = np.mod(xi, L)
        if momenta:
            v = f1 * D1 * psi1[i].astype(np.float64)
            if order >= 2:
                v += f2 * D2 * psi2[i]
            p[i] = (a**2 * cosmo.E(a)) * v
    return x, p


def lattice(n, L):
    q = lattice_1d(n, L)
    return np.stack(np.meshgrid(q, q, q, indexing="ij"))


# ---------------------------------------------------------------------------
# CIC deposit / interpolation (numba)
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def _cic_deposit(x, y, z, w, ng, L, out):
    s = ng / L
    for p in range(x.size):
        fx = x[p] * s; fy = y[p] * s; fz = z[p] * s
        ix = int(np.floor(fx)); iy = int(np.floor(fy)); iz = int(np.floor(fz))
        dx = fx - ix; dy = fy - iy; dz = fz - iz
        tx = 1 - dx; ty = 1 - dy; tz = 1 - dz
        ix %= ng; iy %= ng; iz %= ng
        jx = (ix + 1) % ng; jy = (iy + 1) % ng; jz = (iz + 1) % ng
        ww = w[p] if w.size > 1 else w[0]
        out[ix, iy, iz] += ww * tx * ty * tz
        out[jx, iy, iz] += ww * dx * ty * tz
        out[ix, jy, iz] += ww * tx * dy * tz
        out[ix, iy, jz] += ww * tx * ty * dz
        out[jx, jy, iz] += ww * dx * dy * tz
        out[jx, iy, jz] += ww * dx * ty * dz
        out[ix, jy, jz] += ww * tx * dy * dz
        out[jx, jy, jz] += ww * dx * dy * dz


@nb.njit(cache=True, parallel=True)
def _cic_deposit_par(x, y, z, ng, L, nchunk):
    """Parallel CIC via per-slab ownership: particles pre-binned by x-slab."""
    out = np.zeros((ng, ng, ng), np.float32)
    s = ng / L
    npart = x.size
    # two passes over even / odd chunks of x-slabs to avoid write races
    slab = np.empty(npart, np.int64)
    for p in nb.prange(npart):
        slab[p] = (int(np.floor(x[p] * s)) % ng) * nchunk // ng
    order = np.argsort(slab)
    starts = np.searchsorted(slab[order], np.arange(nchunk + 1))
    for parity in range(2):
        for c in nb.prange(nchunk // 2):
            ch = 2 * c + parity
            for t in range(starts[ch], starts[ch + 1]):
                p = order[t]
                fx = x[p] * s; fy = y[p] * s; fz = z[p] * s
                ix = int(np.floor(fx)); iy = int(np.floor(fy)); iz = int(np.floor(fz))
                dx = fx - ix; dy = fy - iy; dz = fz - iz
                tx = 1 - dx; ty = 1 - dy; tz = 1 - dz
                ix %= ng; iy %= ng; iz %= ng
                jx = (ix + 1) % ng; jy = (iy + 1) % ng; jz = (iz + 1) % ng
                out[ix, iy, iz] += tx * ty * tz
                out[jx, iy, iz] += dx * ty * tz
                out[ix, jy, iz] += tx * dy * tz
                out[ix, iy, jz] += tx * ty * dz
                out[jx, jy, iz] += dx * dy * tz
                out[jx, iy, jz] += dx * ty * dz
                out[ix, jy, jz] += tx * dy * dz
                out[jx, jy, jz] += dx * dy * dz
    return out


def cic_density(x, ng, L):
    """Overdensity 1+delta on an ng^3 mesh from unit-mass particles x (3,...)."""
    xs = [np.ascontiguousarray(c.ravel()) for c in x]
    nchunk = max(2, min(ng, 2 * NTHREADS * 4) // 2 * 2)
    rho = _cic_deposit_par(xs[0], xs[1], xs[2], ng, L, nchunk)
    rho *= ng**3 / xs[0].size
    return rho


@nb.njit(cache=True, parallel=True)
def _cic_interp(f, x, y, z, L, out):
    ng = f.shape[0]
    s = ng / L
    for p in nb.prange(x.size):
        fx = x[p] * s; fy = y[p] * s; fz = z[p] * s
        ix = int(np.floor(fx)); iy = int(np.floor(fy)); iz = int(np.floor(fz))
        dx = fx - ix; dy = fy - iy; dz = fz - iz
        tx = 1 - dx; ty = 1 - dy; tz = 1 - dz
        ix %= ng; iy %= ng; iz %= ng
        jx = (ix + 1) % ng; jy = (iy + 1) % ng; jz = (iz + 1) % ng
        out[p] = (f[ix, iy, iz] * tx * ty * tz + f[jx, iy, iz] * dx * ty * tz +
                  f[ix, jy, iz] * tx * dy * tz + f[ix, iy, jz] * tx * ty * dz +
                  f[jx, jy, iz] * dx * dy * tz + f[jx, iy, jz] * dx * ty * dz +
                  f[ix, jy, jz] * tx * dy * dz + f[jx, jy, jz] * dx * dy * dz)


# ---------------------------------------------------------------------------
# Shell-crossing diagnostic: signed volumes of the 6 Freudenthal tetrahedra
# of every Lagrangian lattice cube (periodic).  A tet with non-positive volume
# belongs to a shell-crossed (multi-stream) region.
# ---------------------------------------------------------------------------
_TETS = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7],
                  [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]], np.int64)
_CORNER = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], np.int64)


@nb.njit(cache=True, parallel=True)
def _tet_flags(x, L, tets, corner, flag):
    """flag[i,j,k,t] |= (signed volume of tet t in cube (i,j,k) <= 0).
    Returns nothing; flag is updated in place (uint8)."""
    n = x.shape[1]
    half = 0.5 * L
    for i in nb.prange(n):
        P = np.empty((8, 3))
        for j in range(n):
            for k in range(n):
                x0 = x[0, i, j, k]; y0 = x[1, i, j, k]; z0 = x[2, i, j, k]
                for c in range(8):
                    ii = (i + corner[c, 0]) % n
                    jj = (j + corner[c, 1]) % n
                    kk = (k + corner[c, 2]) % n
                    ddx = x[0, ii, jj, kk] - x0
                    ddy = x[1, ii, jj, kk] - y0
                    ddz = x[2, ii, jj, kk] - z0
                    # minimum image (displacements between neighbours << L/2)
                    if ddx > half: ddx -= L
                    elif ddx < -half: ddx += L
                    if ddy > half: ddy -= L
                    elif ddy < -half: ddy += L
                    if ddz > half: ddz -= L
                    elif ddz < -half: ddz += L
                    P[c, 0] = ddx; P[c, 1] = ddy; P[c, 2] = ddz
                for t in range(6):
                    a = tets[t, 0]; b = tets[t, 1]; c2 = tets[t, 2]; d = tets[t, 3]
                    ax = P[b, 0] - P[a, 0]; ay = P[b, 1] - P[a, 1]; az = P[b, 2] - P[a, 2]
                    bx = P[c2, 0] - P[a, 0]; by = P[c2, 1] - P[a, 1]; bz = P[c2, 2] - P[a, 2]
                    cx = P[d, 0] - P[a, 0]; cy = P[d, 1] - P[a, 1]; cz = P[d, 2] - P[a, 2]
                    vol = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
                    # orientation of the reference tet (lattice) fixes the sign
                    if t == 0 or t == 3 or t == 4:
                        vol = -vol
                    if vol <= 0:
                        flag[i, j, k, t] = 1


def tet_orientation_signs():
    """Sign of the lattice (unperturbed) volume of each tet, for validation."""
    s = []
    for t in _TETS:
        P = _CORNER[t].astype(float)
        m = np.array([P[1] - P[0], P[2] - P[0], P[3] - P[0]])
        s.append(np.sign(np.linalg.det(m)))
    return np.array(s)


def update_crossing_flags(x, L, flag):
    _tet_flags(x, L, _TETS, _CORNER, flag)
    return flag.mean()


# ---------------------------------------------------------------------------
# PM N-body: KDK leapfrog in a, spectral Poisson + gradient, CIC both ways.
# ---------------------------------------------------------------------------
class PM:
    """Particle-mesh gravity.

    gradient: 'fd4'  -> 4-point finite-difference kernel i(8 sin kh - sin 2kh)/(6h)
              'ik'   -> exact spectral gradient
    deconv:   divide the potential by W_CIC^2 (deposit + interpolation windows).
              Off by default: it amplifies lattice modes and makes a 2x mesh
              unstable (linear-growth test blows up).
    The fd4 kernel vanishes at the mesh Nyquist frequency, which is where the
    density pattern of the particle lattice sits when nmesh = 2 n_part; with
    the naive ik gradient that pattern produces large spurious lattice forces.
    """
    def __init__(self, cosmo, L, nmesh, gradient="fd4", deconv=False):
        self.c, self.L, self.ng = cosmo, L, nmesh
        kx, ky, kz = kgrid(nmesh, L, np.float64)
        h = L / nmesh
        if gradient == "fd4":
            D = lambda k: (8 * np.sin(k * h) - np.sin(2 * k * h)) / (6 * h)
            self.ks = tuple(D(k).astype(np.float32) for k in (kx, ky, kz))
        else:
            self.ks = tuple(k.astype(np.float32) for k in (kx, ky, kz))
        k2 = kx**2 + ky**2 + kz**2
        k2[0, 0, 0] = 1
        g = 1.0 / k2
        if deconv:
            g = g / cic_window(nmesh, L)**2
        self.ik2 = g.astype(np.float32)
        self.ik2[0, 0, 0] = 0

    def accel(self, x, a):
        """-grad Phi at particles, with lap Phi = 1.5 Om delta / a."""
        ng = self.ng
        rho = cic_density(x, ng, self.L)
        rho -= 1.0
        dk = rfftn(rho)
        del rho
        phik = (-1.5 * self.c.Om / a) * dk * self.ik2
        del dk
        acc = np.empty_like(x)
        tmp = np.empty(x[0].size)
        xs = [np.ascontiguousarray(c.ravel()) for c in x]
        for i in range(3):
            g = irfftn(-1j * self.ks[i] * phik, ng).astype(np.float32)   # -dPhi/dx_i
            _cic_interp(g, xs[0], xs[1], xs[2], self.L, tmp)
            acc[i] = tmp.reshape(x[0].shape)
        return acc

    def run(self, x, p, a_i, a_f, nsteps, track_crossing=False, callback=None,
            spacing="a"):
        c, L = self.c, self.L
        if spacing == "a":
            a_s = np.linspace(a_i, a_f, nsteps + 1)
        else:
            a_s = np.geomspace(a_i, a_f, nsteps + 1)
        flag = None
        if track_crossing:
            n = x.shape[1]
            flag = np.zeros((n, n, n, 6), np.uint8)
        acc = self.accel(x, a_s[0])
        for s in range(nsteps):
            a0, a1 = a_s[s], a_s[s + 1]
            am = 0.5 * (a0 + a1)
            p += c.kick_factor(a0, am) * acc
            x += c.drift_factor(a0, a1) * p
            np.mod(x, L, out=x)
            acc = self.accel(x, a1)
            p += c.kick_factor(am, a1) * acc
            if track_crossing:
                update_crossing_flags(x, L, flag)
            if callback is not None:
                callback(s, a1, x, p, flag)
        return x, p, flag


# ---------------------------------------------------------------------------
# Density field smoothing and PDFs
# ---------------------------------------------------------------------------
def cic_window(ng, L):
    kx, ky, kz = kgrid(ng, L, np.float64)
    h = L / ng / 2
    w = (np.sinc(kx * h / np.pi) * np.sinc(ky * h / np.pi) * np.sinc(kz * h / np.pi))**2
    return w


def smoothed_fields(rho, L, radii):
    """Top-hat smoothed 1+delta for each radius (CIC deposit deconvolved)."""
    ng = rho.shape[0]
    rk = rfftn(rho)
    wc = cic_window(ng, L)
    kx, ky, kz = kgrid(ng, L, np.float64)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    out = {}
    for R in radii:
        f = rk * (W_TH(k * R) / wc)
        out[R] = irfftn(f, ng).astype(np.float32)
    return out


LOGBINS = np.linspace(np.log10(0.05), np.log10(20.0), 161)


def pdfs(rho_raw, smoothed, bins=LOGBINS):
    """Volume- and mass-weighted PDFs of log10(1+delta_R).

    Volume weighting: every mesh cell counts equally.
    Mass weighting: each cell weighted by the particle mass CIC-deposited into
    it, i.e. the distribution of rho_R seen by a random mass element.  This is
    independent of particle number, so different resolutions compare cleanly."""
    res = {}
    w = rho_raw.ravel().astype(np.float64)
    for R, f in smoothed.items():
        lf = np.log10(np.clip(f.ravel(), 1e-6, None))
        hv, _ = np.histogram(lf, bins=bins, density=True)
        hm, _ = np.histogram(lf, bins=bins, weights=w, density=True)
        fv = f.ravel().astype(np.float64)
        d = fv - 1.0
        mom = dict(
            var=float(np.mean(d**2)),
            skew=float(np.mean(d**3)),
            kurt=float(np.mean(d**4) - 3 * np.mean(d**2)**2),
            fmin=float(fv.min()), fmax=float(fv.max()),
            neg_frac=float(np.mean(fv <= 0)),
        )
        mom["S3"] = mom["skew"] / mom["var"]**2
        mom["S4"] = mom["kurt"] / mom["var"]**3
        # mass-weighted moments of log density
        lm = np.log(np.clip(fv, 1e-6, None))
        mom["mean_log_V"] = float(np.mean(lm))
        mom["mean_log_M"] = float(np.sum(w * lm) / np.sum(w))
        res[R] = dict(vol=hv, mass=hm, **mom)
    return res
