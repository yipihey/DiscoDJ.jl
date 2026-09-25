"""Phase-space sheet (Abel, Hahn & Kaehler 2012) on the Lagrangian lattice.

Elements are the Lagrangian cubes of the (possibly refined) particle lattice,
each split into the 6 Freudenthal tetrahedra used in core.py.  For element e
with Lagrangian volume Vq = (L/n)^3:

  V_e   = sum of the 6 exact signed tetrahedron volumes      (Eulerian volume)
  flip  = any tetrahedron with signed volume <= 0
  rho_e = Vq / V_e                                           (stream density, rho_bar = 1)

Sheet density on an Eulerian mesh (all streams, exact tetrahedron volumes):
  rho(x_j) = sum over tetrahedra containing x_j of (Vq/6) / |V_t|
evaluated at the mesh nodes x_j = j h (point sampling, no deposit window).

Single-stream test (see `multistream_elements`): every point with stream count
>= 3 is covered by at least one flipped tetrahedron (orientations sum to +1),
so the Eulerian footprint of the flipped tetrahedra (their bounding boxes,
marked on the mesh) contains every multistream region.  An element is
single-stream if it has no flipped tetrahedron and none of its 8 vertices lies
in a marked cell.  This is conservative: it can drop single-stream elements
within about one bounding box / mesh cell of a caustic, never keep a
multistream one whose vertex lies in the footprint.
"""
import numpy as np
import numba as nb

from core import rfftn, irfftn, _TETS, _CORNER

TET_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0])   # lattice orientation of each tet


# ---------------------------------------------------------------------------
# refinement: band-limited (Fourier) interpolation of psi(q) onto a 2^l finer lattice
# ---------------------------------------------------------------------------
def _pad64(ff, n, e):
    h = n // 2
    out = np.zeros((e, e, e // 2 + 1), np.complex128)
    lo, hi_s, hi_d = slice(0, h), slice(h + 1, n), slice(e - h + 1, e)
    for sx_s, sx_d in ((lo, lo), (hi_s, hi_d)):
        for sy_s, sy_d in ((lo, lo), (hi_s, hi_d)):
            out[sx_d, sy_d, :h] = ff[sx_s, sy_s, :h]
    return out * (e / n)**3


def refine(psi, level):
    """psi (3,n,n,n) -> (3,n 2^l, ...), exact for band-limited fields (LPT);
    Nyquist planes of the coarse grid are dropped."""
    if level == 0:
        return psi
    n = psi.shape[1]
    e = n * 2**level
    out = np.empty((3, e, e, e))
    for i in range(3):
        out[i] = irfftn(_pad64(rfftn(psi[i]), n, e), e)
    return out


# ---------------------------------------------------------------------------
# element geometry
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True)
def _elements(psi, L, tets, corner, tsign, V, flip):
    n = psi.shape[1]
    dq = L / n
    for i in nb.prange(n):
        P = np.empty((8, 3))
        for j in range(n):
            for k in range(n):
                for c in range(8):
                    ii = (i + corner[c, 0]) % n
                    jj = (j + corner[c, 1]) % n
                    kk = (k + corner[c, 2]) % n
                    P[c, 0] = corner[c, 0] * dq + psi[0, ii, jj, kk] - psi[0, i, j, k]
                    P[c, 1] = corner[c, 1] * dq + psi[1, ii, jj, kk] - psi[1, i, j, k]
                    P[c, 2] = corner[c, 2] * dq + psi[2, ii, jj, kk] - psi[2, i, j, k]
                vs = 0.0
                f = False
                for t in range(6):
                    a = tets[t, 0]; b = tets[t, 1]; c2 = tets[t, 2]; d = tets[t, 3]
                    ax = P[b, 0] - P[a, 0]; ay = P[b, 1] - P[a, 1]; az = P[b, 2] - P[a, 2]
                    bx = P[c2, 0] - P[a, 0]; by = P[c2, 1] - P[a, 1]; bz = P[c2, 2] - P[a, 2]
                    cx = P[d, 0] - P[a, 0]; cy = P[d, 1] - P[a, 1]; cz = P[d, 2] - P[a, 2]
                    vol = tsign[t] * (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx)
                                      + az * (bx * cy - by * cx)) / 6.0
                    vs += vol
                    if vol <= 0.0:
                        f = True
                V[i, j, k] = vs
                flip[i, j, k] = f


def elements(psi, L):
    n = psi.shape[1]
    V = np.empty((n, n, n))
    flip = np.empty((n, n, n), np.bool_)
    _elements(psi, L, _TETS, _CORNER, TET_SIGN, V, flip)
    return V, flip


@nb.njit(cache=True)
def _mark_footprint(psi, L, tets, corner, tsign, ng, mark):
    """Mark mesh cells overlapped by the bounding box of every flipped tet (serial:
    flipped tets are rare)."""
    n = psi.shape[1]
    dq = L / n
    h = L / ng
    P = np.empty((8, 3))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for c in range(8):
                    ii = (i + corner[c, 0]) % n
                    jj = (j + corner[c, 1]) % n
                    kk = (k + corner[c, 2]) % n
                    P[c, 0] = (i + corner[c, 0]) * dq + psi[0, ii, jj, kk]
                    P[c, 1] = (j + corner[c, 1]) * dq + psi[1, ii, jj, kk]
                    P[c, 2] = (k + corner[c, 2]) * dq + psi[2, ii, jj, kk]
                for t in range(6):
                    a = tets[t, 0]; b = tets[t, 1]; c2 = tets[t, 2]; d = tets[t, 3]
                    ax = P[b, 0] - P[a, 0]; ay = P[b, 1] - P[a, 1]; az = P[b, 2] - P[a, 2]
                    bx = P[c2, 0] - P[a, 0]; by = P[c2, 1] - P[a, 1]; bz = P[c2, 2] - P[a, 2]
                    cx = P[d, 0] - P[a, 0]; cy = P[d, 1] - P[a, 1]; cz = P[d, 2] - P[a, 2]
                    vol = tsign[t] * (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx)
                                      + az * (bx * cy - by * cx))
                    if vol > 0.0:
                        continue
                    lo = np.empty(3, np.int64); hi = np.empty(3, np.int64)
                    for ax_ in range(3):
                        mn = min(P[a, ax_], P[b, ax_], P[c2, ax_], P[d, ax_])
                        mx = max(P[a, ax_], P[b, ax_], P[c2, ax_], P[d, ax_])
                        lo[ax_] = int(np.floor(mn / h)); hi[ax_] = int(np.floor(mx / h))
                    for x in range(lo[0], hi[0] + 1):
                        for y in range(lo[1], hi[1] + 1):
                            for z in range(lo[2], hi[2] + 1):
                                mark[x % ng, y % ng, z % ng] = 1


@nb.njit(cache=True, parallel=True)
def _vertex_in_mark(psi, L, ng, mark, out):
    """out[i,j,k] = True if any of the 8 vertices of element (i,j,k) lies in a marked cell."""
    n = psi.shape[1]
    dq = L / n
    h = L / ng
    for i in nb.prange(n):
        for j in range(n):
            for k in range(n):
                hit = False
                for c in range(8):
                    di = c >> 2; dj = (c >> 1) & 1; dk = c & 1
                    ii = (i + di) % n; jj = (j + dj) % n; kk = (k + dk) % n
                    x = (i + di) * dq + psi[0, ii, jj, kk]
                    y = (j + dj) * dq + psi[1, ii, jj, kk]
                    z = (k + dk) * dq + psi[2, ii, jj, kk]
                    cx = int(np.floor(x / h)) % ng
                    cy = int(np.floor(y / h)) % ng
                    cz = int(np.floor(z / h)) % ng
                    if mark[cx, cy, cz]:
                        hit = True
                        break
                out[i, j, k] = hit


def multistream_elements(psi, L, flip, ng):
    """Boolean (n,n,n): element is flipped or touches the flipped-tet footprint."""
    n = psi.shape[1]
    ms = flip.copy()
    if not flip.any():
        return ms
    mark = np.zeros((ng, ng, ng), np.uint8)
    _mark_footprint(psi, L, _TETS, _CORNER, TET_SIGN, ng, mark)
    near = np.empty((n, n, n), np.bool_)
    _vertex_in_mark(psi, L, ng, mark, near)
    return ms | near


# ---------------------------------------------------------------------------
# sheet density on an Eulerian mesh (point sampling at the mesh nodes)
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True)
def _sheet_density(psi, L, tets, corner, ng, nchunk, out):
    n = psi.shape[1]
    dq = L / n
    h = L / ng
    mt = dq * dq * dq / 6.0
    for ch in nb.prange(nchunk):
        acc = out[ch]
        P = np.empty((8, 3))
        for i in range(ch * n // nchunk, (ch + 1) * n // nchunk):
            for j in range(n):
                for k in range(n):
                    for c in range(8):
                        ii = (i + corner[c, 0]) % n
                        jj = (j + corner[c, 1]) % n
                        kk = (k + corner[c, 2]) % n
                        P[c, 0] = (i + corner[c, 0]) * dq + psi[0, ii, jj, kk]
                        P[c, 1] = (j + corner[c, 1]) * dq + psi[1, ii, jj, kk]
                        P[c, 2] = (k + corner[c, 2]) * dq + psi[2, ii, jj, kk]
                    for t in range(6):
                        a = tets[t, 0]; b = tets[t, 1]; c2 = tets[t, 2]; d = tets[t, 3]
                        e1x = P[b, 0] - P[a, 0]; e1y = P[b, 1] - P[a, 1]; e1z = P[b, 2] - P[a, 2]
                        e2x = P[c2, 0] - P[a, 0]; e2y = P[c2, 1] - P[a, 1]; e2z = P[c2, 2] - P[a, 2]
                        e3x = P[d, 0] - P[a, 0]; e3y = P[d, 1] - P[a, 1]; e3z = P[d, 2] - P[a, 2]
                        # inverse of T = [e1 e2 e3] via cofactors
                        c00 = e2y * e3z - e2z * e3y; c01 = e2z * e3x - e2x * e3z; c02 = e2x * e3y - e2y * e3x
                        c10 = e3y * e1z - e3z * e1y; c11 = e3z * e1x - e3x * e1z; c12 = e3x * e1y - e3y * e1x
                        c20 = e1y * e2z - e1z * e2y; c21 = e1z * e2x - e1x * e2z; c22 = e1x * e2y - e1y * e2x
                        det = e1x * c00 + e1y * c01 + e1z * c02
                        if det == 0.0:
                            continue
                        rho = mt / abs(det / 6.0)
                        mnx = min(P[a, 0], P[b, 0], P[c2, 0], P[d, 0]); mxx = max(P[a, 0], P[b, 0], P[c2, 0], P[d, 0])
                        mny = min(P[a, 1], P[b, 1], P[c2, 1], P[d, 1]); mxy = max(P[a, 1], P[b, 1], P[c2, 1], P[d, 1])
                        mnz = min(P[a, 2], P[b, 2], P[c2, 2], P[d, 2]); mxz = max(P[a, 2], P[b, 2], P[c2, 2], P[d, 2])
                        for x in range(int(np.ceil(mnx / h)), int(np.floor(mxx / h)) + 1):
                            px = x * h - P[a, 0]
                            for y in range(int(np.ceil(mny / h)), int(np.floor(mxy / h)) + 1):
                                py = y * h - P[a, 1]
                                for z in range(int(np.ceil(mnz / h)), int(np.floor(mxz / h)) + 1):
                                    pz = z * h - P[a, 2]
                                    l1 = (c00 * px + c01 * py + c02 * pz) / det
                                    if l1 < 0.0: continue
                                    l2 = (c10 * px + c11 * py + c12 * pz) / det
                                    if l2 < 0.0: continue
                                    l3 = (c20 * px + c21 * py + c22 * pz) / det
                                    if l3 < 0.0 or l1 + l2 + l3 > 1.0: continue
                                    acc[x % ng, y % ng, z % ng] += rho


def sheet_density(psi, L, ng, nchunk=None):
    """rho/rho_bar at the ng^3 mesh nodes, summed over all streams."""
    import os
    T = nb.get_num_threads()
    nchunk = nchunk or T
    out = np.zeros((nchunk, ng, ng, ng), np.float32)
    _sheet_density(psi, L, _TETS, _CORNER, ng, nchunk, out)
    return out.sum(axis=0, dtype=np.float64)


# ---------------------------------------------------------------------------
# sample a mesh field at the element centroids (trilinear, node-centred mesh)
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True)
def _sample_centroids(psi, L, f, out):
    n = psi.shape[1]
    ng = f.shape[0]
    dq = L / n
    s = ng / L
    for i in nb.prange(n):
        for j in range(n):
            for k in range(n):
                sx = 0.0; sy = 0.0; sz = 0.0
                for c in range(8):
                    di = c >> 2; dj = (c >> 1) & 1; dk = c & 1
                    ii = (i + di) % n; jj = (j + dj) % n; kk = (k + dk) % n
                    sx += (i + di) * dq + psi[0, ii, jj, kk]
                    sy += (j + dj) * dq + psi[1, ii, jj, kk]
                    sz += (k + dk) * dq + psi[2, ii, jj, kk]
                fx = sx / 8 * s; fy = sy / 8 * s; fz = sz / 8 * s
                ix = int(np.floor(fx)); iy = int(np.floor(fy)); iz = int(np.floor(fz))
                dx = fx - ix; dy = fy - iy; dz = fz - iz
                ix %= ng; iy %= ng; iz %= ng
                jx = (ix + 1) % ng; jy = (iy + 1) % ng; jz = (iz + 1) % ng
                out[i, j, k] = (f[ix, iy, iz] * (1 - dx) * (1 - dy) * (1 - dz)
                                + f[jx, iy, iz] * dx * (1 - dy) * (1 - dz)
                                + f[ix, jy, iz] * (1 - dx) * dy * (1 - dz)
                                + f[ix, iy, jz] * (1 - dx) * (1 - dy) * dz
                                + f[jx, jy, iz] * dx * dy * (1 - dz)
                                + f[jx, iy, jz] * dx * (1 - dy) * dz
                                + f[ix, jy, jz] * (1 - dx) * dy * dz
                                + f[jx, jy, jz] * dx * dy * dz)


def sample_centroids(psi, L, f):
    n = psi.shape[1]
    out = np.empty((n, n, n))
    _sample_centroids(psi, L, f, out)
    return out
