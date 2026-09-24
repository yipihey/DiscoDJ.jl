"""General-order nLPT (orders 1-4) ported from DiscoDJNative/src/lpt/nlpt_core.jl
(itself a faithful port of DISCO-DJ's `compute_core` / `compute_core_exact`,
arXiv:2010.12584).

* `shapes_exact`: growth-free shape fields for exact LCDM growth up to 3rd order,
      x = q + D1 psi1 + D2 psi2_ex + D3a psi3a + D3b psi3b + D3c psi3c
* `shapes_eds`: the general-order EdS recursion (longitudinal mu2/mu3 sources
      and the transverse C term), x = q + sum_n D1^n psi_n
* 4LPT used in the study = exact growth through 3rd order + D1^4 psi4 (EdS
  recursion), which is how DISCO-DJ extends beyond its exact-growth orders.

All quadratic/cubic sources are formed on the 3/2-rule extended grid
(de-aliasing) exactly as in the Julia/JAX code: every source is
crop(rfft(sum of products of real extended-grid derivative fields)).
Gradient kernels are Nyquist-zeroed; inverse Laplacian is -1/k^2.
"""
import numpy as np

from core import rfftn, irfftn


# ---------------------------------------------------------------------------
# kernels and pad / crop
# ---------------------------------------------------------------------------
def _k1d(n, L):
    dk = 2 * np.pi / L
    kf = np.fft.fftfreq(n, 1.0 / n) * dk      # index n/2 is the (negative) Nyquist
    kh = np.arange(n // 2 + 1) * dk
    return kf, kh


def grad_kernels(n, L):
    """i*k per axis (broadcastable, Nyquist zeroed) on an n^3 rfft grid."""
    kf, kh = _k1d(n, L)
    kx = 1j * kf.copy(); kx[n // 2] = 0
    kz = 1j * kh.copy(); kz[-1] = 0
    return (kx[:, None, None].astype(np.complex64), kx[None, :, None].astype(np.complex64),
            kz[None, None, :].astype(np.complex64))


def inv_lap(n, L):
    kf, kh = _k1d(n, L)
    k2 = kf[:, None, None]**2 + kf[None, :, None]**2 + kh[None, None, :]**2
    k2[0, 0, 0] = 1
    return (-1.0 / k2).astype(np.float32)


def pad(ff, n, e):
    """(n,n,n/2+1) -> (e,e,e/2+1), high modes zero, Nyquist planes dropped,
    amplitude rescaled by (e/n)^3 (DFT normalisation)."""
    h = n // 2
    out = np.zeros((e, e, e // 2 + 1), np.complex64)
    lo, hi_s, hi_d = slice(0, h), slice(h + 1, n), slice(e - h + 1, e)
    for sx_s, sx_d in ((lo, lo), (hi_s, hi_d)):
        for sy_s, sy_d in ((lo, lo), (hi_s, hi_d)):
            out[sx_d, sy_d, :h] = ff[sx_s, sy_s, :h]
    out *= (e / n)**3
    return out


def crop(ff, n, e):
    """Inverse block map of `pad` (Nyquist planes zero), rescaled by (n/e)^3."""
    h = n // 2
    out = np.zeros((n, n, n // 2 + 1), np.complex64)
    lo, hi_s, hi_d = slice(0, h), slice(e - h + 1, e), slice(h + 1, n)
    for sx_s, sx_d in ((lo, lo), (hi_s, hi_d)):
        for sy_s, sy_d in ((lo, lo), (hi_s, hi_d)):
            out[sx_d, sy_d, :h] = ff[sx_s, sy_s, :h]
    out *= (n / e)**3
    return out


class Engine:
    def __init__(self, n, L, dealias=True):
        self.n, self.L = n, L
        self.e = 3 * n // 2 if dealias else n
        self.d = grad_kernels(n, L)
        self.de = grad_kernels(self.e, L)
        self.il = inv_lap(n, L)

    # real extended-grid derivative tensor G[i][k] = d_k psi_i
    def G(self, psik):
        n, e = self.n, self.e
        out = [[None] * 3 for _ in range(3)]
        for i in range(3):
            p = pad(psik[i], n, e) if e != n else psik[i]
            for k in range(3):
                out[i][k] = irfftn(self.de[k] * p, e).astype(np.float32)
        return out

    def to_k(self, real_ext):
        f = rfftn(real_ext).astype(np.complex64)
        return crop(f, self.n, self.e) if self.e != self.n else _zero_nyq(f, self.n)

    # ---- sources (1-based index tables of the Julia code, made 0-based) ----
    def mu2_sym(self, A):
        r = (A[0][0] * A[1][1] + A[0][0] * A[2][2] + A[1][1] * A[2][2]
             - A[0][1] * A[1][0] - A[0][2] * A[2][0] - A[1][2] * A[2][1])
        return self.to_k(r)

    _I = (1, 1, 2, 3, 2, 2, 3, 1, 3, 3, 1, 2); _J = (2, 3, 1, 1, 3, 1, 2, 2, 1, 2, 3, 3)
    _K = (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3); _L = (2, 3, 2, 3, 3, 1, 3, 1, 1, 2, 1, 2)
    _S = (1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1)
    _CI = (1, 1, 2, 2, 3, 3); _CK = (2, 3, 2, 3, 2, 3); _CL = (3, 2, 3, 2, 3, 2)
    _CS = (1, -1, 1, -1, 1, -1)

    def mu2_and_C(self, A, B):
        r = 0
        for i, j, k, l, s in zip(self._I, self._J, self._K, self._L, self._S):
            r = r + s * A[i - 1][k - 1] * B[j - 1][l - 1]
        mu2 = self.to_k(r)
        cyc = lambda x, sh: (x - 1 + sh) % 3        # 0-based result
        C = []
        for sh in range(3):
            r = 0
            for i, k, l, s in zip(self._CI, self._CK, self._CL, self._CS):
                ii, kk, ll = cyc(i, sh), cyc(k, sh), cyc(l, sh)
                r = r + s * A[ii][kk] * B[ii][ll]
            C.append(self.to_k(r))
        return mu2, C

    _MK = (1, 1, 2, 2, 3, 3); _ML = (2, 3, 3, 1, 1, 2); _MM = (3, 2, 1, 3, 2, 1)
    _MS = (1, -1, 1, -1, 1, -1)

    def mu3(self, A, B, Cc):
        r = 0
        for k, l, m, s in zip(self._MK, self._ML, self._MM, self._MS):
            r = r + s * A[0][k - 1] * B[1][l - 1] * Cc[2][m - 1]
        return self.to_k(r)

    def assemble(self, fL, C=None):
        dx, dy, dz = self.d
        if C is None:
            return [self.il * (d * fL) for d in (dx, dy, dz)]
        Cx, Cy, Cz = C
        L_ = 0 if fL is None else fL
        return [self.il * (dx * L_ - (dy * Cz - dz * Cy)),
                self.il * (dy * L_ - (dz * Cx - dx * Cz)),
                self.il * (dz * L_ - (dx * Cy - dy * Cx))]


def _zero_nyq(f, n):
    h = n // 2
    f[h] = 0; f[:, h] = 0; f[:, :, -1] = 0
    return f


def to_real(psik, n):
    return np.stack([irfftn(c, n).astype(np.float32) for c in psik])


def _lin(*terms):
    out = [0, 0, 0]
    for c, v in terms:
        for i in range(3):
            out[i] = out[i] + c * v[i]
    return out


def compute_shapes(dk, n, L, dealias=True, n_order=4):
    """All shape fields needed for orders 1..n_order, exact and EdS.

    phi1 with lap phi1 = delta (a=1 linear density), psi1 = -grad phi1.
    Returns dict of real (3,n,n,n) float32 arrays:
      psi1, psi2_ex, psi3a, psi3b, psi3c   (exact-growth shapes)
      psi2_eds, psi3_eds, psi4_eds         (EdS recursion, multiply D1^n)
    """
    E = Engine(n, L, dealias)
    k2 = -1.0 / E.il
    phi = -dk / k2
    phi[0, 0, 0] = 0
    psi = {1: [-(d * phi) for d in E.d]}                 # Fourier psi_n (EdS; psi1 exact too)
    out = {"psi1": to_real(psi[1], n)}
    G = {1: E.G(psi[1])}

    # ---- exact growth, orders 2-3 (compute_core_exact) ----
    if n_order >= 2:
        mu2_11 = E.mu2_sym(G[1])
        psi2ex = E.assemble(mu2_11)
        out["psi2_ex"] = to_real(psi2ex, n)
    if n_order >= 3:
        G2ex = E.G(psi2ex)
        mu3_111 = E.mu3(G[1], G[1], G[1])
        m2, C12 = E.mu2_and_C(G[1], G2ex)
        del G2ex
        out["psi3a"] = to_real(E.assemble(-mu3_111), n)
        out["psi3b"] = to_real(E.assemble(-0.5 * m2), n)
        out["psi3c"] = to_real(E.assemble(None, [-c for c in C12]), n)

    # ---- EdS recursion (compute_core), general order ----
    for i in range(2, n_order + 1):
        fL = 0
        fT = [0, 0, 0]
        den = (i + 1.5) * (i - 1)
        if i % 2 == 0:
            h = i // 2
            fac = ((3 - i) / 2 - 2 * h * h) / den
            src = mu2_11 if h == 1 else E.mu2_sym(G[h])
            fL = fL + fac * src
        if i > 2:
            for j in range(1, (i + 1) // 2):
                imj = i - j
                fac = ((3 - i) / 2 - j * j - imj * imj) / den
                if (j, imj) == (1, 2):             # reuse mu2_and_C(G1, G2ex), G2 = -3/7 G2ex
                    m2_, C_ = -3 / 7 * m2, [-3 / 7 * c for c in C12]
                else:
                    m2_, C_ = E.mu2_and_C(G[j], G[imj])
                fL = fL + fac * m2_
                facC = 1 - 2 * j / i
                fT = [fT[a] + facC * C_[a] for a in range(3)]
            for k in range(1, i - 1):
                for l in range(1, i - k):
                    m = i - k - l
                    fac = ((3 - i) / 2 - k * k - l * l - m * m) / den
                    if (k, l, m) == (1, 1, 1):
                        src = mu3_111
                    else:
                        src = E.mu3(G[k], G[l], G[m])
                    fL = fL + fac * src
        psi[i] = E.assemble(fL, fT if i > 2 else None)
        out[f"psi{i}_eds"] = to_real(psi[i], n)
        if i <= n_order - 1:              # partners for the higher-order sources
            G[i] = E.G(psi[i])
    return out


# ---------------------------------------------------------------------------
# displacement models: list of (label, [(growth(a)->coef, shape key), ...])
# ---------------------------------------------------------------------------
def models(cosmo):
    D1 = cosmo.D1
    D2 = cosmo.D2
    D3a = lambda a: cosmo.D3(a)[0]
    D3b = lambda a: cosmo.D3(a)[1]
    D3c = lambda a: cosmo.D3(a)[2]
    exact3 = [(D1, "psi1"), (D2, "psi2_ex"), (D3a, "psi3a"), (D3b, "psi3b"), (D3c, "psi3c")]
    return {
        "1lpt": [(D1, "psi1")],
        "2lpt": exact3[:2],
        "3lpt": exact3,
        "4lpt": exact3 + [(lambda a: D1(a)**4, "psi4_eds")],
        # EdS-growth (D1^n) versions: size of the growth approximation
        "2lpt_eds": [(D1, "psi1"), (lambda a: D1(a)**2, "psi2_eds")],
        "3lpt_eds": [(D1, "psi1"), (lambda a: D1(a)**2, "psi2_eds"),
                     (lambda a: D1(a)**3, "psi3_eds")],
        "4lpt_eds": [(D1, "psi1"), (lambda a: D1(a)**2, "psi2_eds"),
                     (lambda a: D1(a)**3, "psi3_eds"), (lambda a: D1(a)**4, "psi4_eds")],
    }


def positions(model, shapes, a, L, dtype=np.float32):
    n = shapes["psi1"].shape[1]
    q = np.arange(n, dtype=np.float64) * (L / n)
    shp = [(n, 1, 1), (1, n, 1), (1, 1, n)]
    x = np.empty((3, n, n, n), dtype)
    for i in range(3):
        xi = np.broadcast_to(q.reshape(shp[i]), (n, n, n)).astype(np.float64)
        for g, key in model:
            xi += g(a) * shapes[key][i]
        x[i] = np.mod(xi, L)
    return x
