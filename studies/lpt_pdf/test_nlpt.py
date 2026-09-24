"""Validation of the nLPT port.

1. Spherical-collapse test.  At a point of cubic symmetry the displacement
   gradient is isotropic at every order and the sources there are local, so
   the central Lagrangian radius must follow the EdS top-hat LPT series
       R/R0 = 1 - d/3 - d^2/21 - 23 d^3/1701 - 1894 d^4/392931 ...
   with d the central linear density.  Each order n is checked separately.
2. EdS consistency of the exact-growth shapes:
       psi3_eds == 1/3 psi3a - 10/21 psi3b + 1/7 psi3c.
3. Plane wave: all orders >= 2 vanish (Zel'dovich exact in 1D).
4. Cauchy invariants (zero Lagrangian vorticity) on a random smooth field fix
   the transverse parts:  curl psi3 = 1/3 eps_ijk d_j psi1_l d_k psi2_l  and
   curl psi4 = 1/2 eps_ijk d_j psi1_l d_k psi3_l  (EdS shapes, x = q + sum D^n psi_n).
"""
import numpy as np
import core
import nlpt

L, n = 100.0, 64


def central_grad(field, n, L):
    """d psi_x / d x at the box centre via spectral derivative."""
    kx = nlpt.grad_kernels(n, L)[0]
    g = core.irfftn(kx * core.rfftn(field[0].astype(np.float64)), n)
    c = n // 2
    return g[c, c, c]


def test_spherical(dealias=True):
    q = (np.arange(n) - n // 2) * (L / n)
    r2 = q[:, None, None]**2 + q[None, :, None]**2 + q[None, None, :]**2
    delta = np.exp(-r2 / (2 * 8.0**2))
    delta -= delta.mean()
    dc = delta[n // 2, n // 2, n // 2]
    sh = nlpt.compute_shapes(core.rfftn(delta).astype(np.complex64), n, L, dealias, 4)
    want = {1: -1 / 3, 2: -1 / 21, 3: -23 / 1701, 4: -1894 / 392931}
    got = {1: central_grad(sh["psi1"], n, L) / dc}
    for k in (2, 3, 4):
        got[k] = central_grad(sh[f"psi{k}_eds"], n, L) / dc**k
    ok = True
    for k in want:
        rel = got[k] / want[k] - 1
        print(f"  order {k}: {got[k]: .6e}  expected {want[k]: .6e}  rel.err {rel: .1e}")
        ok &= abs(rel) < 1e-3
    # exact-growth shapes in the EdS combination
    comb = sh["psi3a"] / 3 - 10 / 21 * sh["psi3b"] + sh["psi3c"] / 7
    e3 = np.abs(comb - sh["psi3_eds"]).max() / np.abs(sh["psi3_eds"]).max()
    e2 = np.abs(-3 / 7 * sh["psi2_ex"] - sh["psi2_eds"]).max() / np.abs(sh["psi2_eds"]).max()
    print(f"  EdS combination: psi2 {e2:.1e}, psi3 {e3:.1e}")
    return ok and e3 < 1e-4 and e2 < 1e-5


def test_plane_wave():
    x = np.arange(n) * (L / n)
    delta = np.broadcast_to(0.5 * np.cos(2 * np.pi * 3 * x / L)[:, None, None], (n, n, n)).copy()
    sh = nlpt.compute_shapes(core.rfftn(delta).astype(np.complex64), n, L, True, 4)
    s1 = np.abs(sh["psi1"]).max()
    rel = max(np.abs(sh[k]).max() for k in ("psi2_ex", "psi3a", "psi3b", "psi3c", "psi4_eds")) / s1
    print(f"  plane wave: max higher-order / psi1 = {rel:.1e}")
    return rel < 1e-6


def _grad(f, n):
    ks = nlpt.grad_kernels(n, L)
    fk = core.rfftn(f.astype(np.float64))
    return [core.irfftn(k * fk, n) for k in ks]


def test_cauchy():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((n, n, n))
    kf, kh = nlpt._k1d(n, L)
    k = np.sqrt(kf[:, None, None]**2 + kf[None, :, None]**2 + kh[None, None, :]**2)
    dk = core.rfftn(w) * np.exp(-(k * 6.0)**2 / 2)          # smooth: aliasing negligible
    dk *= 0.3 / core.irfftn(dk, n).std()
    sh = nlpt.compute_shapes(dk.astype(np.complex64), n, L, True, 4)
    D = {m: [_grad(sh[key][l], n) for l in range(3)]            # D[m][l][j] = d_j psi_m,l
         for m, key in ((1, "psi1"), (2, "psi2_eds"), (3, "psi3_eds"), (4, "psi4_eds"))}
    ok = True
    for order, (a, b, fac) in ((3, (1, 2, 1 / 3)), (4, (1, 3, 1 / 2))):
        curl = [D[order][2][1] - D[order][1][2], D[order][0][2] - D[order][2][0],
                D[order][1][0] - D[order][0][1]]
        rhs = []
        for i in range(3):
            j, kk = (i + 1) % 3, (i + 2) % 3
            rhs.append(fac * sum(D[a][l][j] * D[b][l][kk] - D[a][l][kk] * D[b][l][j] for l in range(3)))
        num = np.sqrt(sum(((c - r)**2).mean() for c, r in zip(curl, rhs)))
        den = np.sqrt(sum((r**2).mean() for r in rhs))
        print(f"  Cauchy invariant order {order}: |curl - rhs| / |rhs| = {num / den:.1e}")
        ok &= num / den < 1e-3
    return ok


if __name__ == "__main__":
    print("spherical collapse (de-aliased)")
    a = test_spherical(True)
    print("spherical collapse (no de-aliasing)")
    b = test_spherical(False)
    c = test_plane_wave()
    d = test_cauchy()
    print("PASS" if a and b and c and d else "FAIL")
