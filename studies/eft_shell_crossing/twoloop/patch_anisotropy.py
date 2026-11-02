#!/usr/bin/env python3
"""Anisotropy of Lagrangian patches: how ln V at fixed delta_L departs from spherical collapse -> $EFTSC_SCRATCH/patchaniso/, analysis below.

    python patch_anisotropy.py <cfg> <seeds> <signs> <D,D,...> [threads]

For a Lagrangian top-hat of radius R0 around lattice points (stride 4), from the PM snapshot and the linear field psi_1 (LPT shapes file):
  V    signed P1 volume of the patch image (cell volumes filtered; the same quantity as sigmaM.jl)
  F_ij deformation gradient averaged over the patch, delta_ij + <d_j Psi_i>_R0: its singular values a1>=a2>=a3 are the axes of the patch's best-fit image ellipsoid
  dL   linear overdensity of the patch; T_ij its linear tidal tensor -d_i d_j phi, with standard shear e = (l1-l3)/(2 dL), p = (l1-2 l2+l3)/(2 dL)
Stored per radius: every sampled point's (nu, lnV, a3/a1, a2/a1, e_abs, p_abs, detF).
"""
import os, sys, numpy as np, h5py, scipy.fft as sf
cfg, seeds, signs, Ds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], [int(x) for x in sys.argv[3].split(",")], [float(x) for x in sys.argv[4].split(",")]
W = int(sys.argv[5]) if len(sys.argv) > 5 else 16
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing"); OD = os.path.join(D0, "patchaniso"); os.makedirs(OD, exist_ok=True)
L = 500.0; N = 256; dq = L / N; R0S = np.array([10.5, 14.8, 20.7, 29.0, 51.0]); ST = 4
kf = 2 * np.pi / L * np.fft.fftfreq(N, 1 / N); kh = 2 * np.pi / L * np.arange(N // 2 + 1)
KZ, KY, KX = np.meshgrid(kf, kf, kh, indexing="ij"); KV = (KX, KY, KZ); K2 = KZ ** 2 + KY ** 2 + KX ** 2; KM = np.sqrt(K2); K2s = np.where(K2 == 0, 1, K2)
def tophat(x):
    x = np.maximum(x, 1e-4); return 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
rf = lambda a: sf.rfftn(a, workers=W); irf = lambda a: sf.irfftn(a, s=(N, N, N), workers=W)
# h5py reverses the dimension order of Julia's (x, y, z, comp) to (comp, z, y, x), but not the values along each dimension: comp 0 = x, and array axis 0 = z.
# So component c runs along array axis 2 - c (verified: curl/div of psi_1 is 5e-7 with this pairing, 0.54 without).  KV[c] is the wavevector of component c.
def cell_volume(P):
    """Signed P1 volume per cell / dq³ from six tetrahedra (positions as displacement + lattice offset)."""
    def pos(o):
        out = []
        for c in range(3):
            a = P[c]
            for ax in range(3):
                if o[ax]: a = np.roll(a, -1, axis=2 - ax)
            out.append(a + dq * o[c])
        return out
    tets = (((0,0,0),(1,0,0),(1,1,0),(1,1,1)), ((0,0,0),(1,0,0),(1,0,1),(1,1,1)), ((0,0,0),(0,1,0),(1,1,0),(1,1,1)), ((0,0,0),(0,1,0),(0,1,1),(1,1,1)), ((0,0,0),(0,0,1),(1,0,1),(1,1,1)), ((0,0,0),(0,0,1),(0,1,1),(1,1,1)))
    sgn = (1, -1, -1, 1, 1, -1); V = 0; cell = {}
    for t, s in zip(tets, sgn):
        for o in t:
            if o not in cell: cell[o] = pos(o)
        A, B, C, E = (cell[o] for o in t)
        u = [B[i] - A[i] for i in range(3)]; v = [C[i] - A[i] for i in range(3)]; w = [E[i] - A[i] for i in range(3)]
        V = V + s * (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])) / 6
    return V / dq ** 3
def unsmoothed(J):
    return dict(mean=float(J.mean()), frac_neg=float((J < 0).mean()), std_ln=float(np.log(np.maximum(np.abs(J), 1e-6)).std()), std=float(J.std()))
for s in seeds:
    with h5py.File(os.path.join(D0, cfg, "snap", f"lpt_IR_L{s}_S0_s+1_shapes.h5")) as h:
        psi1 = h["psi_1"][()].astype(np.float64)
    for sg in signs:
        P1 = psi1 if sg == 1 else None
        if sg == -1:
            with h5py.File(os.path.join(D0, cfg, "snap", f"lpt_IR_L{s}_S0_s-1_shapes.h5")) as h:
                P1 = h["psi_1"][()].astype(np.float64)
        Fp = [rf(P1[c]) for c in range(3)]
        dLk = -sum(1j * KV[c] * Fp[c] for c in range(3))
        phik = -dLk / K2s; phik[0, 0, 0] = 0                                   # ∇²φ = δ_L
        for D in Ds:
            f = os.path.join(OD, f"{cfg}_L{s}_s{sg:+d}_D{D:.3f}.h5")
            if os.path.exists(f): continue
            with h5py.File(os.path.join(D0, cfg, "snap", f"pm_IR_L{s}_S0_s{sg:+d}_D{D:.3f}.h5")) as h:
                P = h["Psi"][()].astype(np.float64)
            J = cell_volume(P); uns = unsmoothed(J); FJ = rf(J); FP = [rf(P[c]) for c in range(3)]
            with h5py.File(f + ".tmp", "w") as h:
                h.attrs["D"] = D
                for k, v in uns.items(): h.attrs["J_" + k] = v
                for R0 in R0S:
                    W_ = tophat(KM * R0); samp = lambda a: a[::ST, ::ST, ::ST].ravel()
                    lnV = np.log(np.maximum(samp(irf(FJ * W_)), 1e-3)); okV = samp(irf(FJ * W_)) > 0
                    dL = samp(irf(dLk * W_)); sL = float(dL.std())
                    Fm = np.empty((lnV.size, 3, 3))
                    for i in range(3):
                        for j in range(3):
                            Fm[:, i, j] = (1.0 if i == j else 0.0) + samp(irf(1j * KV[j] * FP[i] * W_))
                    sv = np.linalg.svd(Fm, compute_uv=False); detF = np.linalg.det(Fm)
                    T = np.empty((lnV.size, 3, 3))
                    for i in range(3):
                        for j in range(i, 3):
                            T[:, i, j] = samp(irf(-1j * KV[i] * 1j * KV[j] * phik * W_ * -1)); T[:, j, i] = T[:, i, j]
                    lam = np.sort(np.linalg.eigvalsh(T), axis=1)[:, ::-1]               # l1 >= l2 >= l3, sum = δ_L
                    g = h.create_group(f"R{R0:.1f}"); g.attrs["R0"] = R0; g.attrs["sig_dL"] = sL
                    g["nu"] = (dL * D / (D * sL)).astype(np.float32); g["lnV"] = lnV.astype(np.float32); g["ok"] = okV; g["a31"] = (sv[:, 2] / sv[:, 0]).astype(np.float32); g["a21"] = (sv[:, 1] / sv[:, 0]).astype(np.float32); g["detF"] = detF.astype(np.float32)
                    g["l1"] = lam[:, 0].astype(np.float32); g["l2"] = lam[:, 1].astype(np.float32); g["l3"] = lam[:, 2].astype(np.float32); g["trT"] = dL.astype(np.float32)
            os.replace(f + ".tmp", f)
            print(os.path.basename(f), "J", {k: round(v, 3) for k, v in uns.items()}, flush=True)
print("PATCH DONE")
