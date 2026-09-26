# Diagnostic for a failed test 1d (not part of the pass/fail gate): per-seed tidal-tensor moments and
# -log10 det J quantiles of the ZA band seeds, computed independently of the Julia sheet pipeline.
#   python3 diag_test_d.py 101 102 ... 108   -> results/step1/diag_test_d_<seeds>.json
import sys, itertools, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import common
from validate import doroshkevich
L = 300.0; qs = np.array([0.005,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.995])
def refine(p):
    n = p.shape[-1]; e = 2*n; h = n//2
    out = np.empty((3, e, e, e))
    for c in range(3):
        F = np.fft.rfftn(p[c]); G = np.zeros((e, e, e//2+1), complex)
        for ys, yd in ((slice(0,h),slice(0,h)), (slice(h+1,n),slice(e-h+1,e))):
            for xs, xd in ((slice(0,h),slice(0,h)), (slice(h+1,n),slice(e-h+1,e))):
                G[xd, yd, :h] = F[xs, ys, :h]
        out[c] = np.fft.irfftn(G * (e/n)**3, s=(e,e,e), axes=(0,1,2))
    return out
res = {}
seeds = [int(s) for s in sys.argv[1:]]
for sd in seeds:
    p = refine(np.load(f"{common.SCRATCH}/psi_1lpt_N128_z0_seed{sd}.npy"))
    e = p.shape[-1]; h = L/e
    lr = []; mom = dict(tr2=0., tr4=0., d2=0., dd=0., o2=0., asym=0.); N = 0
    for path in itertools.permutations(range(3)):
        M = np.empty((3, 3, e, e, e))
        shift = [0,0,0]
        for a in path:
            base = np.roll(p, [-s for s in shift], axis=(1,2,3))
            nxt = np.roll(base, -1, axis=1+a)
            M[:, a] = (nxt - base)/h
            shift[a] += 1
        J = M + np.eye(3)[:, :, None, None, None]
        det = np.linalg.det(np.moveaxis(J, (0,1), (-2,-1)).reshape(-1,3,3))
        k = det > 0; lr.append(-np.log10(det[k]))
        S = 0.5*(M + M.transpose(1,0,2,3,4)); tr = -(S[0,0]+S[1,1]+S[2,2])  # linear δ = -tr ∇ψ
        mom["tr2"] += np.sum(tr**2); mom["tr4"] += np.sum(tr**4)
        mom["d2"] += np.mean([np.sum(S[i,i]**2) for i in range(3)])
        mom["dd"] += np.mean([np.sum(S[i,i]*S[j,j]) for i,j in ((0,1),(0,2),(1,2))])
        mom["o2"] += np.mean([np.sum(S[i,j]**2) for i,j in ((0,1),(0,2),(1,2))])
        mom["asym"] += np.sum((0.5*(M-M.transpose(1,0,2,3,4)))**2)
        N += e**3; del M, J, S
    lr = np.concatenate(lr)
    s2 = mom["tr2"]/N
    r = dict(sigma=float(np.sqrt(s2)), kurt_excess=float(mom["tr4"]/N/s2**2-3),
             d2_over_s2=mom["d2"]/N/s2, dd_over_s2=mom["dd"]/N/s2, o2_over_s2=mom["o2"]/N/s2,
             asym_over_s2=mom["asym"]/N/s2, q_mass=np.quantile(lr, qs).tolist())
    res[sd] = r; print(sd, {k: (round(v,5) if isinstance(v,float) else None) for k,v in r.items()}, flush=True)
json.dump(res, open(f"{common.OUT}/step1/diag_test_d_{'_'.join(sys.argv[1:])}.json","w"))
