"""Synthetic test of tl_fit3 with known coefficients and no simulation data:  python test_tl_fit3.py"""
import numpy as np
import tl_fit as T
import tl_fit3 as F

rng = np.random.default_rng(1)
M = 120
K = np.linspace(0.01, 0.12, M)
Lam = 0.2


def rc(shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def make_ops():
    O = {name: rc((M, 6)) for name, _, _ in T.OPS}
    # the real structure of the linear stress operators: r_ij = k_i k_j/k² δ and δ^K_ij δ (unequal norm, correlated by the trace)
    kh = rng.normal(size=(M, 3)); kh /= np.linalg.norm(kh, axis=1, keepdims=True)
    dl = rc(M)
    O["C1_r1"] = np.stack([kh[:, a] * kh[:, b] * dl for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=1)
    O["C1_d1"] = np.stack([dl, dl, dl, 0 * dl, 0 * dl, 0 * dl], axis=1)
    O["resp"] = {}
    d1, p1 = rc(M), rc((M, 3))
    for lab in F.LABELS:
        O["resp"][lab] = dict(d1=d1 if lab in ("r1", "d1") else rc(M), d2=rc(M), d3=rc(M), p1=p1 if lab in ("r1", "d1") else rc((M, 3)), p2=rc((M, 3)))
    return O


OP = {s: {Lam: make_ops()} for s in F.SEEDS}
TRUE = {"r1": 3.0, "d1": 7.0}
EPOCHS = (0.3, 0.5, 0.8, 1.0)
NMIN = {0.3: 0.5, 0.5: 0.5, 0.8: 0.001, 1.0: 0.001}   # the stress floor must drop D = 0.8 and 1 only for the stress
NOISE = 1e-3


def member(s, sg):
    out = {}
    for D in EPOCHS:
        c = F.columns(OP[s][Lam], D)
        tau = 0; dens = 0; P = 0
        for l, v in TRUE.items():
            tau = tau + v * (sg * c[l]["tau_odd"] + c[l]["tau_even"])
        # r1 and d1 enter density and momentum only through their sum: same response columns above
        dens = sum(v * (sg * c[l]["dens_odd"] + c[l]["dens_even"]) for l, v in TRUE.items())
        P = sum(v * (sg * c[l]["mom_odd"] + c[l]["mom_even"]) for l, v in TRUE.items())
        out[(D, Lam)] = dict(tau=tau + NOISE * rc((M, 6)), delta=dens + NOISE * rc(M), P=P + NOISE * rc((M, 3)), curlv=rc((M, 3)), nmin=NMIN[D])
    return out


MEM = {(s, sg): member(s, sg) for s in F.SEEDS for sg in (1, -1)}
ZERO = {(s, sg): {k: dict(tau=np.zeros((M, 6), complex), delta=np.zeros(M, complex), P=np.zeros((M, 3), complex), curlv=np.zeros((M, 3), complex), nmin=1.0) for k in MEM[(s, sg)]} for s, sg in MEM}
F.load_stress2 = lambda cfg, model, s, sg: (MEM if cfg == "A" else ZERO)[(s, sg)]
F.IRCFG = {Lam: "B"}

ne, raw, Ds, kmask = F.build("A", "IR", "pm", Lam, OP, K, 0.12)
assert Ds == list(EPOCHS), Ds
assert ne.stress_epochs == [0.3, 0.5], ne.stress_epochs
assert (1, 0.8, "dens_odd") in ne.A and (1, 0.8, "tau_odd") not in ne.A
print("floor applies to the stress only: ok")

seeds = list(F.SEEDS)
for D in (0.3, 0.5):
    p, e, _, jk = F.fit_set(ne, [D], ("tau_odd",), F.LIN, seeds)
    assert abs(p[0] - 3) < 0.02 and abs(p[1] - 7) < 0.02, p
    assert abs((p[0] + p[1]) - 10) < 0.02
old = F.fit_lin_sum(ne, [D], ("tau_odd",), seeds)[0]
assert abs(old - 10) > 1, old
print("stress r1, d1 and their sum recovered:", np.round(p, 4), "| the old constrained closure gives", round(old, 2), "not 10, so it would falsely fail")
for D in EPOCHS:
    v, _ = F.fit_lin_sum(ne, [D], ("dens_odd",), seeds)
    assert abs(v - 10) < 0.1, v   # r1 and d1 differ slightly through δ³, D⁵
    v, _ = F.fit_lin_sum(ne, [D], ("mom_odd",), seeds)
    assert abs(v - 10) < 0.1, v   # r1 and d1 differ slightly through δ³, D⁵
print("density and momentum sums = 10 at every epoch: ok")

w = F.weights(ne, ("tau_odd", "dens_odd"))
vt = ne.chi2((1, 0.3, "tau_odd"), F.NE.solve(ne, lambda q: q == (1, 0.3, "tau_odd"), F.LABELS, lambda q: 1.0)[0], F.LABELS) / ne.ndof[(1, 0.3, "tau_odd")]
assert abs(w[(1, 0.3, "tau_odd")] * vt - 1) < 1e-9
vd = ne.chi2((1, 0.3, "dens_odd"), F.NE.solve(ne, lambda q: q == (1, 0.3, "dens_odd"), F.LABELS, lambda q: 1.0)[0], F.LABELS) / ne.ndof[(1, 0.3, "dens_odd")]
print("weights are 1/variance per dof: tau var", round(vt, 8), "dens var", round(vd, 8), "ratio", round(vt / vd, 2), "(same noise per component)")
assert 0.5 < vt / vd < 2

# jackknife when one seed survives at an epoch must not crash, and missing blocks give NaN, not an exception
p, e, _, _ = F.fit_set(ne, [0.8], ("tau_odd",), F.LIN, seeds)
assert np.all(np.isnan(p))
sel = F.fit_set(ne, [0.3], ("tau_odd",), F.LIN, [1])
assert np.all(np.isnan(sel[1]))
print("no stress blocks at an epoch or a single seed: NaN, no crash")

pall, *_ = ne.solve(lambda k: k[1] == 0.3 and k[2] in ("tau_odd", "tau_even"), F.LABELS, lambda k: 1.0)
sp = F.residual_spectra(raw, F.LABELS, pall, K, kmask, 0.3)
got = np.nanmean(sp["odd"]) / F.L ** 3
want = 3 * 2 * NOISE ** 2 / 2   # trace of 3 components, each complex with variance 2 NOISE^2, odd part halves it
assert 0.3 < got / want < 1.5, (got, want)
print("stochastic trace spectrum of pure noise / expected:", round(got / want, 2))
print("ALL OK")
