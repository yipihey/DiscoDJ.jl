#!/usr/bin/env python3
"""Are the dips in kR(M) from Delta^2(k) = sigma^2(R) BAO?  Same linear mapping with ln P smoothed in ln k (Gaussian width 0.25) -> results/k_of_R_nowiggle.json"""
import json, os
import h5py, numpy as np
from scipy.ndimage import gaussian_filter1d
D0 = os.environ.get("EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
with h5py.File(os.path.join(D0, "tl_full", "tl", "ops_sharp_L1.h5")) as h:
    k0, P0 = h["pk_k"][()], h["pk_P"][()]
lk = np.linspace(np.log(1e-4), np.log(3), 6000); lP = np.interp(lk, np.log(k0), np.log(P0)); nw = gaussian_filter1d(lP, 0.25 / (lk[1] - lk[0]), mode="nearest")
Rs = np.exp(np.linspace(np.log(4), np.log(120), 64)); th = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
def kR(y):
    k = np.exp(lk); D2 = k ** 3 * np.exp(y) / (2 * np.pi ** 2); s = k < 1.6
    sig = np.array([np.trapz((D2 * th(k * R) ** 2)[s], lk[s]) for R in Rs])
    return np.exp(np.interp(np.log(sig), np.maximum.accumulate(np.log(D2[s])), lk[s])) * Rs
q = np.exp(lk); r = np.exp(lP - nw); m = (q > 0.03) & (q < 0.5)
peak = q[1:-1][(r[1:-1] > r[:-2]) & (r[1:-1] > r[2:]) & m[1:-1]]; trough = q[1:-1][(r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]) & m[1:-1]]
out = dict(R=Rs.tolist(), kR_bao=kR(lP).tolist(), kR_nowiggle=kR(nw).tolist(), peaks=peak.tolist(), troughs=trough.tolist(), period=float(np.diff(peak).mean()))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "k_of_R_nowiggle.json"), "w"), indent=0)
print("peaks", np.round(peak, 3), "period", round(out["period"], 3))
