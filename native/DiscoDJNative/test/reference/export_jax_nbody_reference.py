"""Export parity reference data from the JAX DISCO-DJ `run_nbody` (yipihey/disco-dj)
for the Julia port in src/nbody/.  Float64 throughout.

    LD_LIBRARY_PATH=<gsl>/lib python3 export_jax_nbody_reference.py [out.h5]

Stored (all arrays in the JAX C-order particle layout (n,n,n,3) / (n,n,n)):
  /cosmo/{a, Dplus, Dplusda, D2plus, D2plusda, superconft}, attrs Dplus_unnormed_at_1, Omega_m
  /ics/{psi_ini, pi_ini, psi1}               (2LPT at a_ini; pi = dPsi/dD; psi1 = 1LPT shape)
  /ics/{fphi_re, fphi_im}                    the IC potential fed to nLPT (rfft layout, z halved)
  /<config>/{alpha, beta, ddrift1, ddrift2, all_a}      stepper coefficients
  /<config>/acc_ini                                     acceleration at psi_ini (force settings)
  /<config>/{psi_out, p_out}                            run_nbody(return_displacement=True)
  /<config>/attrs  the full run_nbody keyword set (JSON)

Upstream bug (kept as evidence, not a parity target): `interpolate_field(which="linear")`
builds the Lagrangian grid in box units of 1 and then divides the shifted coordinate by the
real boxsize, `(q_rel + dshift*L/res) / L * res`, so for L != 1 every resampled particle is
interpolated near grid index `dshift`.  Config "bullfrog_resample_linear" records that
behaviour; "bullfrog_resample_linear_fixed" is produced with the corrected coordinate
`(q_rel*res + dshift)` monkeypatched in and is the parity target for the Julia port.
"""
import json, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import h5py
from functools import partial
from discodj import DiscoDJ
from discodj.core.grids import get_fourier_grid
from discodj.core.utils import set_0_to_val
from discodj.nbody.acc import calc_acc_PM
from discodj.nbody.steppers.dkd_pi_integrator import DKDPiIntegrator
from discodj.nbody.steppers.dkd_symplectic_integrator import DKDSymplecticIntegrator

OUT = sys.argv[1] if len(sys.argv) > 1 else "jax_nbody_reference.h5"
RES, L, A_INI, A_END, SEED = 16, 100.0, 0.05, 1.0, 1

dj = (DiscoDJ(dim=3, res=RES, boxsize=L, precision="double", cosmo="Planck18EEBAOSN")
      .with_timetables().with_linear_ps().with_ics(seed=SEED).with_lpt(n_order=2))
mesh = lambda x: np.asarray(dj.ensure_mesh_shape(x))

BASE = dict(a_ini=A_INI, a_end=A_END, n_steps=6, time_var="D", stepper="bullfrog", method="pm", res_pm=32,
            antialias=0, grad_kernel_order=4, laplace_kernel_order=0, worder=2, deconvolve=False,
            n_resample=1, resampling_method="fourier", ic_method="lpt", nlpt_order_ics=2)
CONFIGS = {
    "bullfrog_D_cic":            {},
    "fastpm_D_tsc":              dict(stepper="fastpm", worder=3),
    "symplectic_a_pcs_deconv":   dict(stepper="symplectic", time_var="a", worder=4, deconvolve=True),
    "symplectic_superconft":     dict(stepper="symplectic", time_var="superconft"),
    "bullfrog_loga_ik_aa1":      dict(time_var="log_a", grad_kernel_order=0, antialias=1),
    "bullfrog_D_aa2_lap2":       dict(antialias=2, laplace_kernel_order=2, grad_kernel_order=2),
    "bullfrog_D_aa3_grad6_lap4": dict(antialias=3, grad_kernel_order=6, laplace_kernel_order=4),
    "bullfrog_D_aam1_lap6":      dict(antialias=-1, laplace_kernel_order=6),
    "bullfrog_resample_fourier": dict(n_resample=2, worder=4, deconvolve=True, antialias=1, grad_kernel_order=0),
    "bullfrog_resample_linear":  dict(n_resample=2, resampling_method="linear"),
    "bullfrog_resample_linear_fixed": dict(n_resample=2, resampling_method="linear"),
    "bullfrog_ic_bullfrog":      dict(ic_method="bullfrog", nlpt_order_ics=None),
    "bullfrog_explicit_a":       dict(time_var=np.array([0.05, 0.1, 0.2, 0.35, 0.6, 1.0]), n_steps=5),
}

import discodj.core.scatter_and_gather as sg
_orig_interp = sg.interpolate_field


def _interpolate_field_fixed(dim, field, dshift, boxsize, dtype_num, which="fourier", with_jax=True):
    """interpolate_field with the linear-branch coordinate fixed (grid index i + dshift)."""
    if which != "linear":
        return _orig_interp(dim, field, dshift, boxsize, dtype_num, which=which, with_jax=with_jax)
    from jax.scipy import ndimage
    res = field.shape[0]
    idx = [jnp.arange(res, dtype=field.dtype) + dshift[d] for d in range(dim)]
    coords = jnp.moveaxis(jnp.asarray(jnp.meshgrid(*idx, indexing="ij")), 0, -1).reshape(-1, dim)
    out = []
    for d in range(field.shape[-1]):
        fp = jnp.pad(field[..., d], tuple([(0, 1)] * dim), mode="wrap")
        out.append(ndimage.map_coordinates(fp, coords.T, order=1, mode="wrap"))
    return jnp.moveaxis(jnp.asarray(out), 0, -1).reshape(field.shape)


with h5py.File(OUT, "w") as f:
    tt = dj.cosmo._timetables
    g = f.create_group("cosmo")
    for k in ("a", "Dplus", "Dplusda", "D2plus", "D2plusda", "superconft"):
        g[k] = np.asarray(tt[k])
    g.attrs["Dplus_unnormed_at_1"] = float(tt["Dplus_unnormed_at_1"])
    g.attrs["Omega_m"] = float(dj.cosmo.Omega_m)
    g.attrs.update(dict(res=RES, boxsize=L))

    psi_ini = dj.evaluate_lpt_psi_at_a(A_INI, n_order=2)
    pi_ini = dj._evaluate_lpt_property_at_a(a=A_INI, n_order=2, include_psi_0=False, D_derivative=True)
    psi1 = dj._evaluate_lpt_property_at_a(a=1.0, n_order=1, include_psi_0=False, D_derivative=True)
    gi = f.create_group("ics")
    gi["psi_ini"], gi["pi_ini"], gi["psi1"] = mesh(psi_ini), mesh(pi_ini), mesh(psi1)
    fphi = np.asarray(dj._ics["fphi"])                   # (n, n, n//2+1) complex, C order
    gi["fphi_re"], gi["fphi_im"] = fphi.real, fphi.imag

    for name, over in CONFIGS.items():
        kw = dict(BASE, **over)
        sg.interpolate_field = _interpolate_field_fixed if name.endswith("_fixed") else _orig_interp
        tv = kw["time_var"]
        cls = DKDSymplecticIntegrator if kw["stepper"] == "symplectic" else DKDPiIntegrator
        solver = cls(cosmo=dj.cosmo, time_dict=dict(n_steps=kw["n_steps"], a_ini=kw["a_ini"], a_end=kw["a_end"],
                                                    time_var=jnp.asarray(tv) if isinstance(tv, np.ndarray) else tv),
                     integrator_name=kw["stepper"], use_diffrax=False, dtype_num=64)
        args = solver.get_integrator_args()
        grp = f.create_group(name)
        for k in ("alpha", "beta", "ddrift1", "ddrift2"):
            grp[k] = np.asarray(args[k])
        grp["all_a"] = np.asarray(solver.all_a_and_internal[0])

        k_dict = get_fourier_grid(shape=(kw["res_pm"],) * 3, boxsize=L, sparse_k_vecs=True, full=False,
                                  relative=False, dtype_num=64, with_jax=False)
        acc = calc_acc_PM(dj.ensure_flat_shape(psi_ini), dim=3, res_pm=kw["res_pm"], n_part=RES,
                          k_vecs=k_dict["k_vecs"], k=set_0_to_val(3, k_dict["|k|"], 1.0), boxsize=L,
                          antialias=kw["antialias"], grad_order=kw["grad_kernel_order"],
                          lap_order=kw["laplace_kernel_order"], dtype_num=64, n_resample=kw["n_resample"],
                          resampling_method=kw["resampling_method"], worder=kw["worder"],
                          deconvolve=kw["deconvolve"], with_jax=True)
        grp["acc_ini"] = mesh(acc)

        run_kw = {k: v for k, v in kw.items() if v is not None}
        if isinstance(tv, np.ndarray):
            run_kw["time_var"] = jnp.asarray(tv)
        psi_out, p_out, _ = dj.run_nbody(**run_kw, return_displacement=True)
        grp["psi_out"], grp["p_out"] = np.asarray(psi_out), np.asarray(p_out)
        grp.attrs["kwargs"] = json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()})
        for k, v in kw.items():                        # plain attributes (readable without JSON)
            if isinstance(v, np.ndarray):
                grp["time_var_array"] = v
                grp.attrs[k] = "array"
            elif v is None:
                grp.attrs[k] = "none"
            else:
                grp.attrs[k] = v
        print(f"{name}: max|psi_out| {np.abs(psi_out).max():.3f}", flush=True)
print("wrote", OUT)
