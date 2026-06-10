# DiscoDJ.jl

Julia bindings to **DISCO-DJ** (the differentiable JAX cosmology code of List,
Hahn, Winkler & Floess — here the `yipihey/DISCO-DJ` fork carrying the
phase-space-sheet / FEM work, sibling checkout `disco-dj-fem`), in the EnzoNG.jl
unified multi-code federation pattern (Enzo / RAMSES / Arepo / MUSIC / Gadget-4 /
Athena++).

**Transport matches the code's nature**: DISCO-DJ is Python/JAX, so the bridge
is an in-process Python host (PythonCall bound to the checkout's `.venv`) rather
than a C-ABI dylib — and that is what preserves the headline capability:
`build(spec).dj` is the *live JAX-traced object*, so the whole pipeline stays
**differentiable** (gradients w.r.t. cosmology/ICs on the Python side), while
the typed Julia surface feeds the federation.

```julia
ENV["JULIA_PYTHONCALL_EXE"] = expanduser("~/Projects/disco-dj-fem/.venv/bin/python")
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"          # BEFORE using DiscoDJLib
using DiscoDJLib

spec = DiscoSpec(res = 128, boxsize = 100.0, n_order = 3, seed = 42)   # 3LPT
b   = build(spec)                                # timetables → P(k) → ICs → nLPT
ic  = lpt_ics(b, 0.02)                           # ψ, pos, vel  (res³×3 Julia arrays)
lc  = lpt_lightcone(b; a_far = 0.2)              # phase-space-sheet lightcone
```

The `(ψ, pos, vel)` arrays are the canonical hand-off to the framework's
existing injectors (Enzo bridge particle setters, RAMSES grafic writers,
`Gadget4Lib.write_snapshot`) — one differentiable IC source for every code.

Validated (10/10 tests, deterministic CPU JAX): 2LPT field shapes/units/zero
bulk displacement, the linear-growth gate (ψ ∝ D(a), D∝a to 0.5% deep in matter
domination), 2LPT−Zel'dovich difference present but perturbative at z≈50, and
seed reproducibility to f32 round-off.

## Roadmap
- MultiCode injectors: `DiscoSpec` as a cosmological problem-setup source next
  to `MusicSpec`/`GenicSpec`, with a 3-way IC cross-check (DISCO-DJ vs MUSIC vs
  NGenIC at matched seeds/cosmology).
- Gradient pass-through to Julia (cosmology-derivative fields via `jax.grad` on
  user-specified summaries; `requires_grad_wrt_cosmo=true` route).
- Lightcone products → EnzoViz / healpix maps; PM `run_nbody` + FEM/cascade
  sheet tools behind typed wrappers.
