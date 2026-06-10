"""
    DiscoDJLib

Julia bindings to **DISCO-DJ** (the differentiable JAX cosmology code of List,
Hahn, Winkler & Floess; here the `yipihey/DISCO-DJ` fork with the phase-space-
sheet / FEM work) for the EnzoNG.jl unified multi-code framework — the same
federation pattern as the Enzo / RAMSES / Arepo / MUSIC / Gadget-4 / Athena++
wrappers, with the transport adapted to the code's nature: DISCO-DJ is a Python/
JAX package, so the bridge is an **in-process Python host** (PythonCall against
the checkout's recorded interpreter, the EnzoViz pattern) instead of a C-ABI
dylib. Differentiability is preserved: the wrapped object is the live JAX-traced
`DiscoDJ` instance, so gradients remain available on the Python side.

Wrapped capabilities (the user-facing federation surface):

- **High-order LPT cosmological ICs**: [`DiscoSpec`](@ref) + [`build`](@ref) +
  [`lpt_ics`](@ref) — nLPT displacement/velocity/position fields at any scale
  factor, returned as Julia arrays in box units, ready for the framework's
  existing injectors (Enzo bridge particle setters, RAMSES grafic writers,
  `Gadget4Lib.write_snapshot`).
- **Phase-space-sheet lightcones**: [`lpt_lightcone`](@ref) wraps
  `evaluate_lpt_lightcone` (LPT-evolved sheet between two scale factors).
- The full `DiscoDJ` Python object is exposed (`build(spec).dj`) for anything
  beyond the typed surface (PM N-body via `run_nbody`, PPT, FEM/cascade tools).

Environment: the Python interpreter is resolved from `ENV["DISCODJ_PYTHON"]`, a
`.python-path` file next to this package, or the sibling checkout's `.venv`
(`/Users/tabel/Projects/disco-dj-fem/.venv/bin/python`). JAX runs on CPU by
default (`JAX_PLATFORMS=cpu`) for deterministic federation gates; unset
`DISCODJ_FORCE_CPU` to use the checkout's default device.

!!! warning "Set the interpreter BEFORE `using DiscoDJLib`"
    PythonCall binds its interpreter when *it* loads, and this module's body
    does not re-run from the precompile cache — so on cached loads the in-module
    `ENV` defaulting below has no effect. In a fresh session do
    `ENV["JULIA_PYTHONCALL_EXE"] = "<…>/disco-dj-fem/.venv/bin/python";
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"` (or export `DISCODJ_PYTHON` and let
    your entry script copy it across, as `test/runtests.jl` does) before
    `using DiscoDJLib`.
"""
module DiscoDJLib

export DiscoSpec, build, lpt_ics, lpt_lightcone, available, pypath

# ── interpreter resolution (must precede PythonCall's init) ───────────────────
function _default_python()
    env = get(ENV, "DISCODJ_PYTHON", "")
    isempty(env) || return env
    rec = joinpath(@__DIR__, "..", ".python-path")
    isfile(rec) && return strip(read(rec, String))
    # sibling checkout convention: Projects/DiscoDJ.jl/lib/DiscoDJLib/src → Projects/disco-dj-fem
    return normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python"))
end

const PYEXE = _default_python()
if !haskey(ENV, "JULIA_PYTHONCALL_EXE") && isfile(PYEXE)
    ENV["JULIA_PYTHONCALL_EXE"] = PYEXE
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
end
get(ENV, "DISCODJ_FORCE_CPU", "1") == "1" && get!(ENV, "JAX_PLATFORMS", "cpu")

using PythonCall

"Path of the Python interpreter the bridge runs DISCO-DJ in."
pypath() = PYEXE

"True when the recorded interpreter exists and `import discodj` succeeds."
function available()
    isfile(PYEXE) || return false
    try
        pyimport("discodj")
        return true
    catch
        return false
    end
end

# ── the typed problem spec (the federation's MusicSpec/GenicSpec analogue) ────
"""
    DiscoSpec

A differentiable-cosmology IC/lightcone problem for DISCO-DJ.

- `dim`, `res` — dimensionality (1/2/3) and particles per dimension
- `boxsize` — comoving box size [Mpc/h]
- `cosmo` — named DISCO-DJ cosmology (e.g. `"Planck18EEBAOSN"`) or a
  `Dict{String,Float64}` of parameters (the differentiable route)
- `n_order` — LPT order (1 = Zel'dovich, 2 = 2LPT, 3, …) — "high order" is the point
- `seed` — white-noise seed (NGenIC-compatible noise generator)
- `transfer` — linear power spectrum transfer function (`"Eisenstein-Hu"`, `"CLASS"`…)
- `precision` — `"single"` (default) or `"double"`
"""
Base.@kwdef struct DiscoSpec
    dim::Int                = 3
    res::Int                = 64
    boxsize::Float64        = 100.0
    cosmo::Union{String,Dict{String,Float64}} = "Planck18EEBAOSN"
    n_order::Int            = 2
    seed::Int               = 42
    transfer::String        = "Eisenstein-Hu"
    precision::String       = "single"
end

"""
    build(spec::DiscoSpec) -> (; dj, spec)

Construct and fully prime the DISCO-DJ pipeline for `spec`
(timetables → linear P(k) → ICs → nLPT). The returned `dj` is the live JAX
`DiscoDJ` object — keep it and evaluate at as many scale factors as you like
(each `evaluate_*` is a cheap traced call), or use it directly for gradients,
PM N-body, PPT, and the FEM/cascade machinery.
"""
function build(spec::DiscoSpec)
    discodj = pyimport("discodj")
    cosmo = spec.cosmo isa String ? spec.cosmo : pydict(spec.cosmo)
    dj = discodj.DiscoDJ(; dim = spec.dim, res = spec.res, boxsize = spec.boxsize,
                         cosmo = cosmo, device = "cpu", precision = spec.precision)
    dj = dj.with_timetables()
    dj = dj.with_linear_ps(transfer_function = spec.transfer)
    dj = dj.with_ics(seed = spec.seed)
    dj = dj.with_lpt(n_order = spec.n_order)
    return (; dj, spec)
end

_np(x) = pyconvert(Array{Float64}, pyimport("numpy").asarray(x))

"""
    lpt_ics(built, a; n_order=nothing) -> (; psi, pos, vel, a)

Evaluate the LPT initial conditions of a [`build`](@ref) result at scale factor
`a`: displacement field `psi`, positions `pos` (box units, periodic), and
conjugate velocities `vel` (`ψ̇`), each a Julia `Array{Float64}` of shape
`(res…, dim)`. `n_order` overrides the spec's LPT order (≤ the built order).
These arrays are the canonical hand-off to the framework's per-code injectors.
"""
function lpt_ics(built, a::Real; n_order::Union{Nothing,Int} = nothing)
    dj = built.dj
    kw = n_order === nothing ? (;) : (; n_order = n_order)
    psi = _np(dj.evaluate_lpt_psi_at_a(a; kw...))
    pos = _np(dj.evaluate_lpt_pos_at_a(a; kw...))
    vel = _np(dj.evaluate_lpt_psi_dot_at_a(a; kw...))
    return (; psi, pos, vel, a = Float64(a))
end

"""
    lpt_lightcone(built; a_far, a_near=1.0, n_shells=64, kwargs...) -> Py

Evaluate the LPT phase-space-sheet lightcone between `a_far` and `a_near`
(DISCO-DJ's `evaluate_lpt_lightcone`; extra `kwargs` pass through). Returns the
Python result object (shell structure is fork-specific; convert fields with
`pyconvert` as needed).
"""
lpt_lightcone(built; a_far::Real, a_near::Real = 1.0, n_shells::Int = 64, kwargs...) =
    built.dj.evaluate_lpt_lightcone(; a_far = a_far, a_near = a_near, n_shells = n_shells, kwargs...)

end # module
