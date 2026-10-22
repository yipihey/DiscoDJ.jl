# ── Python / DISCO-DJ bridge ─────────────────────────────────────────────────
# All Python calls happen on the Julia main thread (PythonCall is not thread-safe);
# heavy element-wise work is done on Julia arrays with Threads.

const np      = PythonCall.pynew()
const jnp     = PythonCall.pynew()
const jax     = PythonCall.pynew()
const discodj = PythonCall.pynew()

function __init__()
    try
        PythonCall.pycopy!(np, pyimport("numpy"))
        PythonCall.pycopy!(jax, pyimport("jax"))
        PythonCall.pycopy!(jnp, pyimport("jax.numpy"))
        PythonCall.pycopy!(discodj, pyimport("discodj"))
    catch err
        @warn "DISCO-DJ Python environment not available (set JULIA_PYTHONCALL_EXE to disco-dj-fem/.venv/bin/python before loading); analysis-only mode" exception = err
    end
end

"Gaussian-filter cutoff wavenumber for DISCO-DJ's `exp(-½(k/k0)²)` on P(k), such that
W(k)=exp(-k²R_f²/2) acts on δ (P_f = P·exp(-k²R_f²))."
k0_cutoff(R_f) = 1 / (sqrt(2) * R_f)

"""
    make_dj(; N, L, R_f, cosmo="Planck18EEBAOSN", precision="single", pk_file=nothing)

DISCO-DJ object with timetables and the Gaussian-filtered linear P(k).  `cosmo` may
be a predefined name or a `Dict` of parameters.  If `pk_file` is given, the (z=0,
unfiltered) P(k) table is loaded from it (`transfer_function="from_file"`, no σ₈
renormalisation) — used to run an EdS background with the ΛCDM spectrum shape.
"""
function make_dj(; N::Int, L::Real, R_f::Real, cosmo = "Planck18EEBAOSN",
                 precision::String = "single", pk_file = nothing)
    cos_py = cosmo isa AbstractString ? cosmo : pydict(Dict(String(k) => Float64(v) for (k, v) in cosmo))
    dj = discodj.DiscoDJ(; dim = 3, res = N, boxsize = Float64(L), cosmo = cos_py, precision = precision)
    dj = dj.with_timetables()
    if pk_file === nothing
        dj = dj.with_linear_ps(; k0_cutoff = k0_cutoff(R_f), cutoff_type = "gaussian")
    else
        dj = dj.with_linear_ps(; transfer_function = "from_file", filename = pk_file, fix_sigma8 = false,
                               k0_cutoff = k0_cutoff(R_f), cutoff_type = "gaussian")
    end
    return dj
end

"Write the unfiltered z=0 linear P(k) of `cosmo` to `path` (two columns k, P) with a k-range
wide enough for grids up to `Nmax` points in a box of size `L`."
function write_pk_table(path::AbstractString; cosmo = "Planck18EEBAOSN", L::Real, Nmax::Int = 512)
    dj = discodj.DiscoDJ(; dim = 3, res = Nmax, boxsize = Float64(L), cosmo = cosmo).with_timetables().with_linear_ps()
    dj.store_linear_ps_as_txt(String(path))
    return path
end

"Tabulated (filtered) P(k) of `dj` as Julia vectors."
pk_table(dj) = (k = pyconvert(Vector{Float64}, np.asarray(dj._pk_table["k"])),
                P = pyconvert(Vector{Float64}, np.asarray(dj._pk_table["Pk"])))

"Continuum σ² = ∫ P_f(k) k² dk / (2π²) of the tabulated filtered spectrum (trapezoid in ln k)."
function sigma2_continuum(dj)
    t = pk_table(dj)
    lnk = log.(t.k); f = t.k .^ 3 .* t.P ./ (2π^2)
    return sum(0.5 .* (f[1:end-1] .+ f[2:end]) .* diff(lnk))
end

"Box (discrete-mode) σ² expected for white noise of unit variance on the N³ grid:
σ²_box = Σ_{k≠0} P_f(|k|) / L³ over the rfft modes (Hermitian pairs counted twice)."
function sigma2_box(dj; N::Int, L::Real)
    t = pk_table(dj)
    kf = 2π / L
    kx = kf .* [i <= N ÷ 2 ? i : i - N for i in 0:N-1]
    kz = kf .* collect(0:N÷2)
    s = 0.0
    @inbounds for iz in eachindex(kz), iy in eachindex(kx), ix in eachindex(kx)
        k = sqrt(kx[ix]^2 + kx[iy]^2 + kz[iz]^2)
        k == 0 && continue
        w = (iz == 1 || iz == length(kz)) ? 1.0 : 2.0    # kz=0 and Nyquist planes are self-paired
        s += w * linterp(t.k, t.P, k)
    end
    return s / L^3
end

function linterp(xs, ys, x)
    i = searchsortedlast(xs, x)
    i < 1 && return ys[1]
    i >= length(xs) && return ys[end]
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + t * (ys[i+1] - ys[i])
end

"Install a real-space white-noise field (Julia `Array{Float32,3}`) as the ICs of `dj`
(DISCO-DJ's `white_noise_space=\"real\"` route: fφ = rfftn(w)·√(P_f·(N/L)³)·(−1/k²))."
function with_white_noise(dj, w::Array{Float32,3})
    return dj.with_ics(; white_noise_space = "real", white_noise_field = np.asarray(w))
end

"Linear growth tables at scale factor `a` (D₊ normalised to 1 at a=1)."
function growth(dj, a::Real)
    c = dj.cosmo
    g(key) = pyconvert(Float64, pybuiltins.float(c.get_interpolated_property(Float64(a), "a", key)))
    return (D1 = g("Dplus"), D2 = g("D2plus"), D3a = g("D3plusa"), D3b = g("D3plusb"), D3c = g("D3plusc"))
end

"Exact-cosmology leading coefficient A₁ = (1 + D₂/D₁²)/(3√(2π)) (equals 4/(21√(2π)) in EdS)."
A1_exact(g) = (1 + g.D2 / g.D1^2) / (3 * sqrt(2π))

"δ_ini (linear density at D=1), flattened in numpy C-order, as `Vector{Float32}`."
delta_ini(dj) = pyconvert(Vector{Float32}, np.ascontiguousarray(np.ravel(np.asarray(dj.delta_ini))))

"""
    deformation_gradient(dj, psi_py) -> Vector{SMatrix{3,3,Float32,9}}

G = ∂ψ/∂q at every Lagrangian grid point (DISCO-DJ `evaluate_jacobian_from_psi`, exact
ik kernel with the Nyquist plane zeroed), flattened in C-order.  Each element is the
transpose of DISCO-DJ's (d, g) matrix; all orders use the same convention, and every
statistic computed downstream (det, traces of products, F·Fᵀ spectrum) is invariant
under this common transposition.
"""
function deformation_gradient(dj, psi_py)
    _, _, dpsi = dj.evaluate_jacobian_from_psi(psi_py; only_det = false)
    flat = pyconvert(Vector{Float32}, np.ascontiguousarray(np.ravel(np.asarray(dpsi))))
    M = reinterpret(SMatrix{3,3,Float32,9}, flat)
    G = Vector{SMatrix{3,3,Float32,9}}(undef, length(M))
    I3 = SMatrix{3,3,Float32,9}(I)
    Threads.@threads for e in eachindex(M)
        @inbounds G[e] = M[e] - I3
    end
    return G
end

"Spatial LPT displacement field ψₙ of the computed `dj` (keys as in DISCO-DJ: `psi_1`…`psi_4`,
`psi_2_ex`, `psi_3a_ex`, `psi_3b_ex`, `psi_3c_ex`)."
lpt_psi(dj, key::String) = dj._lpt.psi[key]

"Flattened (C-order) Lagrangian grid as `(N³,3)` numpy array."
grid_positions_py(N::Int, L::Real) = pyimport("discodj.nbody_fem.geometry").grid_positions(N, Float64(L))
