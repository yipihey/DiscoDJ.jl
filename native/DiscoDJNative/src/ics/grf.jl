"""
Gaussian random field generation for cosmological initial conditions.

Generates the initial Fourier-space potential φ(k) such that:
    <|φ(k)|²> = P(k) * (res/boxsize)^dim  [dimensionless]

Two modes (matching DISCO-DJ's `generate_grf`):
- :real   — draw in real space, FFT → multiply by √P(k)
- :fourier — draw directly in Fourier space with correct Hermitian symmetry

The returned field is φ(k) in the rfft layout: shape (res…, res÷2+1).
"""

export generate_grf, set_dc_zero!
export ICOperator, ic_operator, white_noise_to_fphi

using FFTW
using Random: MersenneTwister

using FFTW
using AbstractFFTs

# ── k-space helpers ───────────────────────────────────────────────────────────

"""
    inv_laplace_kernel(k2) -> Array

Returns 1/k² (with 0 at k=0). Used to convert density field to potential.
"""
function inv_laplace_kernel(k2::AbstractArray{T}) where T
    out = similar(k2)
    @inbounds for i in eachindex(k2)
        out[i] = k2[i] == 0 ? T(0) : T(1) / k2[i]
    end
    return out
end

# ── GRF generation ───────────────────────────────────────────────────────────

"""
    generate_grf(sampling_space, dim, pk_table, res, boxsize, seed;
                 dtype=Float32, dtype_c=ComplexF64) -> Array{dtype_c}

Generate the Gaussian random field φ(k) in Fourier space (rfft layout).

`sampling_space ∈ [:real, :fourier, :ngenic]`
`pk_table` — Dict("k" => k, "Pk" => Pk) from `linear_power_spectrum`.
Returns fphi of shape (res^(dim-1), res÷2+1) for dim=1, or (res,res,res÷2+1) for dim=3.
"""
function generate_grf(sampling_space::Symbol, dim::Int, pk_table::Dict,
                      res::Int, boxsize::Float64, seed::Union{Integer, AbstractVector};
                      dtype::Type=Float32, dtype_c::Type=ComplexF32,
                      white_noise::Union{Nothing,AbstractArray}=nothing)

    k_grid, k2_grid = _rfft_k_grid(dim, res, boxsize, dtype)
    Pk_interp = _interpolate_pk(pk_table, k_grid, dtype)

    fphi = if white_noise !== nothing
        # Explicit real-space white-noise field, applied through the *legacy* (+1/k²)
        # map so it stays consistent with this function's other modes and the
        # optimised `compute_lpt`.  (For the differentiable/faithful path use
        # `ic_operator` + `nlpt_core`, which uses the JAX −1/k² gauge instead.)
        dim == 3 || error("explicit white_noise only supported for dim=3")
        norm_fac = (res / boxsize)^dim
        fw = rfft(convert.(dtype, white_noise), [3, 1, 2])
        convert.(dtype_c, fw .* sqrt.(Pk_interp .* norm_fac))
    elseif sampling_space == :real
        _grf_real(dim, res, boxsize, Pk_interp, seed, dtype, dtype_c)
    elseif sampling_space == :fourier
        _grf_fourier(dim, res, boxsize, Pk_interp, seed, dtype, dtype_c)
    elseif sampling_space == :ngenic
        _grf_ngenic(dim, res, boxsize, Pk_interp, seed, dtype, dtype_c)
    else
        error("sampling_space must be :real, :fourier, or :ngenic")
    end

    # Apply inverse Laplacian to convert white noise to potential φ
    fphi .*= inv_laplace_kernel(k2_grid)

    # Zero DC mode
    set_dc_zero!(fphi)
    return fphi
end

# ── Differentiable IC map  ω → φ(k)  (faithful, JAX convention) ───────────────
# The map from a real-space unit white-noise field ω (res,res,res) to the initial
# Fourier potential  φ(k) = rfft(ω)·√(P(k)·norm)·(−1/k²)  (DC=0) is *linear* in ω,
# hence fully differentiable with no custom adjoint: `rfft` carries an AbstractFFTs
# ChainRule and the rest is a broadcast by a precomputed real `scale`.
#
# This reproduces JAX `generate_grf(white_noise_space="real")` *exactly* (the −1/k²
# inverse-Laplacian sign and the linear P(k) interpolation), so it is the IC map for
# the faithful path:  `lpt_displacement(white_noise_to_fphi(op, ω), K, …)` matches
# JAX end-to-end.  NOTE the −1/k² sign: this is the JAX/physical gauge (ψ₁ = −∇φ
# gives infall into overdensities) and pairs with `nlpt_core`/`lpt_displacement`.
# The legacy optimised `compute_lpt` uses the opposite (+1/k²) gauge throughout and
# is fed by `generate_grf` (below) — do not cross the two.

"""
    ICOperator{T}

Precomputed linear map ω → φ(k): holds `scale = √(P·norm)/k²` (0 at k=0) in the
rfft layout `(res,res,res÷2+1)`.  Build with [`ic_operator`](@ref); apply with
[`white_noise_to_fphi`](@ref).
"""
struct ICOperator{T<:AbstractFloat, A<:AbstractArray{T,3}}
    scale::A
    res::Int
    boxsize::T
end

"""
    ic_operator(res, boxsize, pk_table; T=Float32) -> ICOperator

Precompute the (grid- and cosmology-dependent) linear IC map.  Reuse across
inference iterations.
"""
function ic_operator(res::Int, boxsize::Real, pk_table::Dict; T::Type{<:AbstractFloat}=Float32)
    kgrid, k2 = _rfft_k_grid(3, res, T(boxsize), T)
    Pk  = _interpolate_pk(pk_table, kgrid, T)
    nrm = T((res / boxsize)^3)
    scale = similar(k2, T)
    # −1/k² (JAX inv_laplace_kernel sign), so φ(k) = rfft(ω)·√(P·norm)·(−1/k²).
    @inbounds for i in eachindex(k2)
        scale[i] = k2[i] == 0 ? zero(T) : -sqrt(Pk[i] * nrm) / k2[i]
    end
    return ICOperator{T, typeof(scale)}(scale, res, T(boxsize))
end

"""
    white_noise_to_fphi(op::ICOperator, white)               -> φ(k)
    white_noise_to_fphi(white, res, boxsize, pk_table; T)    -> φ(k)

Differentiable initial potential φ(k) = rfft(ω, [3,1,2]) · scale from a real-space
unit white-noise field `white` (res,res,res).  Linear in `white`, so AD frameworks
(Zygote/ChainRules) differentiate it through the AbstractFFTs `rfft` rule with no
custom adjoint.
"""
white_noise_to_fphi(op::ICOperator, white::AbstractArray{<:Real,3}) =
    rfft(white, [3, 1, 2]) .* op.scale

white_noise_to_fphi(white::AbstractArray{<:Real,3}, res::Int, boxsize::Real, pk_table::Dict;
                    T::Type{<:AbstractFloat}=eltype(white)) =
    white_noise_to_fphi(ic_operator(res, boxsize, pk_table; T=T), white)

# ── Real-space sampling ───────────────────────────────────────────────────────

function _grf_real(dim, res, boxsize, Pk_interp, seed, dtype, dtype_c)
    rng = MersenneTwister(seed)
    shape = ntuple(_ -> res, dim)
    white = randn(rng, dtype, shape)
    fwhite = rfft(white, [dim; collect(1:dim-1)])  # full n-D rfft, half-complex last
    norm_fac = (res / boxsize)^dim
    @inbounds for i in eachindex(fwhite)
        fwhite[i] *= sqrt(Pk_interp[i] * norm_fac)
    end
    return convert.(dtype_c, fwhite)
end

# ── Fourier-space sampling ────────────────────────────────────────────────────

function _grf_fourier(dim, res, boxsize, Pk_interp, seed, dtype, dtype_c)
    rng = MersenneTwister(seed)
    shape_c = (ntuple(_ -> res, dim-1)..., res÷2 + 1)
    re = randn(rng, dtype, shape_c)
    im = randn(rng, dtype, shape_c)
    fraw = complex.(re, im)
    norm_fac = (res / boxsize)^dim
    @inbounds for i in eachindex(fraw)
        fraw[i] *= sqrt(Pk_interp[i] * norm_fac / 2)
    end
    return convert.(dtype_c, fraw)
end

# ── N-GenIC sampling ──────────────────────────────────────────────────────────

function _grf_ngenic(dim, res, boxsize, Pk_interp, seed, dtype, dtype_c)
    dim == 3 || error("NGenIC mode only supports dim=3")
    fwhite = ngenic_white_noise(seed, res)
    norm_fac = (res / boxsize)^dim
    result = Array{dtype_c}(undef, res, res, res÷2+1)
    @inbounds for i in eachindex(fwhite)
        result[i] = fwhite[i] * sqrt(Pk_interp[i] * norm_fac)
    end
    return result
end

# ── k grid ────────────────────────────────────────────────────────────────────

function _rfft_k_grid(dim, res, boxsize, dtype)
    dk = dtype(2π / boxsize)
    kn = dtype(π * res / boxsize)   # Nyquist

    # 1D k-vectors for each axis
    kx_full = [i <= res÷2 ? i : i - res for i in 0:res-1] .* dk
    kz_half = collect(0:res÷2) .* dk

    if dim == 1
        k2 = kz_half.^2
        k  = abs.(kz_half)
    elseif dim == 2
        k_vecs = [kx_full, kz_half]
        grid   = collect(Iterators.product(k_vecs...))
        k2 = [sum(v.^2) for v in grid]
        k  = sqrt.(k2)
    elseif dim == 3
        k_vecs = [kx_full, kx_full, kz_half]
        grid   = collect(Iterators.product(k_vecs...))
        k2 = [sum(v.^2) for v in grid]
        k  = sqrt.(k2)
    else
        error("dim must be 1, 2, or 3")
    end

    return k, k2
end

function _interpolate_pk(pk_table::Dict, k_grid, dtype)
    # Linear interpolation in linear (k, P) space, matching JAX's
    # `np.interp(k, pk_table["k"], pk_table["Pk"])` exactly (incl. edge clamping:
    # k below/above the table → first/last P).  The earlier log-log scheme was a
    # ~1e-5-level departure from the reference.
    k_tab  = pk_table["k"]
    Pk_tab = pk_table["Pk"]
    n = length(k_tab)
    Pk_out = similar(k_grid, dtype)
    @inbounds for i in eachindex(k_grid)
        kval = Float64(k_grid[i])
        if kval <= k_tab[1]
            Pk_out[i] = dtype(Pk_tab[1])
        elseif kval >= k_tab[end]
            Pk_out[i] = dtype(Pk_tab[end])
        else
            j = searchsortedfirst(k_tab, kval)   # k_tab[j-1] < kval <= k_tab[j]
            t = (kval - k_tab[j-1]) / (k_tab[j] - k_tab[j-1])
            Pk_out[i] = dtype(Pk_tab[j-1] * (1 - t) + Pk_tab[j] * t)
        end
    end
    return Pk_out
end

"""
    set_dc_zero!(fphi)

Zero the DC (k=0) mode so there is no bulk displacement.
"""
function set_dc_zero!(fphi::AbstractArray)
    fphi[1] = zero(eltype(fphi))
    return fphi
end
