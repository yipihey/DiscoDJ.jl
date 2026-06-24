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
                      dtype::Type=Float32, dtype_c::Type=ComplexF32)

    k_grid, k2_grid = _rfft_k_grid(dim, res, boxsize, dtype)
    Pk_interp = _interpolate_pk(pk_table, k_grid, dtype)

    fphi = if sampling_space == :real
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
    k_tab  = pk_table["k"]
    Pk_tab = pk_table["Pk"]
    # Linear interpolation on log-log scale
    log_k  = log.(k_tab)
    log_Pk = log.(max.(Pk_tab, 1e-300))
    Pk_out = similar(k_grid, dtype)
    @inbounds for i in eachindex(k_grid)
        kval = Float64(k_grid[i])
        if kval <= 0
            Pk_out[i] = dtype(0)
        else
            lk = log(kval)
            j  = searchsortedfirst(log_k, lk)
            if j <= 1
                Pk_out[i] = dtype(exp(log_Pk[1]))
            elseif j > length(log_k)
                Pk_out[i] = dtype(exp(log_Pk[end]))
            else
                t = (lk - log_k[j-1]) / (log_k[j] - log_k[j-1])
                Pk_out[i] = dtype(exp(log_Pk[j-1] * (1-t) + log_Pk[j] * t))
            end
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
