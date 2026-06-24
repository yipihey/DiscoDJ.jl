"""
Power spectrum analysis.

Implements:
- `evaluate_power_spectrum`        — isotropic P(k) from 3D field
- `evaluate_cross_power_spectrum`  — cross P(k) between two fields

Matches DISCO-DJ's `evaluate_power_spectrum` method in convention.
"""

export evaluate_power_spectrum, evaluate_cross_power_spectrum

using FFTW

"""
    evaluate_power_spectrum(field, boxsize; bins=20, logarithmic=true,
                            deconvolve=false) -> NamedTuple

Compute the isotropic power spectrum P(k) of a 3D real field.

`field`   — (res,res,res) real array (density contrast or displacement)
`boxsize` — box side length [Mpc/h]
Returns (k, Pk, N_modes) where k is bin-center wavenumber [h/Mpc].
"""
function evaluate_power_spectrum(field::AbstractArray{T,3}, boxsize::Real;
                                  bins::Int=20, logarithmic::Bool=true,
                                  deconvolve::Bool=false) where T
    res  = size(field, 1)
    size(field) == (res, res, res) || error("field must be cubic")
    V    = boxsize^3
    dk   = 2π / boxsize
    k_ny = π * res / boxsize

    # FFT
    fft_field = rfft(field)
    # Normalise: |δ(k)|² * V / N²
    fft_norm  = abs2.(fft_field) .* (V / res^6)

    # k-grid
    kfull = [i <= res÷2 ? i : i - res for i in 0:res-1] .* dk
    khalf = collect(0:res÷2) .* dk
    kx = reshape(kfull, res, 1, 1)
    ky = reshape(kfull, 1, res, 1)
    kz = reshape(khalf, 1, 1, res÷2+1)
    k_arr = sqrt.(kx.^2 .+ ky.^2 .+ kz.^2)

    # Bin edges
    k_min = dk
    k_max = k_ny * sqrt(3)
    edges = logarithmic ?
            exp10.(LinRange(log10(k_min), log10(k_max), bins+1)) :
            LinRange(k_min, k_max, bins+1)

    k_cen  = Vector{Float64}(undef, bins)
    Pk_out = Vector{Float64}(undef, bins)
    N_modes = Vector{Int}(undef, bins)

    # Weight factor: real modes count once, complex modes twice
    w = ones(Float64, res, res, res÷2+1)
    w[:, :, 1] .= 0.5; w[:, :, end] .= 0.5

    for b in 1:bins
        k_lo = edges[b]; k_hi = edges[b+1]
        mask  = (k_arr .>= k_lo) .& (k_arr .< k_hi)
        n_k   = sum(mask)
        if n_k == 0
            k_cen[b] = sqrt(k_lo * k_hi)
            Pk_out[b] = 0.0
            N_modes[b] = 0
        else
            k_cen[b]  = sum(k_arr[mask]) / n_k
            Pk_out[b]  = sum(fft_norm[mask] .* w[mask]) / sum(w[mask]) * V
            N_modes[b] = n_k
        end
    end

    return (k=k_cen, Pk=Pk_out, N_modes=N_modes)
end

"""
    evaluate_cross_power_spectrum(field1, field2, boxsize; bins=20, logarithmic=true)

Cross power spectrum P₁₂(k) = Re[<δ₁* δ₂>].
"""
function evaluate_cross_power_spectrum(field1::AbstractArray{T,3},
                                        field2::AbstractArray{T,3},
                                        boxsize::Real;
                                        bins::Int=20, logarithmic::Bool=true) where T
    res = size(field1, 1)
    V   = boxsize^3
    dk  = 2π / boxsize
    k_ny = π * res / boxsize

    f1 = rfft(field1)
    f2 = rfft(field2)
    cross = real.(conj.(f1) .* f2) .* (V / res^6)

    kfull = [i <= res÷2 ? i : i - res for i in 0:res-1] .* dk
    khalf = collect(0:res÷2) .* dk
    kx = reshape(kfull, res, 1, 1)
    ky = reshape(kfull, 1, res, 1)
    kz = reshape(khalf, 1, 1, res÷2+1)
    k_arr = sqrt.(kx.^2 .+ ky.^2 .+ kz.^2)

    k_min = dk; k_max = k_ny * sqrt(3)
    edges = logarithmic ?
            exp10.(LinRange(log10(k_min), log10(k_max), bins+1)) :
            LinRange(k_min, k_max, bins+1)

    k_cen  = Vector{Float64}(undef, bins)
    Pk12   = Vector{Float64}(undef, bins)

    for b in 1:bins
        mask = (k_arr .>= edges[b]) .& (k_arr .< edges[b+1])
        n_k  = sum(mask)
        k_cen[b] = n_k > 0 ? sum(k_arr[mask]) / n_k : sqrt(edges[b]*edges[b+1])
        Pk12[b]  = n_k > 0 ? sum(cross[mask]) / n_k * V : 0.0
    end

    return (k=k_cen, Pk12=Pk12)
end
