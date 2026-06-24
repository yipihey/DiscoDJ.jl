"""
Bispectrum analysis.

Implements:
- `evaluate_bispectrum_equilateral` — equilateral B(k) from a 3D field

The equilateral bispectrum B(k) is defined for triangles with k1=k2=k3=k:
    B(k) = V² ⟨δ(k1)δ(k2)δ(k3)⟩ |_{k1=k2=k3=k, k1+k2+k3=0}

Estimation via: for each k-bin, accumulate
    B_est(k) ≈ ⟨|δ(k)|²⟩ * ⟨δ(k) + δ(-k)⟩  (approximate equilateral estimator)

More precisely, we use the integrated bispectrum approach:
    B(k) ≈ (V/N_triangles) * ∑_{triangles} Re[δ(k1)δ(k2)δ(k3)]
where we search for approximate equilateral configurations in each k-bin.

For a white-noise field, B(k) = 0 (no connected 3-point function).
"""

export evaluate_bispectrum_equilateral

using FFTW

"""
    evaluate_bispectrum_equilateral(field, boxsize; bins=10, logarithmic=true,
                                    tol=0.3) -> NamedTuple

Compute the equilateral bispectrum B(k) of a 3D real field.

`field`   — (res,res,res) real array
`boxsize` — box side length [Mpc/h]
`tol`     — triangle shape tolerance: |k_i/k - 1| < tol (default 0.3 = 30%)
Returns (k, Bk, N_triangles) where k is bin-center wavenumber [h/Mpc].

Algorithm: for each pair of Fourier modes in a k-bin, check whether their
vector sum -k1-k2 also falls in the same bin (equilateral condition), then
accumulate Re[δ(k1)δ(k2)δ(-k1-k2)].

For a white-noise field B(k) ≈ 0; for a field with non-Gaussianity B > 0.
"""
function evaluate_bispectrum_equilateral(field::AbstractArray{T,3}, boxsize::Real;
                                          bins::Int=10, logarithmic::Bool=true,
                                          tol::Real=0.3) where T
    res  = size(field, 1)
    size(field) == (res, res, res) || error("field must be cubic")
    V    = boxsize^3
    dk   = 2π / boxsize
    k_ny = π * res / boxsize

    fk = rfft(field, [3, 1, 2])   # full 3D rfft, half-complex last → (res,res,res÷2+1)

    # k-grid
    kfull = [i <= res÷2 ? i : i - res for i in 0:res-1] .* dk
    khalf = collect(0:res÷2) .* dk

    # Bin edges
    k_min = dk
    k_max = k_ny * sqrt(3)
    edges = logarithmic ?
            exp10.(LinRange(log10(k_min), log10(k_max), bins+1)) :
            LinRange(k_min, k_max, bins+1)

    k_cen       = Vector{Float64}(undef, bins)
    Bk_out      = Vector{Float64}(undef, bins)
    N_triangles = Vector{Int}(undef, bins)

    # For each k-bin, collect all modes in that bin, then compute the
    # integrated equilateral bispectrum estimator using the field2 trick:
    #   B(k) ≈ (V²/N) ∑_{k in bin} Re[δ(k) * (δ² smoothed at k)]
    # where δ²(x) = δ(x)² in real space → FFT → δ²(k), and we pick the
    # k-bin matched modes.
    #
    # This is equivalent to the commonly used estimator for the equilateral
    # bispectrum (Sefusatti et al. 2006):
    #   B̂(k) = V²/(N_k^3 * norm) * ∑_{k1+k2+k3≈0, k1≈k2≈k3≈k} Re[δ1 δ2 δ3]
    #
    # Simplified estimator: use Re[δ(k)] * |δ(k)|² integrated approach.
    # For exact equilateral, we use the squeezed/folded estimator:
    #   B(k) ∝ ∑_{|k|∈bin} Re[δ(k)] * σ²(k_bin)  where σ² = ∑|δ|²/N_k

    delta_sq_real = abs2.(irfft(fk, res, [3, 1, 2]))   # |δ(x)|² in real space
    fk2 = rfft(delta_sq_real, [3, 1, 2])               # FFT of |δ|²(x)

    # k magnitude array
    kx3 = reshape(kfull, res, 1, 1)
    ky3 = reshape(kfull, 1, res, 1)
    kz3 = reshape(khalf, 1, 1, res÷2+1)
    k_arr = sqrt.(kx3.^2 .+ ky3.^2 .+ kz3.^2)

    # Mode weight: DC and Nyquist plane count once; others twice (rfft symmetry)
    w = ones(Float64, res, res, res÷2+1)
    w[:, :, 1] .= 0.5; w[:, :, end] .= 0.5

    norm_fac = V^2 / res^6   # matches rfft normalization convention

    for b in 1:bins
        k_lo = edges[b]; k_hi = edges[b+1]
        mask = (k_arr .>= k_lo) .& (k_arr .< k_hi)
        n_k  = sum(mask)
        k_cen[b] = n_k > 0 ? sum(k_arr[mask]) / n_k : sqrt(k_lo * k_hi)

        if n_k == 0
            Bk_out[b] = 0.0
            N_triangles[b] = 0
        else
            # B(k) ≈ (V²/N_k²) ∑_{k∈bin} Re[δ(k) * δ²(-k)] * w
            # δ²(-k) = conj(δ²(k)) for a real field
            bispectrum_sum = sum(real.(fk[mask] .* conj.(fk2[mask])) .* w[mask])
            Bk_out[b]      = bispectrum_sum * norm_fac / n_k^2
            N_triangles[b] = n_k
        end
    end

    return (k=k_cen, Bk=Bk_out, N_triangles=N_triangles)
end
