# ── matched-phase white noise ────────────────────────────────────────────────
#
# A realization is defined by (seed, N_fine).  The master white-noise field is a unit-
# variance real Gaussian field on the N_fine³ grid; its rfft `w(k)` is
#   * Gaussian ICs:        used as is;
#   * fixed-amplitude ICs: w(k) ← √(N_fine³) · w(k)/|w(k)|  (Angulo & Pontzen 2016 style:
#                          every mode gets exactly |δ_k|² = P(k), only the phases are random;
#                          self-conjugate modes become ±√(N³), preserving Hermitian symmetry).
# For a coarser grid N < N_fine the modes with |k_i| < k_Nyq(N) are kept (Nyquist planes of
# the coarse grid zeroed) and rescaled by (N/N_fine)^{3/2}, so that the *same* large-scale
# modes are realised at every resolution — resolution convergence becomes a same-phase
# difference.  Paired fields (−δ₀) are realised downstream by ε → −ε.

"""
    white_noise(seed, N; N_fine=N, fixed=false) -> Array{Float32,3}

Real-space white-noise field on the N³ grid with the matched-phase construction above.
"""
function white_noise(seed::Integer, N::Int; N_fine::Int = N, fixed::Bool = false)
    @assert N <= N_fine && N_fine % N == 0 "N must divide N_fine"
    rng = MersenneTwister(hash((:mass_cdf_white_noise, seed, N_fine)))
    w = randn(rng, Float64, N_fine, N_fine, N_fine)
    wk = rfft(w)
    if fixed
        amp = sqrt(Float64(N_fine)^3)
        @inbounds for i in eachindex(wk)
            a = abs(wk[i])
            wk[i] = a > 0 ? wk[i] * (amp / a) : zero(wk[i])
        end
        wk[1, 1, 1] = 0
    end
    if N == N_fine
        out = irfft(wk, N_fine)
        return Array{Float32,3}(out)
    end
    # truncate to the coarse grid (indices: kx,ky ∈ {0..N/2-1} ∪ {-(N/2-1)..-1}, kz ∈ 0..N/2-1)
    h = N ÷ 2
    sub = zeros(ComplexF64, h + 1, N, N)
    keep = vcat(1:h, (N_fine - h + 2):N_fine)          # positive then negative frequencies (no Nyquist)
    dest = vcat(1:h, (h + 2):N)
    sub[1:h, dest, dest] .= wk[1:h, keep, keep]
    sub .*= (N / N_fine)^1.5
    return Array{Float32,3}(irfft(sub, N))
end

"Linear density at D=1 for white noise `w` installed in `dj`: convenience wrapper."
function delta_ini_from_white_noise(dj, w::Array{Float32,3})
    return delta_ini(with_white_noise(dj, w))
end
