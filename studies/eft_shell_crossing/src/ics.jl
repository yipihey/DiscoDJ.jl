# ── Paired initial conditions with a sharp-k split ───────────────────────────────────────────
#
# One real-space unit white-noise field per seed, ω_s(x) ~ N(0,1) (MersenneTwister, Float64), so
# its rfft is canonically Hermitian.  The initial potential follows the DISCO-DJ / DiscoDJNative
# faithful convention (`ic_operator`):  φ(k) = ω̂(k) √(P(k) (N/L)³) (−1/k²),  ψ₁ = −∇φ,  ∇·ψ₁ = −δ,
# with P(k) the linear spectrum at D = 1.
#
# For a pair (seed_L, seed_S):
#     ω̂_IR = s_L M_L ω̂_L                          M_L = 1[|k| <  k_split]
#     ω̂_UV = s_L M_L ω̂_L + A M_B ω̂_S              M_B = 1[k_split ≤ |k| < k_uv]   (k_lo ≤ |k| if set)
# so every mode with |k| < k_split is *bitwise identical* in the two members; the UV member only adds
# the band.  s_L = ±1 flips the long modes (odd/even-in-L decomposition), A rescales the band
# amplitude.  The masks depend on |k| only, so they commute with Hermitian conjugation and the
# spectra stay canonical (required by cuFFT's C2R).

export eds_cosmology, planck_pk, PairSpec, planewave_phi_hat, kmag_half, white_noise_hat, make_fphi, make_fphi_gauss, embed_fphi,
       band_sigmas

"""Einstein–de Sitter background (Ω_m = 1, D₊ = a) with Planck18 h, n_s: the nLPT growth
coefficients of `compute_core` (Dⁿ ψₙ) are then exact, and PM integrates the same equations."""
eds_cosmology() = compute_timetables(Cosmology{Float64}(Omega_c=1 - 0.0488911, Omega_b=0.0488911,
                                                        h=0.67742, sigma8=0.8105, n_s=0.96822))

"""Planck18EEBAOSN linear P(k) (Eisenstein–Hu, σ₈ = 0.8105) — used as the spectrum *shape* at D = 1."""
planck_pk() = linear_power_spectrum(Cosmology("Planck18EEBAOSN"); kmin=1e-4, kmax=100.0, n_modes=2048)

Base.@kwdef struct PairSpec
    N::Int = 256                 # particle / LPT grid
    N_ic::Int = 256              # white-noise grid defining the realization (modes embedded into N ≥ N_ic)
    L::Float64 = 500.0
    k_split::Float64 = 0.2       # h/Mpc: modes below are shared
    k_uv::Float64 = 1.2          # h/Mpc: UV member's cutoff (sharp)
    k_lo::Float64 = 0.0          # optional lower edge of the added band (band = [max(k_split,k_lo), k_uv))
    band_amp::Float64 = 1.0      # A: amplitude of the added band relative to the ΛCDM spectrum
    long_sign::Int = 1           # s_L = ±1
    seed_L::Int = 1
    seed_S::Int = 1001
    pw::Vector{NTuple{3,Int}} = NTuple{3,Int}[]   # if set: long modes replaced by plane waves at these integer
    pw_amp::Float64 = 0.0                          # wave-vectors, δ = pw_amp · cos(k·x) each at D = 1 (sign s_L)
    pw_flip::Vector{NTuple{3,Int}} = NTuple{3,Int}[]   # subset of pw whose sign is additionally flipped (group B of
                                                        # an A/B sign design: isolates the A×B coupling at k_a ± k_b)
    pw_bg::Float64 = 0.0         # with plane waves: add the seed_L long modes (|k| < k_split) scaled by pw_bg as a fixed
                                 # background (NOT flipped by s_L; its modes at ±n_j are removed)
end

"""|k| on the rfft half-grid (N,N,N÷2+1), DISCO-DJ k-vector convention."""
function kmag_half(N::Int, L::Real)
    kf = DJ._kfull(N, Float64(L)); kh = DJ._khalf(N, Float64(L))
    return [sqrt(kf[i]^2 + kf[j]^2 + kh[l]^2) for i in 1:N, j in 1:N, l in 1:N÷2+1]
end

"""rfft of the unit white noise of `seed` (Float64, [3,1,2] layout)."""
white_noise_hat(seed::Int, N::Int) = rfft(randn(MersenneTwister(seed), Float64, N, N, N), [3, 1, 2])

# (−1/k²)·√(P (N/L)³) in Float64 on the half grid (0 at DC)
function _phi_scale(N, L, pk)
    op = ic_operator(N, L, pk; T=Float64)
    return op.scale
end

"""
    make_fphi(spec, pk, which; T=Float32, cache=nothing) -> Array{Complex{T},3}

Initial potential of the `:UV` or `:IR` member of the pair (host, rfft half layout).  `cache` may be
a Dict to memoise white-noise transforms and the IC scale between calls."""
function make_fphi(s::PairSpec, pk, which::Symbol; T=Float32, cache=nothing)
    which in (:UV, :IR) || error("which must be :UV or :IR")
    s.k_uv < π * s.N_ic / s.L || s.N == s.N_ic ||
        error("embedding needs every band mode component below the N_ic Nyquist frequency (k_uv < πN_ic/L)")
    s.N == s.N_ic || return embed_fphi(make_fphi(PairSpec(s; N=s.N_ic), pk, which; T=Float64, cache=cache), s.N; T=T)
    get!(f, key) = cache === nothing ? f() : Base.get!(f, cache, key)
    kk = get!(() -> kmag_half(s.N, s.L), (:kmag, s.N, s.L))
    sc = get!(() -> _phi_scale(s.N, s.L, pk), (:scale, s.N, s.L))
    wL = get!(() -> white_noise_hat(s.seed_L, s.N), (:wn, s.seed_L, s.N))
    out = similar(wL)
    lo = max(s.k_split, s.k_lo)
    if isempty(s.pw)
        @. out = ifelse(kk < s.k_split, s.long_sign * wL, zero(eltype(wL))) * sc
    else
        all(n -> 2π / s.L * sqrt(sum(abs2, n)) < s.k_split, s.pw) || error("plane waves must lie below k_split")
        issubset(s.pw_flip, s.pw) || error("pw_flip must be a subset of pw")
        out .= s.long_sign .* get!(() -> planewave_phi_hat(s.pw, s.pw_amp, s.N, s.L), (:pw, s.pw, s.pw_amp, s.N, s.L))
        isempty(s.pw_flip) || (out .-= (2 * s.long_sign) .* planewave_phi_hat(s.pw_flip, s.pw_amp, s.N, s.L))
        if s.pw_bg != 0
            bg = @. ifelse(kk < s.k_split, s.pw_bg * wL, zero(eltype(wL))) * sc
            for n in s.pw, m in (n, .-n)                 # remove the background's own modes at the plane waves
                m[3] >= 0 && (bg[mod(m[1], s.N) + 1, mod(m[2], s.N) + 1, m[3] + 1] = 0)
            end
            out .+= bg
        end
    end
    if which === :UV && s.band_amp != 0
        wS = get!(() -> white_noise_hat(s.seed_S, s.N), (:wn, s.seed_S, s.N))
        @. out += ifelse((kk >= lo) & (kk < s.k_uv), s.band_amp * wS, zero(eltype(wS))) * sc
    end
    return Complex{T}.(out)
end

"""
    planewave_phi_hat(ns, amp, N, L) -> (N,N,N÷2+1) ComplexF64

Initial potential (same normalisation as `make_fphi`: unnormalised rfft of the real-space φ with ∇²φ = δ) of
δ(x) = amp Σ_j cos(2π n_j·x/L) at D = 1.  Built in real space so the half spectrum is exactly Hermitian; the
cosine phase makes the result independent of the sign convention of k."""
function planewave_phi_hat(ns, amp, N::Int, L::Real)
    φ = zeros(Float64, N, N, N)
    for n in ns
        k2 = (2π / L)^2 * sum(abs2, n)
        @inbounds for l in 1:N, j in 1:N, i in 1:N
            φ[i, j, l] -= amp / k2 * cospi(2 * mod(n[1] * (i - 1) + n[2] * (j - 1) + n[3] * (l - 1), N) / N)
        end
    end
    return rfft(φ, [3, 1, 2])
end

PairSpec(s::PairSpec; kw...) = PairSpec(; (f => getfield(s, f) for f in fieldnames(PairSpec))..., kw...)

"""
    embed_fphi(f, N2) -> (N2,N2,N2÷2+1)

Exact embedding of an rfft field from grid N1 into a finer grid N2 (same box): every mode keeps its
physical wave-vector and its real-space amplitude (coefficients scale by (N2/N1)³).  Requires the
N1 Nyquist planes to be empty (true for any spectrum cut at k_uv below the N1 Nyquist)."""
function embed_fphi(f::AbstractArray{Complex{S},3}, N2::Int; T=Float32) where {S}
    N1 = size(f, 1); N2 >= N1 || error("embedding needs N2 ≥ N1")
    out = zeros(Complex{T}, N2, N2, N2 ÷ 2 + 1)
    sc = (N2 / N1)^3
    sidx(m, N) = m < N ÷ 2 ? m : m - N            # signed frequency of 0-based index m
    for l in 1:N1÷2, j in 1:N1, i in 1:N1           # skip the N1 Nyquist plane l = N1/2+1
        nx = sidx(i - 1, N1); ny = sidx(j - 1, N1)
        (nx == -N1 ÷ 2 || ny == -N1 ÷ 2) && continue
        out[mod(nx, N2) + 1, mod(ny, N2) + 1, l] = Complex{T}(sc * f[i, j, l])
    end
    return out
end

"""
    make_fphi_gauss(spec, pk, R; T=Float32, cache=nothing)

Control: the PreCrossedSheet-style Gaussian-filtered member, ω̂_L(|k| < k_uv) · exp(−k²R²/2), all
modes from seed_L.  Two radii give two members whose long modes differ by the filter ratio
exp(−k²(R₁²−R₂²)/2) ≈ 1 − k²(R₁²−R₂²)/2 — an initial-condition k² 'response'."""
function make_fphi_gauss(s::PairSpec, pk, R::Real; T=Float32, cache=nothing)
    get!(f, key) = cache === nothing ? f() : Base.get!(f, cache, key)
    kk = get!(() -> kmag_half(s.N, s.L), (:kmag, s.N, s.L))
    sc = get!(() -> _phi_scale(s.N, s.L, pk), (:scale, s.N, s.L))
    wL = get!(() -> white_noise_hat(s.seed_L, s.N), (:wn, s.seed_L, s.N))
    out = @. ifelse(kk < s.k_uv, wL * exp(-kk^2 * R^2 / 2), zero(eltype(wL))) * sc
    return Complex{T}.(out)
end

"""
    band_sigmas(pk, k_lo, k_hi) -> (σ_δ, σ_ψ)

Continuum linear rms density contrast and 3-D rms displacement at D = 1 from modes
k_lo ≤ k < k_hi:  σ_δ² = ∫ k² P dk / 2π²,  σ_ψ² = ∫ P dk / 2π²."""
function band_sigmas(pk, k_lo, k_hi; n=4000)
    k = exp.(range(log(max(k_lo, 1e-5)), log(k_hi), length=n))
    P = [DJ._jinterp(log(x), log.(pk["k"]), pk["Pk"]) for x in k]
    trap(y) = sum((y[2:end] .+ y[1:end-1]) ./ 2 .* diff(k))
    return sqrt(trap(k .^ 2 .* P) / (2π^2)), sqrt(trap(P) / (2π^2))
end
