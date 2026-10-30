# ── Response and stochasticity estimators (host, mode level) ───────────────────────────────
#
# Conventions (all lengths Mpc/h, k in h/Mpc):
#   ψ_L(k)   = D ψ̂₁(k), the linear displacement — identical in both members for |k| < k_split.
#   Δψ(k)    = ψ̂_UV(k) − ψ̂_IR(k).
#   Longitudinal projection  ψ_∥ = k̂·ψ  (the density-carrying part, δ = −i k ψ_∥ at linear order).
#   Response R(k) = ⟨Re Δψ_∥ ψ_L∥*⟩ / ⟨|ψ_L∥|²⟩  (mode-summed in |k| bins).
#   EFT form R(k) = −C₂ k² + C₄ k⁴, i.e. Δψ_L = −C₂ k² ψ_L + …   (C₂ in (Mpc/h)²).
#   With δ = −i k·ψ the same R multiplies δ_L:  Δδ = R δ_L,  ΔP/P = 2R = −2C₂k² + …

export longitudinal, transverse, kbin_index, BinAcc, accum!, response, fit_counterterm,
       kmax_sweep, binned_power, jackknife

longitudinal(ψ::AbstractMatrix, kvec::AbstractMatrix) =
    vec(sum(ψ .* kvec; dims=2)) ./ vec(sqrt.(sum(abs2, kvec; dims=2)))

function transverse(ψ::AbstractMatrix, kvec::AbstractMatrix)
    kh = kvec ./ sqrt.(sum(abs2, kvec; dims=2))
    return ψ .- kh .* sum(ψ .* kh; dims=2)
end

"""Bin index (1..nb, 0 = outside) of each mode for bin `edges`."""
kbin_index(kmag::AbstractVector, edges::AbstractVector) =
    [(b = searchsortedlast(edges, k); (1 <= b < length(edges)) ? b : 0) for k in kmag]

"""Per-bin accumulators of Σ Re(y x*), Σ|x|², Σ|y|², Σ k²|x|², Σ k⁴|x|², mode count."""
struct BinAcc
    yx::Vector{Float64}; xx::Vector{Float64}; yy::Vector{Float64}
    k2xx::Vector{Float64}; k4xx::Vector{Float64}; n::Vector{Float64}
end
BinAcc(nb::Int) = BinAcc(zeros(nb), zeros(nb), zeros(nb), zeros(nb), zeros(nb), zeros(nb))
Base.:+(a::BinAcc, b::BinAcc) = BinAcc(a.yx .+ b.yx, a.xx .+ b.xx, a.yy .+ b.yy, a.k2xx .+ b.k2xx,
                                       a.k4xx .+ b.k4xx, a.n .+ b.n)

function accum!(A::BinAcc, y::AbstractVector, x::AbstractVector, kmag::AbstractVector, bi::AbstractVector)
    @inbounds for m in eachindex(y)
        b = bi[m]; b == 0 && continue
        ax = abs2(x[m]); k2 = kmag[m]^2
        A.yx[b] += real(y[m] * conj(x[m])); A.xx[b] += ax; A.yy[b] += abs2(y[m])
        A.k2xx[b] += k2 * ax; A.k4xx[b] += k2^2 * ax; A.n[b] += 1
    end
    return A
end
accum!(y, x, kmag, edges::AbstractVector{<:Real}) =
    accum!(BinAcc(length(edges) - 1), y, x, kmag, kbin_index(kmag, edges))

"""Binned response R_b = Σ Re(y x*)/Σ|x|², and the |x|²-weighted ⟨k²⟩_b, ⟨k⁴⟩_b."""
response(A::BinAcc) = (R=A.yx ./ A.xx, k2=A.k2xx ./ A.xx, k4=A.k4xx ./ A.xx, n=A.n)

"""
    fit_counterterm(k2, k4, R, σ; kmax, order=1) -> (C2, C4, σC2, σC4, χ², dof)

Weighted least squares of R_b = −C₂⟨k²⟩_b (+ C₄⟨k⁴⟩_b for order = 2) over bins with ⟨k²⟩ ≤ kmax²."""
function fit_counterterm(k2, k4, R, σ; kmax::Real, order::Int=1)
    sel = findall(i -> isfinite(R[i]) && isfinite(σ[i]) && σ[i] > 0 && k2[i] <= kmax^2, eachindex(R))
    length(sel) < order + 1 && return (C2=NaN, C4=NaN, σC2=NaN, σC4=NaN, chi2=NaN, dof=0)
    A = order == 1 ? reshape(-k2[sel], :, 1) : hcat(-k2[sel], k4[sel])
    W = Diagonal(1 ./ σ[sel] .^ 2)
    F = A' * W * A
    p = F \ (A' * W * R[sel]); cov = inv(F)
    r = R[sel] .- A * p
    chi2 = sum(abs2, r ./ σ[sel])
    return order == 1 ? (C2=p[1], C4=0.0, σC2=sqrt(cov[1, 1]), σC4=0.0, chi2=chi2, dof=length(sel) - 1) :
                        (C2=p[1], C4=p[2], σC2=sqrt(cov[1, 1]), σC4=sqrt(cov[2, 2]), chi2=chi2, dof=length(sel) - 2)
end

"""C₂ (and C₄) as a function of the maximum fitted wavenumber."""
kmax_sweep(k2, k4, R, σ, kmaxs; order=1) = [fit_counterterm(k2, k4, R, σ; kmax=km, order=order) for km in kmaxs]

"""Binned power L³⟨|f|²⟩ of mode coefficients `f` (f_k = rfft/N³)."""
function binned_power(f::AbstractVector, kmag, edges, L)
    bi = kbin_index(kmag, edges); nb = length(edges) - 1
    s = zeros(nb); n = zeros(nb); ks = zeros(nb)
    for m in eachindex(f)
        b = bi[m]; b == 0 && continue
        s[b] += abs2(f[m]); n[b] += 1; ks[b] += kmag[m]
    end
    return (k=ks ./ n, P=L^3 .* s ./ n, n=n)
end

"""
    jackknife(f, items) -> (mean, σ)

Delete-one jackknife of a statistic `f(subset)` over a vector of independent items (e.g. per-seed
`BinAcc`s).  `f` returns a number or an array."""
function jackknife(f, items::AbstractVector)
    n = length(items)
    full = f(items)
    n < 2 && return (full, fill(NaN, size(full)))
    th = [f(items[setdiff(1:n, i)]) for i in 1:n]
    tbar = sum(th) / n
    σ = sqrt.((n - 1) / n .* sum(t -> (t .- tbar) .^ 2, th))
    return (full, σ)
end
