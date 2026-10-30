# ── GPU measurements ─────────────────────────────────────────────────────────────────────────
#
# Everything the analysis needs is a *long-wavelength* quantity, so each field is reduced on the
# device to its Fourier coefficients on a fixed set of low-k modes (|k| ≤ k_store, one member of
# every ±k pair), plus a few scalar diagnostics.  Coefficients are normalised as discrete Fourier
# series, f(x) = Σ_k f_k e^{ik·x}  (f_k = rfft(f)/N³), so a power spectrum is P(k) = L³⟨|f_k|²⟩.

export ModeSet, field_modes, vector_modes, density_modes, inverted_fraction, lattice_q

"""Fixed list of low-k modes, defined by integer wave-vectors n (k = 2πn/L) independent of the
grid size, so that fields on different particle grids (N) are compared mode by mode: |k| ≤ k_store,
DC excluded, one member of each ±k pair (n_z > 0; on n_z = 0: n_y > 0, or n_y = 0 and n_x > 0)."""
struct ModeSet{AI}
    N::Int
    L::Float64
    idx::AI                    # linear indices into this grid's (N,N,N÷2+1) rfft array
    nvec::Matrix{Int}          # (M,3) integer wave-vectors (grid independent)
    kvec::Matrix{Float64}      # (M,3) host
    kmag::Vector{Float64}
end

function ModeSet(N::Int, L::Real, k_store::Real; to_device=identity)
    nmax = floor(Int, k_store * L / 2π)
    2nmax < N || error("k_store beyond the Nyquist frequency of N = $N")
    li = LinearIndices((N, N, N ÷ 2 + 1))
    idx = Int[]; nv = NTuple{3,Int}[]
    for nz in 0:nmax, ny in -nmax:nmax, nx in -nmax:nmax
        n2 = nx^2 + ny^2 + nz^2
        (n2 == 0 || 2π / L * sqrt(n2) > k_store) && continue
        if nz == 0 && !(ny > 0 || (ny == 0 && nx > 0))
            continue
        end
        push!(idx, li[mod(nx, N) + 1, mod(ny, N) + 1, nz + 1]); push!(nv, (nx, ny, nz))
    end
    nvec = [nv[m][d] for m in eachindex(nv), d in 1:3]
    K = (2π / L) .* nvec
    return ModeSet(N, Float64(L), to_device(idx), nvec, K, vec(sqrt.(sum(abs2, K; dims=2))))
end

Base.length(ms::ModeSet) = length(ms.kmag)

"""Low-k Fourier coefficients of a real (N,N,N) field (host Vector{ComplexF64})."""
function field_modes(ms::ModeSet, f::AbstractArray{T,3}) where {T}
    fh = DJ._rfftn(f)
    v = Array(fh[ms.idx])
    return ComplexF64.(v) ./ ms.N^3
end

"""Low-k coefficients of the 3 components of a vector field (N,N,N,3) → (M,3)."""
vector_modes(ms::ModeSet, V::AbstractArray{T,4}) where {T} =
    hcat((field_modes(ms, view(V, :, :, :, c) |> copy) for c in 1:3)...)

"""Lagrangian lattice coordinates q (as in run_nbody): q_i = (i−1) L/N, as 3 broadcastable arrays."""
function lattice_q(like::AbstractArray{T}, N::Int, L::Real) where {T}
    q1 = DJ._lagr1d(N, Float64(L), T)
    dev(x) = (y = similar(like, T, size(x)); copyto!(y, x); y)
    return (dev(reshape(q1, N, 1, 1)), dev(reshape(q1, 1, N, 1)), dev(reshape(q1, 1, 1, N)))
end

"""
    deposit_delta!(mesh, X, Ψ, q) -> mesh

CIC density contrast δ = n/n̄ − 1 of the particles X = q + Ψ on the N³ mesh (one particle per cell)."""
function deposit_delta!(mesh::AbstractArray{T,3}, X, Ψ::AbstractArray{T,4}, q, L) where {T}
    N = size(mesh, 1)
    for d in 1:3
        X[:, :, :, d] .= view(Ψ, :, :, :, d) .+ q[d]
    end
    fill!(mesh, zero(T))
    DJ.pm_scatter!(mesh, reshape(X, N^3, 3), N, L, 2)
    mesh .-= one(T)
    return mesh
end

"""
    density_modes(ms, Ψ, q; mesh, X, cis=nothing) -> Vector{ComplexF64}  (or (modes, cis samples))

CIC density contrast of X = q + Ψ, low-k coefficients with the CIC window deconvolved
(W = Π sinc²(k_i Δ/2)).  With a `CiSPlan` the counts-in-spheres samples of the same field are returned too."""
function density_modes(ms::ModeSet, Ψ::AbstractArray{T,4}, q; mesh=nothing, X=nothing, cis=nothing) where {T}
    N = ms.N; L = ms.L
    mesh === nothing && (mesh = similar(Ψ, T, N, N, N))
    X === nothing && (X = similar(Ψ))
    deposit_delta!(mesh, X, Ψ, q, L)
    δh = DJ._rfftn(mesh)
    δk = ComplexF64.(Array(δh[ms.idx])) ./ N^3
    Δ = L / N
    W = [prod(sinc(ms.kvec[m, d] * Δ / 2π)^2 for d in 1:3) for m in 1:length(ms)]
    cis === nothing && return δk ./ W
    return δk ./ W, cis_sample(cis, δh)
end

# ── Counts in spheres ────────────────────────────────────────────────────────────────────────
# δ_R(x) = ∫ W_TH(|x−y|; R) δ(y) d³y, evaluated through the full-grid FFT (top-hat window
# W(kR) = 3(sin kR − kR cos kR)/(kR)³, CIC window deconvolved) at a fixed lattice of sphere centres
# (every `stride`-th grid point, identical for every member and model), together with ∇²δ_R.

struct CiSPlan{AR}
    R::Vector{Float64}
    stride::Int
    W::Vector{AR}          # per radius: W_TH(kR)/W_CIC(k) on the rfft half grid (device)
    mk2::AR                # −k² on the half grid (device)
    lowpass::Vector{AR}    # 1[|k| < k_cut] for each k_cut (channels 3… = spheres of the long-mode part only)
    kcut::Vector{Float64}
    N::Int
end

_wth(x) = x < 1e-4 ? 1 - x^2 / 10 : 3 * (sin(x) - x * cos(x)) / x^3

function CiSPlan(N::Int, L::Real, Rs; stride::Int=8, to_device=identity, T=Float32, kcut=Float64[])
    kf = DJ._kfull(N, Float64(L)); kh = DJ._khalf(N, Float64(L)); Δ = L / N
    kv(i, j, l) = (kf[i], kf[j], kh[l])
    km = [sqrt(sum(abs2, kv(i, j, l))) for i in 1:N, j in 1:N, l in 1:N÷2+1]
    wc = [prod(sinc(x * Δ / 2π)^2 for x in kv(i, j, l)) for i in 1:N, j in 1:N, l in 1:N÷2+1]
    W = [to_device(T.(_wth.(km .* R) ./ wc)) for R in Rs]
    lp = [to_device(T.(km .< kc)) for kc in kcut]
    return CiSPlan(Float64.(collect(Rs)), stride, W, to_device(T.(-(km .^ 2))), lp, Float64.(collect(kcut)), N)
end

"""(nR, 2 + n_cut, Nc) samples [δ_R, ∇²δ_R, δ_R of the part with |k| < k_cut,1, …] at the sphere centres
from an rfft field δ̂ (full half grid, DJ._rfftn layout)."""
nchan(P::CiSPlan) = 2 + length(P.kcut)
function cis_sample(P::CiSPlan, δh)
    N = P.N; s = P.stride; Nc = (N ÷ s)^3
    out = zeros(Float32, length(P.R), nchan(P), Nc)
    for (r, W) in enumerate(P.W)
        f = DJ._irfftn(δh .* W, N)
        out[r, 1, :] .= vec(Array(f[1:s:N, 1:s:N, 1:s:N]))
        f = DJ._irfftn(δh .* W .* P.mk2, N)
        out[r, 2, :] .= vec(Array(f[1:s:N, 1:s:N, 1:s:N]))
        for (c, lp) in enumerate(P.lowpass)
            f = DJ._irfftn(δh .* W .* lp, N)
            out[r, 2 + c, :] .= vec(Array(f[1:s:N, 1:s:N, 1:s:N]))
        end
    end
    return out
end

"""Linear δ_R and ∇²δ_R at D = 1 from the initial potential (δ̂ = −k²φ̂ in the faithful convention)."""
function cis_linear(P::CiSPlan, fphi)
    T = real(eltype(fphi))
    # undo the CIC deconvolution built into W: the linear field is exact, not deposited
    N = P.N; s = P.stride; Nc = (N ÷ s)^3
    out = zeros(Float32, length(P.R), nchan(P), Nc)
    δh = fphi .* P.mk2
    for (r, R) in enumerate(P.R)
        kmag = sqrt.(-P.mk2)
        W = T.(_wth.(kmag .* T(R)))
        f = DJ._irfftn(δh .* W, N)
        out[r, 1, :] .= vec(Array(f[1:s:N, 1:s:N, 1:s:N]))
        f = DJ._irfftn(δh .* W .* P.mk2, N)
        out[r, 2, :] .= vec(Array(f[1:s:N, 1:s:N, 1:s:N]))
        for (c, lp) in enumerate(P.lowpass)
            f = DJ._irfftn(δh .* W .* lp, N)
            out[r, 2 + c, :] .= vec(Array(f[1:s:N, 1:s:N, 1:s:N]))
        end
    end
    return out
end
export CiSPlan, cis_sample, cis_linear, deposit_delta!, nchan

# Kuhn decomposition of the unit cube into 6 tetrahedra along the 0→7 diagonal: for each permutation
# (a,b,c) of the axes the path 0 → e_a → e_a+e_b → (1,1,1).  The Lagrangian orientation of each tet
# is the permutation parity, so a tet is inverted iff det(edges in x) · parity < 0.
const _KUHN = ((1, 2, 3, 1), (2, 3, 1, 1), (3, 1, 2, 1), (1, 3, 2, -1), (3, 2, 1, -1), (2, 1, 3, -1))

@inline function _corner(Ψ, i, j, k, bx, by, bz, N, Δ)
    ii = mod1(i + bx, N); jj = mod1(j + by, N); kk = mod1(k + bz, N)
    @inbounds return ((i - 1 + bx) * Δ + Ψ[ii, jj, kk, 1],
                      (j - 1 + by) * Δ + Ψ[ii, jj, kk, 2],
                      (k - 1 + bz) * Δ + Ψ[ii, jj, kk, 3])
end

@kernel function _k_inverted!(cnt, @Const(Ψ), N::Int, Δ)
    i, j, k = @index(Global, NTuple)
    T = eltype(Ψ)
    x0 = _corner(Ψ, i, j, k, 0, 0, 0, N, Δ)
    x7 = _corner(Ψ, i, j, k, 1, 1, 1, N, Δ)
    n = Int32(0)
    for (a, b, c, par) in _KUHN
        e1 = (a == 1, a == 2, a == 3)
        e2 = (a == 1 || b == 1, a == 2 || b == 2, a == 3 || b == 3)
        x1 = _corner(Ψ, i, j, k, Int(e1[1]), Int(e1[2]), Int(e1[3]), N, Δ)
        x2 = _corner(Ψ, i, j, k, Int(e2[1]), Int(e2[2]), Int(e2[3]), N, Δ)
        u = (x1[1] - x0[1], x1[2] - x0[2], x1[3] - x0[3])
        v = (x2[1] - x0[1], x2[2] - x0[2], x2[3] - x0[3])
        w = (x7[1] - x0[1], x7[2] - x0[2], x7[3] - x0[3])
        det = u[1] * (v[2] * w[3] - v[3] * w[2]) - u[2] * (v[1] * w[3] - v[3] * w[1]) +
              u[3] * (v[1] * w[2] - v[2] * w[1])
        n += Int32(det * par < zero(T))
    end
    @inbounds cnt[i, j, k] = n
end

"""Fraction of the 6N³ Lagrangian Kuhn tetrahedra (equal mass) whose orientation is inverted by
the map x = q + Ψ (Ψ unwrapped) — the shell-crossed (odd stream-parity) mass fraction."""
function inverted_fraction(Ψ::AbstractArray{T,4}, L::Real; cnt=nothing) where {T}
    N = size(Ψ, 1)
    cnt === nothing && (cnt = similar(Ψ, Int32, N, N, N))
    be = get_backend(Ψ)
    _k_inverted!(be)(cnt, Ψ, N, T(L / N); ndrange=(N, N, N)); synchronize(be)
    return sum(Int64, cnt) / (6 * N^3)
end

"""
    cross_transfer(ΨA, ΨB, L, edges) -> (k, T, n)

Binned cross transfer of two vector fields (all modes of the grid, host binning):
T(k) = Σ Re(Â·B̂*) / Σ |B̂|² over the three components — e.g. PM displacement vs an LPT reference."""
function cross_transfer(ΨA::AbstractArray{T,4}, ΨB::AbstractArray{T,4}, L::Real, edges) where {T}
    N = size(ΨA, 1)
    kf = DJ._kfull(N, Float64(L)); kh = DJ._khalf(N, Float64(L))
    km = [sqrt(kf[i]^2 + kf[j]^2 + kh[l]^2) for i in 1:N, j in 1:N, l in 1:N÷2+1]
    bi = kbin_index(vec(km), edges); nb = length(edges) - 1
    ab = zeros(nb); bb = zeros(nb); n = zeros(nb); ks = zeros(nb)
    for c in 1:3
        A = Array(DJ._rfftn(copy(view(ΨA, :, :, :, c)))); B = Array(DJ._rfftn(copy(view(ΨB, :, :, :, c))))
        @inbounds for m in eachindex(A)
            b = bi[m]; b == 0 && continue
            ab[b] += real(A[m] * conj(B[m])); bb[b] += abs2(B[m]); n[b] += 1; ks[b] += vec(km)[m]
        end
    end
    return (k=ks ./ n, T=ab ./ bb, n=n ./ 3)
end
export cross_transfer
