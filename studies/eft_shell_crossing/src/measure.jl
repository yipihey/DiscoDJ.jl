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

# ── Kinetic stress of the coarse-grained flow (experiment 2) ─────────────────────────────────────
# CIC moments on an N_m³ mesh: mass n, momentum p_i = Σ u_i, kinetic K_ij = Σ u_i u_j (unit masses,
# u = dx/dln a).  The velocity-dispersion stress τ_ij = K_ij − p_i p_j / n (in units of mass × u²) vanishes for
# a single cold stream sampled infinitely finely; at finite cell size Δ a smooth flow contributes
# ≈ n Δ²(∂u)²/12, which two meshes allow to remove.  Multistream cells are flagged from particles that touch
# an inverted Lagrangian tetrahedron; "boundary" cells are flagged cells with an unflagged 6-neighbour
# (the outermost caustic / splashback shell at the mesh resolution).

@kernel function _k_inverted_flags!(flag, @Const(Ψ), N::Int, Δ)
    i, j, k = @index(Global, NTuple)
    T = eltype(Ψ)
    x0 = _corner(Ψ, i, j, k, 0, 0, 0, N, Δ)
    x7 = _corner(Ψ, i, j, k, 1, 1, 1, N, Δ)
    any_inv = false
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
        any_inv |= det * par < zero(T)
    end
    if any_inv            # flag all 8 particles of the cube (race-free: every writer writes 1)
        for bz in 0:1, by in 0:1, bx in 0:1
            @inbounds flag[mod1(i + bx, N), mod1(j + by, N), mod1(k + bz, N)] = one(eltype(flag))
        end
    end
end

"""Per-particle flag (N,N,N): 1 if the particle is a vertex of a Lagrangian cube with an inverted Kuhn tet."""
function inverted_flags(Ψ::AbstractArray{T,4}, L) where {T}
    N = size(Ψ, 1); flag = similar(Ψ, T, N, N, N); fill!(flag, zero(T))
    be = get_backend(Ψ)
    _k_inverted_flags!(be)(flag, Ψ, N, T(L / N); ndrange=(N, N, N)); synchronize(be)
    return flag
end

# NGP moments per cell, positions relative to the cell centre: n, Σx(3), Σu(3), Σxx(6), Σxu(9), Σuu(6), flag
const _NMOM = 29
@kernel function _k_ngp_moments!(M, @Const(X), @Const(U), @Const(w), Nm::Int, L)
    p = @index(Global)
    @inbounds begin
        T = eltype(X); Δ = L / Nm
        g = (mod(X[p, 1], L) / Δ, mod(X[p, 2], L) / Δ, mod(X[p, 3], L) / Δ)
        c = (min(floor(Int, g[1]), Nm - 1) + 1, min(floor(Int, g[2]), Nm - 1) + 1, min(floor(Int, g[3]), Nm - 1) + 1)
        x = ((g[1] - c[1] + T(0.5)) * Δ, (g[2] - c[2] + T(0.5)) * Δ, (g[3] - c[3] + T(0.5)) * Δ)
        u = (U[p, 1], U[p, 2], U[p, 3])
        KernelAbstractions.@atomic M[c[1], c[2], c[3], 1] += one(T)
        for a in 1:3
            KernelAbstractions.@atomic M[c[1], c[2], c[3], 1 + a] += x[a]
            KernelAbstractions.@atomic M[c[1], c[2], c[3], 4 + a] += u[a]
        end
        s = 7
        for a in 1:3, b in a:3
            s += 1; KernelAbstractions.@atomic M[c[1], c[2], c[3], s] += x[a] * x[b]
        end
        for a in 1:3, b in 1:3
            s += 1; KernelAbstractions.@atomic M[c[1], c[2], c[3], s] += x[a] * u[b]
        end
        for a in 1:3, b in a:3
            s += 1; KernelAbstractions.@atomic M[c[1], c[2], c[3], s] += u[a] * u[b]
        end
        KernelAbstractions.@atomic M[c[1], c[2], c[3], 29] += w[p]
    end
end

@inline _sym(S, a, b) = S[a <= b ? (a == 1 ? b : a == 2 ? b + 2 : 6) : (b == 1 ? a : b == 2 ? a + 2 : 6)]

# per cell: raw dispersion τ⁰ = n Cov(u) and gradient-corrected τ = n [Cov(u) − Cov(u,x) Cov(x)⁻¹ Cov(x,u)]
@kernel function _k_cell_stress!(τ0, τc, @Const(M), nmin::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(M)
        n = M[i, j, k, 1]
        if n < one(T)
            for c in 1:6; τ0[i, j, k, c] = zero(T); τc[i, j, k, c] = zero(T); end
        else
            mx = (M[i, j, k, 2] / n, M[i, j, k, 3] / n, M[i, j, k, 4] / n)
            mu = (M[i, j, k, 5] / n, M[i, j, k, 6] / n, M[i, j, k, 7] / n)
            # covariances (symmetric 6-vectors ordered 11,12,13,22,23,33)
            Sxx = ntuple(Val(6)) do c
                a, b = c == 1 ? (1, 1) : c == 2 ? (1, 2) : c == 3 ? (1, 3) : c == 4 ? (2, 2) : c == 5 ? (2, 3) : (3, 3)
                M[i, j, k, 7 + c] / n - mx[a] * mx[b]
            end
            Suu = ntuple(Val(6)) do c
                a, b = c == 1 ? (1, 1) : c == 2 ? (1, 2) : c == 3 ? (1, 3) : c == 4 ? (2, 2) : c == 5 ? (2, 3) : (3, 3)
                M[i, j, k, 22 + c] / n - mu[a] * mu[b]
            end
            Sxu = ntuple(Val(9)) do c      # (a = x index, b = u index), c = 3(a−1) + b
                a = (c - 1) ÷ 3 + 1; b = (c - 1) % 3 + 1
                M[i, j, k, 13 + c] / n - mx[a] * mu[b]
            end
            for c in 1:6; τ0[i, j, k, c] = n * Suu[c]; end
            # inverse of Sxx (adjugate)
            a11, a12, a13, a22, a23, a33 = Sxx
            det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13) + a13 * (a12 * a23 - a22 * a13)
            scale = (a11 + a22 + a33)^3
            if n >= nmin && det > T(1e-6) * scale && scale > 0
                i11 = (a22 * a33 - a23 * a23) / det; i12 = -(a12 * a33 - a13 * a23) / det; i13 = (a12 * a23 - a13 * a22) / det
                i22 = (a11 * a33 - a13 * a13) / det; i23 = -(a11 * a23 - a13 * a12) / det; i33 = (a11 * a22 - a12 * a12) / det
                Ii = (i11, i12, i13, i22, i23, i33)
                for c in 1:6
                    p_, q_ = c == 1 ? (1, 1) : c == 2 ? (1, 2) : c == 3 ? (1, 3) : c == 4 ? (2, 2) : c == 5 ? (2, 3) : (3, 3)
                    # (Sux Sxx⁻¹ Sxu)_{pq} = Σ_ab Sxu[a,p] Iinv[a,b] Sxu[b,q]
                    acc = zero(T)
                    for a in 1:3, b in 1:3
                        acc += Sxu[3(a - 1) + p_] * _sym(Ii, a, b) * Sxu[3(b - 1) + q_]
                    end
                    τc[i, j, k, c] = n * (Suu[c] - acc)
                end
            else
                for c in 1:6; τc[i, j, k, c] = n * Suu[c]; end     # too few particles to fit a gradient
            end
        end
    end
end

"""
    stress_modes(ms, Ψ, U, q, L, Nm; nmin=6) -> NamedTuple

Velocity-dispersion stress of the particles (positions q + Ψ, velocities U = dx/dln a) on an Nm³ NGP mesh,
normalised by the mean number of particles per cell (τ/n̄ in units of u²):
  modes[1, :, c]  raw τ⁰ = n Cov(u)                               (includes the smooth-flow gradient within cells)
  modes[2, :, c]  corrected τ = n [Cov(u) − Cov(u,x)Cov(x)⁻¹Cov(x,u)] (linear smooth flow removed per cell)
  modes[3, :, c]  corrected τ in multistream cells (cells containing a particle of a cube with an inverted tet)
  modes[4, :, c]  corrected τ in multistream boundary cells (flagged cells with an unflagged 6-neighbour)
c = 11, 12, 13, 22, 23, 33.  Also mass / volume fractions of multistream and boundary cells and the
mass-weighted mean of tr τ/n in them."""
function stress_modes(ms::ModeSet, Ψ::AbstractArray{T,4}, U::AbstractArray{T,4}, q, L, Nm::Int; nmin::Int=6) where {T}
    N = size(Ψ, 1); np = N^3; nbar = T(np / Nm^3)
    X = similar(Ψ)
    for d in 1:3
        X[:, :, :, d] .= view(Ψ, :, :, :, d) .+ q[d]
    end
    flag = inverted_flags(Ψ, L)
    Mo = similar(Ψ, T, Nm, Nm, Nm, _NMOM); fill!(Mo, zero(T))
    be = get_backend(Ψ)
    _k_ngp_moments!(be)(Mo, reshape(X, np, 3), reshape(U, np, 3), reshape(flag, np), Nm, T(L); ndrange=np)
    synchronize(be)
    τ0 = similar(Ψ, T, Nm, Nm, Nm, 6); τc = similar(τ0)
    _k_cell_stress!(be)(τ0, τc, Mo, nmin; ndrange=(Nm, Nm, Nm)); synchronize(be)
    n = view(Mo, :, :, :, 1)
    msk = T.(view(Mo, :, :, :, 29) .> 0)
    nb = msk .* T.((circshift(msk, (1, 0, 0)) .* circshift(msk, (-1, 0, 0)) .* circshift(msk, (0, 1, 0)) .*
                    circshift(msk, (0, -1, 0)) .* circshift(msk, (0, 0, 1)) .* circshift(msk, (0, 0, -1))) .< T(0.5))
    out = zeros(ComplexF64, 4, length(ms), 6)
    for c in 1:6
        t0 = view(τ0, :, :, :, c) ./ nbar; tc = view(τc, :, :, :, c) ./ nbar
        out[1, :, c] .= field_modes(ms, copy(t0)); out[2, :, c] .= field_modes(ms, copy(tc))
        out[3, :, c] .= field_modes(ms, tc .* msk); out[4, :, c] .= field_modes(ms, tc .* nb)
    end
    trc = (view(τc, :, :, :, 1) .+ view(τc, :, :, :, 4) .+ view(τc, :, :, :, 6))
    tr0 = (view(τ0, :, :, :, 1) .+ view(τ0, :, :, :, 4) .+ view(τ0, :, :, :, 6))
    nms = sum(n .* msk)
    return (modes=out, mass_frac=nms / np, vol_frac=sum(msk) / Nm^3,
            sigma2_ms=nms > 0 ? sum(trc .* msk) / nms : zero(T),          # mass-weighted ⟨tr σ²⟩ in multistream cells
            bnd_vol_frac=sum(nb) / Nm^3, mean_trace=sum(trc) / np,         # mass-weighted ⟨tr σ²⟩ over all mass
            mean_trace_raw=sum(tr0) / np)
end
export inverted_flags, stress_modes

# ── Full-range power spectra (small-scale convergence) ────────────────────────────────────────────
# Binned on the host from |f̂|² of the full rfft grid (half-space weights 1 on the self-conjugate planes,
# 2 elsewhere).  Density: CIC deposit on a common Nm³ mesh (window deconvolved), so runs with different
# particle numbers are binned identically.  Displacement: Σ_c |Ψ̂_c|² on the particle grid.

struct PkPlan
    L::Float64; Nm::Int; edges::Vector{Float64}
    bin_m::Array{Int32,3}; w_m::Array{Float32,3}; W2_m::Array{Float32,3}       # deposit mesh (Nm)
    bins_p::Dict{Int,Tuple{Array{Int32,3},Array{Float32,3}}}                    # particle grids (N)
end

function _pk_bins(N, L, edges)
    kf = DJ._kfull(N, Float64(L)); kh = DJ._khalf(N, Float64(L)); Δ = L / N
    b = zeros(Int32, N, N, N ÷ 2 + 1); w = zeros(Float32, N, N, N ÷ 2 + 1); W2 = similar(w)
    for l in 1:N÷2+1, j in 1:N, i in 1:N
        k = sqrt(kf[i]^2 + kf[j]^2 + kh[l]^2)
        bb = searchsortedlast(edges, k); b[i, j, l] = (k > 0 && 1 <= bb < length(edges)) ? bb : 0
        w[i, j, l] = (l == 1 || l == N ÷ 2 + 1) ? 1 : 2
        W2[i, j, l] = prod(sinc(x * Δ / 2π)^2 for x in (kf[i], kf[j], kh[l]))^2
    end
    return b, w, W2
end

function PkPlan(L, Nm; nbins=40, kmax=π * Nm / L)
    edges = collect(exp.(range(log(2π / L * 0.999), log(kmax), length=nbins + 1)))
    b, w, W2 = _pk_bins(Nm, L, edges)
    return PkPlan(Float64(L), Nm, edges, b, w, W2, Dict{Int,Tuple{Array{Int32,3},Array{Float32,3}}}())
end

function _bin!(acc, cnt, v, b, w)
    @inbounds for i in eachindex(v)
        bb = b[i]; bb == 0 && continue
        acc[bb] += w[i] * v[i]; cnt[bb] += w[i]
    end
end

"""(k, P_δ) of the CIC density of X = q + Ψ on the plan's deposit mesh (window deconvolved, no shot-noise
subtraction), and (k, P_Ψ) of the displacement on the particle grid; P = L³ ⟨|f_k|²⟩."""
function pk_full(P::PkPlan, Ψ::AbstractArray{T,4}, q) where {T}
    N = size(Ψ, 1); Nm = P.Nm; L = P.L; nb = length(P.edges) - 1
    X = similar(Ψ); mesh = similar(Ψ, T, Nm, Nm, Nm)
    for d in 1:3; X[:, :, :, d] .= view(Ψ, :, :, :, d) .+ q[d]; end
    fill!(mesh, zero(T)); DJ.pm_scatter!(mesh, reshape(X, N^3, 3), Nm, L, 2)
    mesh .= mesh .* T(Nm^3 / N^3) .- one(T)
    v = Array(abs2.(DJ._rfftn(mesh))) ./ P.W2_m
    acc = zeros(nb); cnt = zeros(nb); _bin!(acc, cnt, v, P.bin_m, P.w_m)
    Pd = L^3 .* acc ./ cnt ./ Float64(Nm)^6
    bp, wp = get!(() -> (bb = _pk_bins(N, L, P.edges); (bb[1], bb[2])), P.bins_p, N)
    acc2 = zeros(nb); cnt2 = zeros(nb)
    for c in 1:3
        _bin!(acc2, cnt2, Array(abs2.(DJ._rfftn(copy(view(Ψ, :, :, :, c))))), bp, wp)
    end
    Pp = L^3 .* acc2 ./ (cnt2 ./ 3) ./ Float64(N)^6
    kc = sqrt.(P.edges[1:end-1] .* P.edges[2:end])
    return kc, Pd, Pp
end
export PkPlan, pk_full

# ── exact (window- and alias-free) density modes: direct Fourier sum over the particles ────────────
#
# δ_k = (1/N_p) Σ_p e^{−ik·x_p}  (k ≠ 0), the same normalisation as `field_modes`.  The CIC deposit of a near-lattice
# particle load carries a coherent, time-dependent k² bias at low k (≈ −0.18 k² h⁻²Mpc² in R at 256³/500 h⁻¹Mpc), which
# is harmless when two particle sets are compared with the same deposit (PM vs LPT) but not against analytic PT.

export ExactModes, exact_density_modes

struct ExactModes{A}
    nvec::Matrix{Int}          # (M,3) integer wave-vectors (ModeSet order for k ≤ k_exact)
    nv::A                      # device copy (Int32)
    kmag::Vector{Float64}
end
function ExactModes(N::Int, L::Real, k_exact::Real; to_device=identity)
    ms = ModeSet(N, L, k_exact)
    return ExactModes(ms.nvec, to_device(Int32.(ms.nvec)), ms.kmag)
end
Base.length(em::ExactModes) = length(em.kmag)

@kernel function _k_dft_partial!(Pr, Pi, @Const(U), @Const(nv), Np::Int, chunk::Int)
    m, c = @index(Global, NTuple)
    nx = nv[m, 1]; ny = nv[m, 2]; nz = nv[m, 3]
    sr = 0.0f0; si = 0.0f0
    i0 = (c - 1) * chunk + 1; i1 = min(c * chunk, Np)
    for i in i0:i1
        t = nx * U[i, 1] + ny * U[i, 2] + nz * U[i, 3]          # k·x / 2π in cycles (U = x/L)
        t -= round(t)
        s, co = sincospi(2t)
        sr += co; si -= s
    end
    @inbounds Pr[m, c] = sr; @inbounds Pi[m, c] = si
end

"""
    exact_density_modes(em, Ψ, q, L; chunk=65536) -> Vector{ComplexF64}

Density contrast of the particles X = q + Ψ on the modes of `em` by a direct sum (no mesh, no window, no aliasing)."""
function exact_density_modes(em::ExactModes, Ψ::AbstractArray{T,4}, q, L::Real; chunk::Int=65536) where {T}
    N = size(Ψ, 1); Np = N^3; M = length(em)
    U = similar(Ψ, Float32, N, N, N, 3)
    for d in 1:3
        U[:, :, :, d] .= Float32.((view(Ψ, :, :, :, d) .+ q[d]) ./ T(L))
    end
    U2 = reshape(U, Np, 3)
    nc = cld(Np, chunk)
    Pr = similar(U, Float32, M, nc); Pi = similar(Pr)
    be = get_backend(U)
    _k_dft_partial!(be, (64, 1))(Pr, Pi, U2, em.nv, Np, chunk; ndrange=(M, nc))
    KernelAbstractions.synchronize(be)
    sr = vec(sum(Float64.(Pr); dims=2)); si = vec(sum(Float64.(Pi); dims=2))
    return ComplexF64.(Array(sr), Array(si)) ./ Np
end
