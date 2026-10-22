# Multiscale (cube / 3x3x3-parent) locality diagnostic.  Included by multiscale/bin/*.jl after `using MassCDF`.
# Layout convention: every 3D array here is indexed [i,j,k] = (x,y,z) lattice indices (numpy logical order);
# DISCO-DJ / MassCDF flat C-order vectors (e = (i*N+j)*N+k) map to it via `to_xyz`.
module MS

using LinearAlgebra, Statistics, StaticArrays, FFTW, Printf, Random
import ..MassCDF
const M = MassCDF

const COMPS = ((1,1),(2,2),(3,3),(1,2),(1,3),(2,3))     # xx yy zz xy xz yz

# ---------------------------------------------------------------- layout helpers
to_xyz(v::AbstractVector, N) = permutedims(reshape(v, N, N, N), (3, 2, 1))
from_xyz(a::AbstractArray{<:Any,3}) = vec(permutedims(a, (3, 2, 1)))

"6 independent components (xx,yy,zz,xy,xz,yz) of a symmetrised tensor field, as Float64 [i,j,k] arrays."
function tensor6(G::Vector{<:SMatrix{3,3}}, N)
    out = [Array{Float64,3}(undef, N, N, N) for _ in 1:6]
    flat = [Vector{Float64}(undef, length(G)) for _ in 1:6]
    Threads.@threads for e in eachindex(G)
        g = G[e]
        @inbounds for (c, (a, b)) in enumerate(COMPS)
            flat[c][e] = 0.5 * (Float64(g[a, b]) + Float64(g[b, a]))
        end
    end
    return [to_xyz(f, N) |> collect for f in flat]
end

# ---------------------------------------------------------------- Fourier cube filtering
sincx(x) = x == 0 ? 1.0 : sin(x) / x

"integer FFT index -> wavenumber (fftfreq order) for an axis of N points in a box L"
kaxis(N, L) = [2π / L * (i <= N ÷ 2 ? i : i - N) for i in 0:N-1]
kaxis_r(N, L) = [2π / L * i for i in 0:N÷2]

"""
    cube_filter(F, N, L, width_cubes) -> Array

Exact top-hat average of the (band-limited, Nyquist-zeroed) periodic field `F` over the cubic window of side
`width_cubes*h` centred on the centre (i+1/2,j+1/2,k+1/2)h of lattice cube (i,j,k): multiply by
prod_i sinc(k_i w h/2) and by the phase e^{i k.h/2}.  Returns Float64 [i,j,k].
"""
# rfft halves the FIRST array dimension (= x here).  Use dedicated routine with that in mind.
function cube_filter(F::AbstractArray{Float64,3}, N::Int, L::Real, width_cubes::Real)
    h = L / N
    Fk = rfft(F)                       # size (N/2+1, N, N); dim 1 = x (halved)
    kxh = kaxis_r(N, L); ky = kaxis(N, L)
    w1h = [sincx(k * width_cubes * h / 2) * cis(k * h / 2) for k in kxh]
    w1f = [sincx(k * width_cubes * h / 2) * cis(k * h / 2) for k in ky]
    nyq = N ÷ 2
    @inbounds for k3 in 1:N, k2 in 1:N, k1 in 1:N÷2+1
        if (k1 - 1) == nyq || (k2 - 1) == nyq || (k3 - 1) == nyq
            Fk[k1, k2, k3] = 0
        else
            Fk[k1, k2, k3] *= w1h[k1] * w1f[k2] * w1f[k3]
        end
    end
    return irfft(Fk, N)
end

"fraction of the power of F sitting on the Nyquist planes (should be ~0 for band-limited gradients)"
function nyquist_fraction(F::AbstractArray{Float64,3}, N)
    Fk = rfft(F); nyq = N ÷ 2; tot = 0.0; ny = 0.0
    @inbounds for k3 in 1:N, k2 in 1:N, k1 in 1:N÷2+1
        w = (k1 == 1 || k1 == N ÷ 2 + 1) ? 1.0 : 2.0
        p = w * abs2(Fk[k1, k2, k3]); tot += p
        ((k1 - 1) == nyq || (k2 - 1) == nyq || (k3 - 1) == nyq) && (ny += p)
    end
    return ny / tot
end

"Cube-averaged (width 1) and 3x3x3-parent-averaged (width 3) 6-component fields from vertex tensor components."
function cube_and_parent(A6, N, L)
    AC = [cube_filter(a, N, L, 1) for a in A6]
    AP = [cube_filter(a, N, L, 3) for a in A6]
    return AC, AP
end

"periodic 27-cube average of a field (check of A_P)"
function avg27(a::AbstractArray{Float64,3})
    out = zeros(size(a))
    for dx in -1:1, dy in -1:1, dz in -1:1
        out .+= circshift(a, (dx, dy, dz))
    end
    return out ./ 27
end

# ---------------------------------------------------------------- FFT gradient of a displacement (Julia version of DISCO-DJ's ik kernel, Nyquist zeroed per axis)
function gradient_from_psi(psi::Matrix{Float32}, N, L)
    comps = [rfft(Float64.(to_xyz(psi[:, b], N))) for b in 1:3]       # ψ_b(k), dim1 = x halved
    kh = kaxis_r(N, L); kf = kaxis(N, L); nyq = N ÷ 2
    kvec(a, i1, i2, i3) = a == 1 ? (i1 - 1 == nyq ? 0.0 : kh[i1]) : a == 2 ? (i2 - 1 == nyq ? 0.0 : kf[i2]) : (i3 - 1 == nyq ? 0.0 : kf[i3])
    G = Vector{Array{Float64,3}}()
    out = Dict{Tuple{Int,Int},Array{Float64,3}}()
    for a in 1:3, b in 1:3
        Fk = similar(comps[b])
        @inbounds for i3 in 1:N, i2 in 1:N, i1 in 1:N÷2+1
            Fk[i1, i2, i3] = im * kvec(a, i1, i2, i3) * comps[b][i1, i2, i3]
        end
        out[(a, b)] = irfft(Fk, N)
    end
    return out     # out[(a,b)] = ∂_a ψ_b
end

function sym6_from_grad(g)
    [0.5 * (g[(a, b)] .+ g[(b, a)]) for (a, b) in COMPS]
end
function asym_fraction(g)
    num = 0.0; den = 0.0
    for a in 1:3, b in 1:3
        num += sum(abs2, g[(a, b)] .- g[(b, a)]); den += sum(abs2, g[(a, b)])
    end
    return num / den
end

# ---------------------------------------------------------------- Gaussian prediction
"U = ∫P W_aW_b k̂x⁴ , V = ∫P W_aW_b k̂x²k̂y² over the rfft grid of the (L,N) box (Nyquist planes excluded, k=0 excluded), /L³.
 `wa`,`wb` are cube widths in units of h (1 = cube C, 3 = parent P).  Axis-symmetrised."
function box_UV(kt, Pt, N, L, wa, wb)
    h = L / N; kf = 2π / L
    kx = kf .* [i <= N ÷ 2 ? i : i - N for i in 0:N-1]
    kz = kf .* collect(0:N÷2)
    sA = [sincx(k * wa * h / 2) for k in kx]; sB = [sincx(k * wb * h / 2) for k in kx]
    szA = [sincx(k * wa * h / 2) for k in kz]; szB = [sincx(k * wb * h / 2) for k in kz]
    Us = zeros(N); Vs = zeros(N)
    Threads.@threads for iy in 1:N
        U = 0.0; V = 0.0
        @inbounds for iz in eachindex(kz)
            (iz - 1 == N ÷ 2) && continue
            wz = (iz == 1) ? 1.0 : 2.0
            for ix in 1:N
                (ix - 1 == N ÷ 2 || iy - 1 == N ÷ 2) && continue
                kx_, ky_, kz_ = kx[ix], kx[iy], kz[iz]
                k2 = kx_^2 + ky_^2 + kz_^2
                k2 == 0 && continue
                P = M.linterp(kt, Pt, sqrt(k2))
                W = sA[ix] * sA[iy] * szA[iz] * sB[ix] * sB[iy] * szB[iz]
                c = wz * P * W / k2^2
                x2, y2, z2 = kx_^2, ky_^2, kz_^2
                U += c * (x2^2 + y2^2 + z2^2) / 3
                V += c * (x2 * y2 + x2 * z2 + y2 * z2) / 3
            end
        end
        Us[iy] = U; Vs[iy] = V
    end
    return sum(Us) / L^3, sum(Vs) / L^3
end

"gauss-legendre nodes/weights on [a,b]"
function gl(n, a, b)
    x, w = gausslegendre_nodes(n)
    return (b - a) / 2 .* x .+ (a + b) / 2, (b - a) / 2 .* w
end
function gausslegendre_nodes(n)
    # Golub-Welsch
    β = [i / sqrt(4i^2 - 1) for i in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

"Continuum U,V = ∫d³k/(2π)³ P W_aW_b k̂x⁴ (k̂x²k̂y²) via spherical quadrature over the first octant (×8)."
function continuum_UV(kt, Pt, h, wa, wb; nang = 40, nrad = 3000, kmin = 1e-4, kmax = 1.0)
    mu, wmu = gl(nang, 0.0, 1.0)             # cosθ ∈ [0,1]
    ph, wph = gl(nang, 0.0, π / 2)
    # angular tables
    dirs = Vector{NTuple{3,Float64}}(); wang = Float64[]
    for (m, wm) in zip(mu, wmu), (p, wp) in zip(ph, wph)
        s = sqrt(1 - m^2); push!(dirs, (s * cos(p), s * sin(p), m)); push!(wang, wm * wp)
    end
    lnk = range(log(kmin), log(kmax), length = nrad); dl = step(lnk)
    U = 0.0; V = 0.0
    for (ii, lk) in enumerate(lnk)
        k = exp(lk); P = M.linterp(kt, Pt, k)
        P <= 0 && continue
        a = 0.0; b = 0.0
        for (d, wa_) in zip(dirs, wang)
            W = prod(sincx(k * d[i] * wa * h / 2) * sincx(k * d[i] * wb * h / 2) for i in 1:3)
            x2, y2, z2 = d[1]^2, d[2]^2, d[3]^2
            a += wa_ * W * (x2^2 + y2^2 + z2^2) / 3
            b += wa_ * W * (x2 * y2 + x2 * z2 + y2 * z2) / 3
        end
        tw = (ii == 1 || ii == nrad) ? 0.5 : 1.0
        f = tw * dl * k^3 * P / (2π)^3 * 8
        U += f * a; V += f * b
    end
    return U, V
end

"Sector variances (S0,SE,ST) from U,V:  S0=U+2V (trace), SE=U−V (diag traceless), ST=2V (off-diagonal)."
sectors(U, V) = (S0 = U + 2V, SE = U - V, ST = 2V)

function predict_sectors(kt, Pt, N, L; method = :box, kw...)
    h = L / N
    f = (wa, wb) -> method == :box ? box_UV(kt, Pt, N, L, wa, wb) : continuum_UV(kt, Pt, h, wa, wb; kw...)
    UVcc = f(1, 1); UVpp = f(3, 3); UVcp = f(1, 3)
    return (CC = sectors(UVcc...), PP = sectors(UVpp...), CP = sectors(UVcp...))
end

"derived quantities from predict_sectors"
function ratios(S)
    names = (:S0, :SE, :ST)
    r = map(s -> S.CP[s] / sqrt(S.PP[s] * S.CC[s]), names)
    β = map(s -> S.CP[s] / S.PP[s], names)
    return (r = r, f = map(x -> 1 - x^2, r), β = β)
end

# ---------------------------------------------------------------- measured sector second moments (6-component fields, [i,j,k] arrays or flat vectors)
"Second moments of the sector amplitudes for pairs of fields X,Y (each a vector of 6 component arrays). Returns (S0,SE,ST)."
function sector_moment(X, Y)
    xx, yy, zz, xy, xz, yz = 1, 2, 3, 4, 5, 6
    t(Z) = (Z[xx] .+ Z[yy] .+ Z[zz]) ./ sqrt(3)
    e1(Z) = (Z[xx] .- Z[yy]) ./ sqrt(2)
    e2(Z) = (2 .* Z[zz] .- Z[xx] .- Z[yy]) ./ sqrt(6)
    S0 = mean(t(X) .* t(Y))
    SE = 0.5 * (mean(e1(X) .* e1(Y)) + mean(e2(X) .* e2(Y)))
    ST = (2 / 3) * (mean(X[xy] .* Y[xy]) + mean(X[xz] .* Y[xz]) + mean(X[yz] .* Y[yz]))
    return (S0 = S0, SE = SE, ST = ST)
end

# ---------------------------------------------------------------- Jacobian of cubes from positions
"J_cube [i,j,k] (Float32) from flattened positions ψ_total (N³,3 Julia Float32 matrix, C order) via Kuhn cubes."
function cube_jacobian(psi_total::Matrix{Float32}, N, L; q_py = nothing)
    pss = M.pyimport("discodj.analysis.phase_space_sheet")
    q_py === nothing && (q_py = M.grid_positions_py(N, L))
    ψ = M.jnp.asarray(M.np.asarray(psi_total))
    ψ = ψ - L * M.jnp.round(ψ / L)
    x = (q_py + ψ).reshape(N, N, N, 3)
    V = pss.kuhn_cube_volumes_from_positions(x, Float64(L); stride = 1, overlapping = true)
    h = L / N
    return M.pyconvert(Array{Float32,3}, M.np.asarray(V)) .* Float32(1 / h^3)
end

"unwrapped J from FEM-style 6N³ tet volumes (ordering of kuhn_connectivity: 6 consecutive tets per cube, cubes in C order) -> cube J [i,j,k]"
function cube_J_from_tets(Jtet::Vector{Float32}, N)
    s = reshape(Jtet, 6, :)
    Jc = vec(sum(s; dims = 1)) ./ 6
    return to_xyz(Jc, N) |> collect
end

# ---------------------------------------------------------------- invariants basis
const SM3d = SMatrix{3,3,Float64,9}
@inline function mat(c::AbstractVector, ε)       # c: 6 components -> symmetric 3x3 scaled by ε
    SM3d(ε*c[1], ε*c[4], ε*c[5], ε*c[4], ε*c[2], ε*c[6], ε*c[5], ε*c[6], ε*c[3])
end
@inline invs(A::SM3d) = (tr(A), 0.5 * (tr(A)^2 - tr(A * A)), det(A))

const BASIS_ONE = ["1", "I1C", "I2C", "I3C", "I1C^2", "I1C*I2C", "I1C^3", "I1C*I3C", "I2C^2", "I1C^2*I2C", "I1C^4"]
const BASIS_EXTRA = ["I1P", "I2P", "I3P", "tr(AC AP)", "tr(AC^2 AP)", "tr(AC AP^2)", "I1C*I1P", "I1C*I2P", "I2C*I1P", "I1P^2",
                     "I1P^3", "I1C*I1P^2", "I1C^2*I1P", "tr(AC AP)*I1C", "tr(AC AP)*I1P"]
const BASIS_TWO = vcat(BASIS_ONE, BASIS_EXTRA)

@inline function row_inv!(x::AbstractVector, AC::SM3d, AP::SM3d)
    i1, i2, i3 = invs(AC)
    x[1] = 1; x[2] = i1; x[3] = i2; x[4] = i3; x[5] = i1^2; x[6] = i1 * i2; x[7] = i1^3; x[8] = i1 * i3
    x[9] = i2^2; x[10] = i1^2 * i2; x[11] = i1^4
    p1, p2, p3 = invs(AP)
    cp = tr(AC * AP)
    x[12] = p1; x[13] = p2; x[14] = p3; x[15] = cp; x[16] = tr(AC * AC * AP); x[17] = tr(AC * AP * AP)
    x[18] = i1 * p1; x[19] = i1 * p2; x[20] = i2 * p1; x[21] = p1^2; x[22] = p1^3; x[23] = i1 * p1^2
    x[24] = i1^2 * p1; x[25] = cp * i1; x[26] = cp * p1
    return x
end

# monomial index sets in n variables up to total degree d (including the constant)
function monomials(n, d)
    out = Vector{Vector{Int}}()
    function rec(start, left, cur)
        push!(out, copy(cur))
        left == 0 && return
        for v in start:n
            push!(cur, v); rec(v, left - 1, cur); pop!(cur)
        end
    end
    rec(1, d, Int[])
    return out
end

"design matrix chunk: rows e in rng. kind = :inv (26 cols) | :flex1 (A_C monomials deg ≤ d) | :flex2 (A_C,A_P monomials deg ≤ d).
 AC, AP are (6,n) Float32 (ε=1 units); ε scales them."
function design!(X::Matrix{Float64}, AC, AP, rng, ε, kind; mons = nothing)
    n = length(rng)
    if kind == :inv
        Threads.@threads for r in 1:n
            e = rng[r]
            row_inv!(view(X, r, :), mat(view(AC, :, e), ε), mat(view(AP, :, e), ε))
        end
    else
        nv = kind == :flex1 ? 6 : 12
        Threads.@threads for r in 1:n
            e = rng[r]
            v = ntuple(i -> i <= 6 ? ε * Float64(AC[i, e]) : ε * Float64(AP[i-6, e]), nv)
            @inbounds for (m, idx) in enumerate(mons)
                p = 1.0
                for t in idx
                    p *= v[t]
                end
                X[r, m] = p
            end
        end
    end
    return X
end

end # module
