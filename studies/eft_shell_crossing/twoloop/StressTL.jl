# ── Experiment 3: coarse-grained effective stress from particles (CPU) ─────────────────────────────────────────────
#
# τ̂_ij ≡ τ_ij/(ρ̄Ω_mℋ²) (EdS; length²) of a particle state (Ψ, Π = dΨ/dD) at growth D, coarse-grained with
# W_Λ(k) = exp(−k²/2Λ²):
#   kinetic        D² { [nΠ_iΠ_j]_Λ − [nΠ_i]_Λ[nΠ_j]_Λ / [n]_Λ }                          (n = 1 + δ)
#   gravitational  (2/3){ [∂_iϕ∂_jϕ − ½δ_ij(∂ϕ)²]_Λ − (∂_iϕ_Λ∂_jϕ_Λ − ½δ_ij(∂ϕ_Λ)²) },   ∇²ϕ = (3/2)δ
# (twoloop/README.md §2).  The Lagrangian sheet is Fourier-resampled 2³ (Ψ and Π) before a CIC deposit of the ten
# moments n, nΠ_i, nΠ_iΠ_j on an Nm³ mesh; the CIC window is deconvolved; the smoothed fields are formed on a coarse
# grid Nc(Λ) whose Nyquist frequency is ≥ 2·k_keep (k_keep = 3.5Λ, W = 2e−3) so the nonlinear products are alias-free.
# Output per Λ: low-k Fourier coefficients (rfft/N³ convention of `field_modes`) of τ̂ (kinetic and gravitational
# separately, 6 components each), δ_Λ, π_Λ = [nΠ]_Λ, and the curl of v_Λ = π_Λ/n_Λ.
module StressTL

using FFTW, KernelAbstractions, LinearAlgebra, Printf
using KernelAbstractions: @atomic

export sheet_resample, deposit_moments, stress_modes, TLModes, COMP

const COMP = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))     # tensor component order (xx, yy, zz, xy, xz, yz)

"""Low-k mode list (integer n, |k| ≤ k_max, half space as `ModeSet`): grid-independent ordering."""
struct TLModes
    nvec::Matrix{Int}; kmag::Vector{Float64}; L::Float64
end
function TLModes(L::Real, kmax::Real)
    nmax = floor(Int, kmax * L / 2π); nv = NTuple{3,Int}[]
    for nz in 0:nmax, ny in -nmax:nmax, nx in -nmax:nmax
        n2 = nx^2 + ny^2 + nz^2
        (n2 == 0 || 2π / L * sqrt(n2) > kmax) && continue
        (nz == 0 && !(ny > 0 || (ny == 0 && nx > 0))) && continue
        push!(nv, (nx, ny, nz))
    end
    nvec = [nv[m][d] for m in eachindex(nv), d in 1:3]
    return TLModes(nvec, vec(2π / L .* sqrt.(sum(abs2, nvec; dims=2))), Float64(L))
end
# linear indices of the modes on an (N,N,N÷2+1) rfft array ([1,2,3] layout: x fastest)
modeidx(md::TLModes, N::Int) = (li = LinearIndices((N ÷ 2 + 1, N, N));
    [li[md.nvec[m, 1] >= 0 ? md.nvec[m, 1] + 1 : error("x must be the half axis"), mod(md.nvec[m, 2], N) + 1, mod(md.nvec[m, 3], N) + 1] for m in 1:size(md.nvec, 1)])

# We use rfft along x (dim 1) so the half axis is x; modes with n_x < 0 are taken as conj of (−n).
function extract(fh::AbstractArray{Complex{T},3}, md::TLModes, N::Int) where {T}
    M = size(md.nvec, 1); out = Vector{ComplexF64}(undef, M)
    @inbounds for m in 1:M
        nx, ny, nz = md.nvec[m, 1], md.nvec[m, 2], md.nvec[m, 3]
        if nx >= 0
            out[m] = fh[nx + 1, mod(ny, N) + 1, mod(nz, N) + 1]
        else
            out[m] = conj(fh[-nx + 1, mod(-ny, N) + 1, mod(-nz, N) + 1])
        end
    end
    return out ./ N^3
end

kvec1(N, L) = [2π / L * (m < N ÷ 2 ? m : m - N) for m in 0:N-1]
khalf(N, L) = [2π / L * m for m in 0:N÷2]

"""2³ Fourier resampling of the Lagrangian sheet: returns (X, V) as (8N³, 3) Float64 (positions in [0, L))."""
function sheet_resample(Ψ::AbstractArray{<:Real,4}, Π::AbstractArray{<:Real,4}, L::Real; nr::Int=2)
    N = size(Ψ, 1); Np = N^3
    X = Matrix{Float64}(undef, nr^3 * Np, 3); V = similar(X)
    q1 = [(i - 1) * L / N for i in 1:N]
    kx = khalf(N, L); ky = kvec1(N, L); kz = kvec1(N, L)
    Fh = [rfft(Float64.(view(F, :, :, :, c))) for F in (Ψ, Π), c in 1:3]     # 2×3
    s = 0
    for a in 0:nr-1, b in 0:nr-1, cc in 0:nr-1
        d = (a, b, cc) ./ nr .* (L / N)                                        # Lagrangian offset (length)
        ph = [exp(im * (kx[i] * d[1] + ky[j] * d[2] + kz[l] * d[3])) for i in eachindex(kx), j in eachindex(ky), l in eachindex(kz)]
        for c in 1:3
            ψ = irfft(Fh[1, c] .* ph, N); π_ = irfft(Fh[2, c] .* ph, N)
            Threads.@threads for l in 1:N
                @inbounds for j in 1:N, i in 1:N
                    p = s * Np + i + N * (j - 1) + N^2 * (l - 1)
                    qq = (q1[i], q1[j], q1[l])[c] + d[c]
                    X[p, c] = mod(qq + ψ[i, j, l], L); V[p, c] = π_[i, j, l]
                end
            end
        end
        s += 1
    end
    return X, V
end

@kernel function _k_cic_moments!(M, @Const(X), @Const(V), Nm::Int, L)
    p = @index(Global)
    @inbounds begin
        gx = X[p, 1] / L * Nm; gy = X[p, 2] / L * Nm; gz = X[p, 3] / L * Nm
        ix = floor(Int, gx); iy = floor(Int, gy); iz = floor(Int, gz)
        fx = gx - ix; fy = gy - iy; fz = gz - iz
        v1 = V[p, 1]; v2 = V[p, 2]; v3 = V[p, 3]
        w = (1.0, v1, v2, v3, v1 * v1, v2 * v2, v3 * v3, v1 * v2, v1 * v3, v2 * v3)
        for oz in 0:1, oy in 0:1, ox in 0:1
            wt = (ox == 0 ? 1 - fx : fx) * (oy == 0 ? 1 - fy : fy) * (oz == 0 ? 1 - fz : fz)
            kx = mod(ix + ox, Nm) + 1; ky = mod(iy + oy, Nm) + 1; kz = mod(iz + oz, Nm) + 1
            for f in 1:10
                @atomic M[kx, ky, kz, f] += wt * w[f]
            end
        end
    end
end

"""CIC deposit of the ten moments (n, nV_i, nV_iV_j), normalised so n = 1 + δ (mean 1)."""
function deposit_moments(X::Matrix{Float64}, V::Matrix{Float64}, Nm::Int, L::Real)
    M = zeros(Float64, Nm, Nm, Nm, 10)
    be = CPU()
    _k_cic_moments!(be, 1024)(M, X, V, Nm, Float64(L); ndrange=size(X, 1))
    KernelAbstractions.synchronize(be)
    M .*= Nm^3 / size(X, 1)
    return M
end

# Fourier coefficients of the coarse grid Nc taken from a fine rfft (Nm) array, with a filter applied
function to_coarse(fh::Array{ComplexF64,3}, Nm::Int, Nc::Int, filt)
    out = zeros(ComplexF64, Nc ÷ 2 + 1, Nc, Nc)
    h = Nc ÷ 2
    @inbounds for l in 1:Nc, j in 1:Nc, i in 1:h+1
        my = j - 1 < h ? j - 1 : j - 1 - Nc; mz = l - 1 < h ? l - 1 : l - 1 - Nc
        (my == -h || mz == -h || i == h + 1) && continue                     # drop coarse Nyquist planes
        out[i, j, l] = fh[i, mod(my, Nm) + 1, mod(mz, Nm) + 1] * filt(i - 1, my, mz) * (Nc / Nm)^3
    end
    return out
end

cic_window(m, N) = m == 0 ? 1.0 : (s = sin(π * m / N) / (π * m / N); s^2)

"""
    stress_modes(Mom, D, L, Λs, md; f2om=1.0) -> Dict(Λ => NamedTuple)

From the deposited moments `Mom` (Nm³×10) of a state at growth D: coarse-grained τ̂ (kinetic, gravitational),
δ_Λ, π_Λ and curl v_Λ on the modes `md`, for each Λ in `Λs`.  `f2om` = f²/Ω_m (1 in EdS) multiplies the kinetic part."""
function stress_modes(Mom::Array{Float64,4}, D::Real, L::Real, Λs, md::TLModes; f2om::Real=1.0, filter::Symbol=:gauss)
    Nm = size(Mom, 1)
    # deconvolved rfft of every moment on the fine mesh
    dec = [1 / (cic_window(i, Nm) * cic_window(j < Nm ÷ 2 ? j : j - Nm, Nm) * cic_window(l < Nm ÷ 2 ? l : l - Nm, Nm))
           for i in 0:Nm÷2, j in 0:Nm-1, l in 0:Nm-1]
    Fm = [rfft(view(Mom, :, :, :, f)) .* dec for f in 1:10]
    # gravitational: ϕ (∇²ϕ = 3/2 δ) and g_ij on the fine mesh
    kx = khalf(Nm, L); ky = kvec1(Nm, L); kz = kvec1(Nm, L)
    k2 = [kx[i]^2 + ky[j]^2 + kz[l]^2 for i in eachindex(kx), j in eachindex(ky), l in eachindex(kz)]; k2[1] = 1
    ϕh = -1.5 .* Fm[1] ./ k2; ϕh[1] = 0
    kv = (reshape(kx, :, 1, 1), reshape(ky, 1, :, 1), reshape(kz, 1, 1, :))
    gϕ = [irfft(im .* kv[c] .* ϕh, Nm) for c in 1:3]
    g2 = gϕ[1] .^ 2 .+ gϕ[2] .^ 2 .+ gϕ[3] .^ 2
    Gh = [rfft(gϕ[a] .* gϕ[b] .- (a == b ? 0.5 .* g2 : 0.0)) for (a, b) in COMP]
    gϕ = nothing; g2 = nothing
    res = Dict{Float64,Any}()
    for Λ in Λs
        kkeep = filter === :sharp ? Λ : 3.5Λ          # sharp: W = 1[k < Λ] (the clean EFT split, matched to the IR member)
        Nc = max(64, 2 * ceil(Int, 2kkeep * L / (2π) + 1)); Nc += isodd(Nc) ? 1 : 0
        Nc = min(Nc, Nm)
        W(i, my, mz) = (k2_ = (2π / L)^2 * (i^2 + my^2 + mz^2); k2_ >= kkeep^2 ? 0.0 : (filter === :sharp ? 1.0 : exp(-k2_ / (2Λ^2))))
        S = [irfft(to_coarse(Fm[f], Nm, Nc, W), Nc) for f in 1:10]          # smoothed moments on Nc³
        n = S[1]
        ckx = khalf(Nc, L); cky = kvec1(Nc, L); ckz = kvec1(Nc, L)
        ck = (reshape(ckx, :, 1, 1), reshape(cky, 1, :, 1), reshape(ckz, 1, 1, :))
        # kinetic stress
        kin = [D^2 * f2om .* (S[4 + c] .- S[1 + a] .* S[1 + b] ./ n) for (c, (a, b)) in enumerate(COMP)]   # S[5:10] = nV_aV_b
        # gravitational stress
        ϕΛ = -1.5 .* rfft(n .- 1) ./ [max(ckx[i]^2 + cky[j]^2 + ckz[l]^2, 1e-30) for i in eachindex(ckx), j in eachindex(cky), l in eachindex(ckz)]
        ϕΛ[1] = 0
        gΛ = [irfft(im .* ck[c] .* ϕΛ, Nc) for c in 1:3]; gl2 = gΛ[1] .^ 2 .+ gΛ[2] .^ 2 .+ gΛ[3] .^ 2
        grav = [(2 / 3) .* (irfft(to_coarse(Gh[c], Nm, Nc, W), Nc) .- (gΛ[a] .* gΛ[b] .- (a == b ? 0.5 .* gl2 : 0.0)))
                for (c, (a, b)) in enumerate(COMP)]
        # long fields: δ_Λ, π_Λ, curl v_Λ
        v = [S[1 + c] ./ n for c in 1:3]; vh = [rfft(v[c]) for c in 1:3]
        curl = [im .* ck[2] .* vh[3] .- im .* ck[3] .* vh[2], im .* ck[3] .* vh[1] .- im .* ck[1] .* vh[3],
                im .* ck[1] .* vh[2] .- im .* ck[2] .* vh[1]]
        ex(f) = extract(rfft(f), md, Nc); exh(fh) = extract(fh, md, Nc)
        res[Λ] = (kin=reduce(hcat, ex.(kin)), grav=reduce(hcat, ex.(grav)), delta=ex(n .- 1),
                  pi=reduce(hcat, [ex(S[1 + c]) for c in 1:3]), curlv=reduce(hcat, exh.(curl)), Nc=Nc,
                  nmin=minimum(n))                                   # min of n_Λ (1/n_Λ in the kinetic stress)
    end
    return res
end

end # module
