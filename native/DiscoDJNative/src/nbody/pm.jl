# ── Particle-mesh force: port of DISCO-DJ `calc_acc_PM` (nbody/acc.py) ──────────
#
# Faithful port of the JAX PM acceleration, including every option of `run_nbody(method="pm")`:
#   worder         2 (CIC), 3 (TSC), 4 (PCS) mass assignment, identical kernels and stencils
#   deconvolve     divide δ̂ by the mass-assignment window twice (scatter + gather), Jing (2005)
#   antialias      0 / -1 / 1 / 2 / 3 interlacing shifts (the `dealias` decorator)
#   grad_order     0 (ik, Nyquist zeroed), 2 / 4 / 6 finite-difference gradient kernels
#   lap_order      0 (−1/k²), 2 / 4 / 6 finite-difference inverse-Laplacian kernels
#   n_resample     sheet resampling: extra particles interpolated on the Lagrangian sheet
#   resampling     :fourier (phase-shift interpolation) or :linear (trilinear, periodic)
#
# Normalisation as in DISCO-DJ: δ = mesh/ρ̄_part − 1 with ρ̄_part = n_part³/res³, and the returned
# "acceleration" is −∇φ with ∇²φ = δ (no 3Ωm/2a factor — the stepper coefficients carry it).
#
# Layout: particles are (n,n,n,3) arrays indexed [i,j,k,c] (c = 1 is x) — the JAX mesh layout
# (C order [i,j,k,c]) read column-major.  Meshes are (res,res,res) [ix,iy,iz]; FFTs use the
# pipeline's `_rfftn`/`_irfftn` ([3,1,2] region: the z axis is halved, as in numpy/JAX).
#
# Kernels are KernelAbstractions (atomic scatter, gather), so the same code runs on CPU and on
# CUDA (FFT through the CUDA extension's `_rfftn` override).  Metal: KA kernels run, but Metal.jl
# has no FFT, so a host FFT would be needed there.
#
# One deliberate difference from the JAX code: the `resampling = :linear` branch of DISCO-DJ's
# `interpolate_field` mixes box units (it evaluates at grid index (q/L_unit + dshift·L/res)/L·res),
# which is only correct for boxsize = 1.  This port evaluates at the intended grid index i + dshift;
# the parity test compares against the JAX code with that one line corrected.

export PMConfig, PMSolver, pm_acceleration, pm_acceleration!, pm_acceleration_reference

"""
    PMConfig(; res_pm, n_part, boxsize, worder=2, deconvolve=false, antialias=0,
               grad_order=4, lap_order=0, n_resample=1, resampling=:fourier)

Settings of the DISCO-DJ PM force (`run_nbody(method="pm")` keyword names, defaults as there)."""
Base.@kwdef struct PMConfig
    res_pm::Int
    n_part::Int
    boxsize::Float64
    worder::Int = 2
    deconvolve::Bool = false
    antialias::Int = 0
    grad_order::Int = 4
    lap_order::Int = 0
    n_resample::Int = 1
    resampling::Symbol = :fourier
end

# ── mass-assignment kernels (identical to scatter_and_gather.py) ───────────────
@inline _wk(::Val{2}, d) = d > 1 ? zero(d) : 1 - d
@inline _wk(::Val{3}, d) = d > 1.5 ? zero(d) : (d > 0.5 ? oftype(d, 0.5) * (oftype(d, 1.5) - d)^2 : oftype(d, 0.75) - d^2)
@inline _wk(::Val{4}, d) = d > 2 ? zero(d) : (d > 1 ? (2 - d)^3 / 6 : (4 - 6d^2 + 3d^3) / 6)
@inline _wbase(::Val{W}, g) where {W} = W % 2 == 0 ? floor(Int, g) : round(Int, g)   # even: floor, odd: round
@inline _wlo(::Val{2}) = 0;  @inline _whi(::Val{2}) = 1
@inline _wlo(::Val{3}) = -1; @inline _whi(::Val{3}) = 1
@inline _wlo(::Val{4}) = -1; @inline _whi(::Val{4}) = 2

@kernel function _pm_scatter!(mesh, @Const(X), res::Int, L, ::Val{W}) where {W}
    p = @index(Global)
    @inbounds begin
        T = eltype(X)
        gx = mod(X[p, 1], L) / L * res; gy = mod(X[p, 2], L) / L * res; gz = mod(X[p, 3], L) / L * res
        bx = _wbase(Val(W), gx); by = _wbase(Val(W), gy); bz = _wbase(Val(W), gz)
        for oz in _wlo(Val(W)):_whi(Val(W))
            wz = _wk(Val(W), abs(gz - T(bz + oz))); kz = mod(bz + oz, res) + 1
            for oy in _wlo(Val(W)):_whi(Val(W))
                wy = _wk(Val(W), abs(gy - T(by + oy))); ky = mod(by + oy, res) + 1
                for ox in _wlo(Val(W)):_whi(Val(W))
                    wx = _wk(Val(W), abs(gx - T(bx + ox))); kx = mod(bx + ox, res) + 1
                    KernelAbstractions.@atomic mesh[kx, ky, kz] += wx * wy * wz
                end
            end
        end
    end
end

@kernel function _pm_gather!(out, @Const(mesh), @Const(X), res::Int, L, ::Val{W}) where {W}
    p = @index(Global)
    @inbounds begin
        T = eltype(X)
        gx = mod(X[p, 1], L) / L * res; gy = mod(X[p, 2], L) / L * res; gz = mod(X[p, 3], L) / L * res
        bx = _wbase(Val(W), gx); by = _wbase(Val(W), gy); bz = _wbase(Val(W), gz)
        s = zero(T)
        for oz in _wlo(Val(W)):_whi(Val(W))
            wz = _wk(Val(W), abs(gz - T(bz + oz))); kz = mod(bz + oz, res) + 1
            for oy in _wlo(Val(W)):_whi(Val(W))
                wy = _wk(Val(W), abs(gy - T(by + oy))); ky = mod(by + oy, res) + 1
                for ox in _wlo(Val(W)):_whi(Val(W))
                    wx = _wk(Val(W), abs(gx - T(bx + ox))); kx = mod(bx + ox, res) + 1
                    s += mesh[kx, ky, kz] * (wx * wy * wz)
                end
            end
        end
        out[p] = s
    end
end

"""    pm_scatter!(mesh, X::(Np,3), res, L, worder)  — accumulate unit masses (no normalisation)

KernelAbstractions kernel with atomic adds on every backend.  (A lock-free CPU variant — counting
sort into stencil-wide x-slab chunks, even/odd passes — was measured 2–4× slower: particles in
Lagrangian order already give coherent, low-contention atomics.)"""
function pm_scatter!(mesh::AbstractArray{T,3}, X::AbstractMatrix{T}, res::Int, L, worder::Int) where {T}
    be = get_backend(mesh)
    _pm_scatter!(be)(mesh, X, res, T(L), Val(worder); ndrange=size(X, 1))
    synchronize(be); mesh
end

"""    pm_gather(mesh, X::(Np,3), res, L, worder) -> (Np,)"""
function pm_gather(mesh::AbstractArray{T,3}, X::AbstractMatrix{T}, res::Int, L, worder::Int) where {T}
    be = get_backend(mesh)
    out = similar(X, T, size(X, 1))
    _pm_gather!(be)(out, mesh, X, res, T(L), Val(worder); ndrange=size(X, 1))
    synchronize(be); out
end

# ── Fourier kernels on the [3,1,2] rfft layout: (res, res, res÷2+1) ─────────────
# k vectors exactly as DISCO-DJ get_fourier_grid: full axes fftshift(arange(-N/2, N/2))·2π/L
# (index N/2 holds −N/2), the halved z axis arange(0, N/2+1)·2π/L.
function _pm_kvecs(res::Int, L::Real, ::Type{T}) where {T}
    kf = T[(m < res ÷ 2 ? m : m - res) * 2π / L for m in 0:res-1]
    kh = T[m * 2π / L for m in 0:res÷2]
    return kf, kh
end

function _grad1d(k::Vector{T}, order::Int, half::Bool) where {T}
    if order == 0
        g = Complex{T}.(im .* k)
        half ? (g[end] = 0) : (g[length(k) ÷ 2 + 1] = 0)       # zero the Nyquist mode
        return g
    end
    m = maximum(abs.(k)) / π                                    # = res / L (k carries units)
    s(j) = sin.(j .* k ./ m)
    kern = order == 2 ? s(1) :
           order == 4 ? (8 .* s(1) .- s(2)) ./ 6 :
           order == 6 ? (45 .* s(1) .- 9 .* s(2) .+ s(3)) ./ 30 :
           error("grad_order must be 0, 2, 4 or 6")
    return Complex{T}.(im .* m .* kern)
end

function _invlap1d_terms(k::Vector{T}, order::Int) where {T}
    m = maximum(abs.(k)) / π
    c(j) = cos.(j .* k ./ m)
    order == 2 && return -2 .* (1 .- c(1)) .* m^2
    order == 4 && return -(c(2) .- 16 .* c(1) .+ 15) ./ 6 .* m^2
    order == 6 && return -(-2 .* c(3) .+ 27 .* c(2) .- 270 .* c(1) .+ 245) ./ 90 .* m^2
    error("lap_order must be 0, 2, 4 or 6")
end

# inverse mass-assignment kernel (Jing 2005, no shot-noise term), exponent −2·worder (double)
_invmak1d(k::Vector{T}, worder::Int) where {T} =
    (m = maximum(abs.(k)) / π; [x == 0 ? one(T) : sin(π * x) / (π * x) for x in (k ./ (2π) ./ m)] .^ (-2worder))

struct PMKernels{AC, AR}
    gx::AC; gy::AC; gz::AC           # gradient kernels, reshaped for broadcasting
    invlap::AR                       # (res,res,res÷2+1) inverse Laplacian (dense; not separable)
    mak::Union{Nothing, AR}          # deconvolution factor (dense) or nothing
end

function PMKernels(cfg::PMConfig, ::Type{T}, like::AbstractArray) where {T}
    res = cfg.res_pm
    kf, kh = _pm_kvecs(res, cfg.boxsize, T)
    gx = reshape(_grad1d(kf, cfg.grad_order, false), res, 1, 1)
    gy = reshape(_grad1d(kf, cfg.grad_order, false), 1, res, 1)
    gz = reshape(_grad1d(kh, cfg.grad_order, true), 1, 1, res ÷ 2 + 1)
    KX = reshape(kf, res, 1, 1); KY = reshape(kf, 1, res, 1); KZ = reshape(kh, 1, 1, res ÷ 2 + 1)
    if cfg.lap_order == 0
        k2 = KX .^ 2 .+ KY .^ 2 .+ KZ .^ 2
        k2[1, 1, 1] = one(T)
        il = -one(T) ./ k2
        il[1, 1, 1] = zero(T)
    else
        inv = reshape(_invlap1d_terms(kf, cfg.lap_order), res, 1, 1) .+
              reshape(_invlap1d_terms(kf, cfg.lap_order), 1, res, 1) .+
              reshape(_invlap1d_terms(kh, cfg.lap_order), 1, 1, res ÷ 2 + 1)
        inv[1, 1, 1] = -one(T)
        il = one(T) ./ inv
    end
    mak = cfg.deconvolve ?
        reshape(_invmak1d(kf, cfg.worder), res, 1, 1) .* reshape(_invmak1d(kf, cfg.worder), 1, res, 1) .*
        reshape(_invmak1d(kh, cfg.worder), 1, 1, res ÷ 2 + 1) : nothing
    dev(x) = (y = similar(like, eltype(x), size(x)); copyto!(y, x); y)
    PMKernels(dev(gx), dev(gy), dev(gz), dev(il), mak === nothing ? nothing : dev(mak))
end

# ── sheet resampling (spawn_interpolated_particles) ─────────────────────────────
_lagr1d(n, L, ::Type{T}) where {T} = T[(i - 1) * (L / n) for i in 1:n]

# X::(n,n,n,3) → resampled positions at Lagrangian offset `d` (grid cells), wrapped to [0,L)
function _spawn(X::AbstractArray{T,4}, d::NTuple{3,Float64}, cfg::PMConfig, like) where {T}
    n = size(X, 1); L = T(cfg.boxsize)
    q1 = _lagr1d(n, cfg.boxsize, T)
    qs = (reshape(q1, n, 1, 1), reshape(q1, 1, n, 1), reshape(q1, 1, 1, n))
    dev(x) = (y = similar(like, eltype(x), size(x)); copyto!(y, x); y)
    qd = map(dev, qs)
    out = similar(X)
    if cfg.resampling == :fourier
        # DISCO-DJ: relative k (2πm/N) / 2π = m/N; phase exp(i 2π Σ_d (m_d/N) d_d); m as get_fourier_grid
        mf = T[(m < n ÷ 2 ? m : m - n) / n for m in 0:n-1]; mh = T[m / n for m in 0:n÷2]
        kd = reshape(mf, n, 1, 1) .* T(d[1]) .+ reshape(mf, 1, n, 1) .* T(d[2]) .+ reshape(mh, 1, 1, n ÷ 2 + 1) .* T(d[3])
        ph = exp.(im .* T(2π) .* kd)
        phd = dev(Complex{T}.(ph))
        for c in 1:3
            psi = mod.(view(X, :, :, :, c) .- qd[c] .+ L / 2, L) .- L / 2
            npsi = _irfftn(_rfftn(psi) .* phd, n)
            out[:, :, :, c] .= mod.(qd[c] .+ npsi .+ T(d[c] * cfg.boxsize / n) .+ L, L)
        end
    elseif cfg.resampling == :linear
        for c in 1:3
            psi = Array(mod.(view(X, :, :, :, c) .- qd[c] .+ L / 2, L) .- L / 2)
            npsi = _trilinear_shift(psi, d)
            out[:, :, :, c] .= mod.(qd[c] .+ dev(npsi) .+ T(d[c] * cfg.boxsize / n) .+ L, L)
        end
    else
        error("resampling must be :fourier or :linear")
    end
    return out
end

# periodic trilinear interpolation of f at grid index (i + d1, j + d2, k + d3) (0 <= d < 1)
function _trilinear_shift(f::Array{T,3}, d::NTuple{3,Float64}) where {T}
    n = size(f, 1); out = similar(f)
    dx, dy, dz = T.(d)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        i1 = mod1(i + 1, n); j1 = mod1(j + 1, n); k1 = mod1(k + 1, n)
        out[i, j, k] = f[i,j,k]*(1-dx)*(1-dy)*(1-dz) + f[i1,j,k]*dx*(1-dy)*(1-dz) +
                       f[i,j1,k]*(1-dx)*dy*(1-dz) + f[i,j,k1]*(1-dx)*(1-dy)*dz +
                       f[i1,j1,k]*dx*dy*(1-dz) + f[i1,j,k1]*dx*(1-dy)*dz +
                       f[i,j1,k1]*(1-dx)*dy*dz + f[i1,j1,k1]*dx*dy*dz
    end
    out
end

_resample_shifts(nr::Int) = nr == 1 ? NTuple{3,Float64}[] :
    [(a, b, c) for a in (0:nr-1) ./ nr for b in (0:nr-1) ./ nr for c in (0:nr-1) ./ nr][2:end]

# interlacing shift list of the `dealias` decorator (same order as itertools.product)
function _aa_shifts(aa::Int, dhalf::Float64)
    aa == 0 && return [(0.0, 0.0, 0.0)]
    aa == -1 && return [(dhalf, dhalf, dhalf)]
    base = [(a, b, c) for a in (0.0, dhalf) for b in (0.0, dhalf) for c in (0.0, dhalf)]
    aa == 1 && return [(0.0, 0.0, 0.0), (dhalf, dhalf, dhalf)]
    aa == 2 && return base
    q = dhalf / 2
    aa == 3 && return vcat(base, [(a, b, c) for a in (-q, q) for b in (-q, q) for c in (-q, q)])
    error("antialias must be -1, 0, 1, 2 or 3")
end

"""
    pm_acceleration_reference(psi, cfg; kernels) -> (n,n,n,3)

Straightforward (allocating, one inverse FFT per force component) transcription of DISCO-DJ
`calc_acc_PM`; kept as the reference the optimised `PMSolver` path is tested against."""
function pm_acceleration_reference(psi::AbstractArray{T,4}, cfg::PMConfig;
                                   kernels::PMKernels = PMKernels(cfg, T, psi)) where {T}
    n = cfg.n_part; @assert size(psi, 1) == n
    L = cfg.boxsize; res = cfg.res_pm
    q1 = _lagr1d(n, L, T)
    dev(x) = (y = similar(psi, eltype(x), size(x)); copyto!(y, x); y)
    qd = (dev(reshape(q1, n, 1, 1)), dev(reshape(q1, 1, n, 1)), dev(reshape(q1, 1, 1, n)))
    X = similar(psi)
    for c in 1:3; X[:, :, :, c] .= view(psi, :, :, :, c) .+ qd[c]; end
    shifts = _aa_shifts(cfg.antialias, 0.5 * L / res)
    acc = fill!(similar(psi), zero(T))
    Xs = similar(X)
    for s in shifts
        for c in 1:3; Xs[:, :, :, c] .= view(X, :, :, :, c) .+ T(s[c]); end
        acc .+= _acc_pm_single(Xs, cfg, kernels)
    end
    length(shifts) > 1 && (acc ./= length(shifts))
    return acc
end

function _acc_pm_single(X::AbstractArray{T,4}, cfg::PMConfig, K::PMKernels) where {T}
    n = cfg.n_part; res = cfg.res_pm; L = cfg.boxsize; W = cfg.worder
    Xf = reshape(X, n^3, 3)
    mesh = fill!(similar(X, T, res, res, res), zero(T))
    rhomean = T(n^3 / res^3)
    pm_scatter!(mesh, Xf, res, L, W)
    delta = mesh ./ rhomean .- one(T)
    if cfg.n_resample > 1
        for d in _resample_shifts(cfg.n_resample)
            m2 = fill!(similar(mesh), zero(T))
            pm_scatter!(m2, reshape(_spawn(X, d, cfg, X), n^3, 3), res, L, W)
            delta .+= m2 ./ rhomean .- one(T)
        end
        delta ./= T(cfg.n_resample^3)
    end
    fd = _rfftn(delta)
    K.mak === nothing || (fd .*= K.mak)
    CUDA_safe_setdc!(fd)
    acc = similar(X)
    for (c, g) in enumerate((K.gx, K.gy, K.gz))
        fa = .-(g .* K.invlap .* fd)
        field = _irfftn(fa, res)
        acc[:, :, :, c] .= reshape(pm_gather(field, Xf, res, L, W), n, n, n)
    end
    return acc
end

# set the DC mode of a Fourier field to zero without scalar indexing on device arrays
CUDA_safe_setdc!(f) = (view(f, 1:1, 1:1, 1:1) .= 0; f)

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Optimised PM solver (same mathematics, round-off-level differences only)
#
#  * one combined Fourier multiplier  Kφ = invlap · [deconvolution] · 1/(ρ̄·n_sets·N_FFT), DC = 0
#    (the "−1" of δ = mesh/ρ̄ − 1 only touches the DC mode, which DISCO-DJ zeroes)
#  * FFTW plans + preallocated buffers on the CPU (unnormalised brfft; 1/N folded into Kφ)
#  * finite-difference gradients (order 2/4/6): ONE inverse FFT of φ, then the equivalent
#    real-space central-difference stencil — i(8 sin kh − sin 2kh)/6h ≡ [8(φ₊₁−φ₋₁) − (φ₊₂−φ₋₂)]/12h
#    exactly (circulant identity) — instead of three inverse FFTs
#  * ik gradient (order 0): three inverse FFTs as before, fused multiply kernel
#  * one fused 3-component gather (weights computed once per particle)
#  * sheet resampling: ψ̂ transformed once per force evaluation, not once per resampling offset
# ═══════════════════════════════════════════════════════════════════════════════════════════════

@kernel function _k_mulK!(out, @Const(fd), @Const(Kφ))
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = fd[i, j, k] * Kφ[i, j, k]
end

@kernel function _k_mulKg!(out, @Const(fd), @Const(Kφ), @Const(g), ax::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        gv = ax == 1 ? g[i] : ax == 2 ? g[j] : g[k]
        out[i, j, k] = -(gv * Kφ[i, j, k] * fd[i, j, k])
    end
end

# F[.,c] = −D_c φ with the order-2/4/6 central-difference stencil (periodic), 1/h folded in c1..c3.
# `nb[i, s+4]` = mod1(i + s, res) for s = −3..3 (precomputed: no integer division in the kernel).
@kernel function _k_fdgrad!(F, @Const(φ), @Const(nb), c1, c2, c3)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        F[1, i, j, k] = -(c1 * (φ[nb[i, 5], j, k] - φ[nb[i, 3], j, k]) + c2 * (φ[nb[i, 6], j, k] - φ[nb[i, 2], j, k]) +
                          c3 * (φ[nb[i, 7], j, k] - φ[nb[i, 1], j, k]))
        F[2, i, j, k] = -(c1 * (φ[i, nb[j, 5], k] - φ[i, nb[j, 3], k]) + c2 * (φ[i, nb[j, 6], k] - φ[i, nb[j, 2], k]) +
                          c3 * (φ[i, nb[j, 7], k] - φ[i, nb[j, 1], k]))
        F[3, i, j, k] = -(c1 * (φ[i, j, nb[k, 5]] - φ[i, j, nb[k, 3]]) + c2 * (φ[i, j, nb[k, 6]] - φ[i, j, nb[k, 2]]) +
                          c3 * (φ[i, j, nb[k, 7]] - φ[i, j, nb[k, 1]]))
    end
end

@kernel function _k_gather3!(acc, @Const(F), @Const(X), res::Int, L, ::Val{W}) where {W}
    p = @index(Global)
    @inbounds begin
        T = eltype(X)
        gx = mod(X[p, 1], L) / L * res; gy = mod(X[p, 2], L) / L * res; gz = mod(X[p, 3], L) / L * res
        bx = _wbase(Val(W), gx); by = _wbase(Val(W), gy); bz = _wbase(Val(W), gz)
        s1 = zero(T); s2 = zero(T); s3 = zero(T)
        for oz in _wlo(Val(W)):_whi(Val(W))
            wz = _wk(Val(W), abs(gz - T(bz + oz))); kz = mod(bz + oz, res) + 1
            for oy in _wlo(Val(W)):_whi(Val(W))
                wy = _wk(Val(W), abs(gy - T(by + oy))); ky = mod(by + oy, res) + 1
                for ox in _wlo(Val(W)):_whi(Val(W))
                    wx = _wk(Val(W), abs(gx - T(bx + ox))); kx = mod(bx + ox, res) + 1
                    w = wx * wy * wz
                    s1 += F[1, kx, ky, kz] * w; s2 += F[2, kx, ky, kz] * w; s3 += F[3, kx, ky, kz] * w
                end
            end
        end
        acc[p, 1] += s1; acc[p, 2] += s2; acc[p, 3] += s3
    end
end

"""
    PMSolver(cfg::PMConfig, like::AbstractArray{T,4})

Preallocated workspace for the DISCO-DJ PM force (see `pm_acceleration!`): combined Fourier
multiplier, gradient kernels, meshes, FFT buffers and (on the CPU) FFTW plans."""
struct PMSolver{T, AR3, AC3, AR4, AR4p, AGV, P1, P2}
    cfg::PMConfig
    Kφ::AR3                     # (res,res,res÷2+1) combined real multiplier
    g::NTuple{3,AGV}            # 1-D ik gradient kernels (order 0)
    mesh::AR3                   # (res,res,res)
    fd::AC3; cbuf::AC3          # (res,res,res÷2+1)
    φ::AR3                      # (res,res,res) potential (FD path)
    F::AR4                      # (3,res,res,res) force meshes (components interleaved per node)
    X::AR4p; Xs::AR4p; acc::AR4p  # particle work arrays (n,n,n,3)
    pf::P1; pb::P2              # FFTW plans (CPU) or nothing
    nb::AbstractMatrix{Int32}   # periodic neighbour table for the FD stencil (res × 7)
    rs::Any                     # Fourier-resampling workspace (particle grid) or nothing
end

function PMSolver(cfg::PMConfig, like::AbstractArray{T,4}) where {T}
    res = cfg.res_pm; n = cfg.n_part; nh = res ÷ 2 + 1
    kf, kh = _pm_kvecs(res, cfg.boxsize, T)
    K = PMKernels(cfg, T, Array(like[1:1, 1:1, 1:1, 1:1]))          # host copies of the kernels
    nsets = cfg.n_resample^3
    Kφ = K.invlap .* (K.mak === nothing ? one(T) : K.mak) ./ T(n^3 / res^3) ./ T(nsets) ./ T(res)^3
    Kφ[1, 1, 1] = zero(T)
    dev(x) = (y = similar(like, eltype(x), size(x)); copyto!(y, x); y)
    gk = (dev(_grad1d(kf, 0, false)), dev(_grad1d(kf, 0, false)), dev(_grad1d(kh, 0, true)))
    mesh = similar(like, T, res, res, res); φ = similar(mesh)
    fd = similar(like, Complex{T}, res, res, nh); cbuf = similar(fd)
    F = similar(like, T, 3, res, res, res)
    X = similar(like, T, n, n, n, 3); Xs = similar(X); acc = similar(X)
    pf = pb = nothing
    if like isa Array
        pf = FFTW.plan_rfft(mesh, [3, 1, 2]; flags=FFTW.ESTIMATE)
        pb = FFTW.plan_brfft(cbuf, res, [3, 1, 2]; flags=FFTW.ESTIMATE)
    end
    nb = dev(Int32[mod1(i + s, res) for i in 1:res, s in -3:3])
    rs = nothing
    if cfg.n_resample > 1 && cfg.resampling == :fourier
        ψhat = similar(like, Complex{T}, n, n, n ÷ 2 + 1, 3); cb = similar(like, Complex{T}, n, n, n ÷ 2 + 1)
        rb = similar(like, T, n, n, n); Xr = similar(X)
        pfn = like isa Array ? FFTW.plan_rfft(rb, [3, 1, 2]; flags=FFTW.ESTIMATE) : nothing
        pbn = like isa Array ? FFTW.plan_brfft(cb, n, [3, 1, 2]; flags=FFTW.ESTIMATE) : nothing
        rs = (ψhat=ψhat, cb=cb, rb=rb, Xr=Xr, pf=pfn, pb=pbn)
    end
    PMSolver{T, typeof(mesh), typeof(fd), typeof(F), typeof(X), typeof(gk[1]), typeof(pf), typeof(pb)}(
        cfg, dev(Kφ), gk, mesh, fd, cbuf, φ, F, X, Xs, acc, pf, pb, nb, rs)
end

_fwd!(S::PMSolver) = S.pf === nothing ? (S.fd .= _rfftn(S.mesh)) : mul!(S.fd, S.pf, S.mesh)
_bwd!(out, S::PMSolver) = S.pb === nothing ? (out .= _brfftn(S.cbuf, S.cfg.res_pm)) : mul!(out, S.pb, S.cbuf)

function _fd_coeffs(order::Int, h)
    order == 2 && return (1 / (2h), 0.0, 0.0)
    order == 4 && return (8 / (12h), -1 / (12h), 0.0)
    order == 6 && return (45 / (60h), -9 / (60h), 1 / (60h))
    error("grad_order must be 0, 2, 4 or 6")
end

# ── Fourier sheet resampling on the particle grid (planned FFTs, fused threaded kernels) ──────────
# ψ = mod(X − q + L/2, L) − L/2 is transformed ONCE per force evaluation; each resampling offset d
# then needs one phase multiply (separable: exp(2πi Σ m_d d_d / n) = pₓ pᵧ p_z) and one inverse FFT
# per component.  Same result as `_spawn` up to round-off.
@kernel function _k_wrap_psi!(out, @Const(X), @Const(q), c::Int, L)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = mod(X[i, j, k, c] - q[c == 1 ? i : c == 2 ? j : k] + L / 2, L) - L / 2
end
@kernel function _k_phase!(out, @Const(ψh), @Const(px), @Const(py), @Const(pz), scale)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = ψh[i, j, k] * (px[i] * py[j] * pz[k] * scale)
end
@kernel function _k_place!(Xr, @Const(npsi), @Const(q), c::Int, shift, L)
    i, j, k = @index(Global, NTuple)
    @inbounds Xr[i, j, k, c] = mod(q[c == 1 ? i : c == 2 ? j : k] + npsi[i, j, k] + shift + L, L)
end

function _resample_prepare!(S::PMSolver{T}, Xs, q) where {T}
    R = S.rs; n = S.cfg.n_part; L = T(S.cfg.boxsize); be = get_backend(Xs)
    for c in 1:3
        _k_wrap_psi!(be)(R.rb, Xs, q, c, L; ndrange=(n, n, n)); synchronize(be)
        R.pf === nothing ? (R.ψhat[:, :, :, c] .= _rfftn(R.rb)) : mul!(view(R.ψhat, :, :, :, c), R.pf, R.rb)
    end
end

function _resample_offset!(S::PMSolver{T}, d, q) where {T}
    R = S.rs; n = S.cfg.n_part; L = T(S.cfg.boxsize); be = get_backend(R.rb)
    mf = T[(m < n ÷ 2 ? m : m - n) / n for m in 0:n-1]; mh = T[m / n for m in 0:n÷2]
    dev(x) = (y = similar(R.rb, eltype(x), size(x)); copyto!(y, x); y)
    px = dev(exp.(im .* T(2π) .* mf .* T(d[1]))); py = dev(exp.(im .* T(2π) .* mf .* T(d[2])))
    pz = dev(exp.(im .* T(2π) .* mh .* T(d[3])))
    for c in 1:3
        _k_phase!(be)(R.cb, view(R.ψhat, :, :, :, c), px, py, pz, one(T) / T(n)^3; ndrange=(n, n, n ÷ 2 + 1))
        synchronize(be)
        R.pb === nothing ? (R.rb .= _brfftn(R.cb, n)) : mul!(R.rb, R.pb, R.cb)
        _k_place!(be)(R.Xr, R.rb, q, c, T(d[c] * S.cfg.boxsize / n), L; ndrange=(n, n, n)); synchronize(be)
    end
    return R.Xr
end

function _acc_single!(S::PMSolver{T}, Xs, qd) where {T}
    cfg = S.cfg; n = cfg.n_part; res = cfg.res_pm; L = cfg.boxsize; W = cfg.worder
    be = get_backend(S.mesh)
    fill!(S.mesh, zero(T))
    pm_scatter!(S.mesh, reshape(Xs, n^3, 3), res, L, W)
    if cfg.n_resample > 1
        if cfg.resampling == :fourier
            q = qd[1][:]                                   # 1-D lattice coordinates (device)
            _resample_prepare!(S, Xs, q)
            for d in _resample_shifts(cfg.n_resample)
                Xr = _resample_offset!(S, d, q)
                pm_scatter!(S.mesh, reshape(Xr, n^3, 3), res, L, W)
            end
        else
            for d in _resample_shifts(cfg.n_resample)
                pm_scatter!(S.mesh, reshape(_spawn(Xs, d, cfg, Xs), n^3, 3), res, L, W)
            end
        end
    end
    _fwd!(S)
    nh = res ÷ 2 + 1
    if cfg.grad_order == 0
        for ax in 1:3
            _k_mulKg!(be)(S.cbuf, S.fd, S.Kφ, S.g[ax], ax; ndrange=(res, res, nh)); synchronize(be)
            _bwd!(S.φ, S)
            S.F[ax, :, :, :] .= S.φ
        end
    else
        _k_mulK!(be)(S.cbuf, S.fd, S.Kφ; ndrange=(res, res, nh)); synchronize(be)
        _bwd!(S.φ, S)
        c1, c2, c3 = T.(_fd_coeffs(cfg.grad_order, L / res))
        _k_fdgrad!(be)(S.F, S.φ, S.nb, c1, c2, c3; ndrange=(res, res, res)); synchronize(be)
    end
    _k_gather3!(be)(reshape(S.acc, n^3, 3), S.F, reshape(Xs, n^3, 3), res, T(L), Val(W); ndrange=n^3)
    synchronize(be)
end

"""
    pm_acceleration!(S::PMSolver, psi) -> S.acc

DISCO-DJ `calc_acc_PM` using the preallocated solver `S`; returns (a reference to) `S.acc`."""
function pm_acceleration!(S::PMSolver{T}, psi::AbstractArray{T,4}) where {T}
    cfg = S.cfg; n = cfg.n_part; L = cfg.boxsize
    q1 = _lagr1d(n, L, T)
    dev(x) = (y = similar(psi, eltype(x), size(x)); copyto!(y, x); y)
    qd = (dev(reshape(q1, n, 1, 1)), dev(reshape(q1, 1, n, 1)), dev(reshape(q1, 1, 1, n)))
    for c in 1:3; S.X[:, :, :, c] .= view(psi, :, :, :, c) .+ qd[c]; end
    shifts = _aa_shifts(cfg.antialias, 0.5 * L / cfg.res_pm)
    fill!(S.acc, zero(T))
    for s in shifts
        for c in 1:3; S.Xs[:, :, :, c] .= view(S.X, :, :, :, c) .+ T(s[c]); end
        _acc_single!(S, S.Xs, qd)
    end
    length(shifts) > 1 && (S.acc ./= length(shifts))
    return S.acc
end

"""
    pm_acceleration(psi::(n,n,n,3), cfg::PMConfig; solver=PMSolver(cfg, psi)) -> (n,n,n,3)

DISCO-DJ `calc_acc_PM(psi, …)`: particles at X = q + ψ, returns −∇φ (∇²φ = δ) at the particles,
with all the options of `PMConfig` (a copy of the solver's output buffer)."""
pm_acceleration(psi::AbstractArray{T,4}, cfg::PMConfig; solver::PMSolver = PMSolver(cfg, psi)) where {T} =
    copy(pm_acceleration!(solver, psi))
