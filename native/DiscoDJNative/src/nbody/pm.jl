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

export PMConfig, pm_acceleration

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

"""    pm_scatter!(mesh, X::(Np,3), res, L, worder)  — accumulate unit masses (no normalisation)"""
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
    pm_acceleration(psi::(n,n,n,3), cfg::PMConfig; kernels=PMKernels(cfg, T, psi)) -> (n,n,n,3)

DISCO-DJ `calc_acc_PM(psi, …)`: particles at X = q + ψ, returns −∇φ (∇²φ = δ) at the particles,
with all the options of `PMConfig`."""
function pm_acceleration(psi::AbstractArray{T,4}, cfg::PMConfig;
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
