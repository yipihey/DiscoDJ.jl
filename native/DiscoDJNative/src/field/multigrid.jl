#=
Geometric multigrid Poisson solver — FFT-FREE, constant-memory, differentiable.

Solves ∇²φ = δ (periodic 7-point Laplacian) by `ncycle` geometric V-cycles: red-black GS smoother
(`gs_smooth`) + full-weighting restriction + trilinear prolongation + rediscretized Laplacian per
level, recursing to a small coarse grid. Converges geometrically (~10×/V-cycle); 3–4 cycles reach
the ~1% the sheet geometry needs. No FFT ⇒ no FFT tape ⇒ no resolution ceiling — the alternative to
the FFT Poisson for the high-resolution forward.

Differentiable with a one-line rrule: ∇² is symmetric, so the MG solve is (to convergence) self-
adjoint — the adjoint solve IS the same solve. The pullback is another `mg_solve`, constant memory,
no tape (mirrors the `gs_smooth` adjoint).

`mask` (Bool, size of δ) switches the GLOBAL solve into the AMR refined-patch solve: the fine-level
smoother is restricted to the active footprint while the COARSE levels stay global (they must — the
long-range potential comes from the whole box; masking all levels would Dirichlet-cut it). The fine
grid is then resolved only on the ~10% footprint, coarse grid global — the RAMSES/Enzo structure.
=#
export mg_solve

# full-weighting restriction (res → res÷2): [1 2 1]³/64 smooth, then sample the coarse (odd) points
function mg_restrict(r::AbstractArray{T,3}) where {T}
    s = r
    for d in 1:3
        sh = ntuple(i -> i == d ? 1 : 0, 3)
        s = (circshift(s, sh) .+ 2 .* s .+ circshift(s, .-sh)) ./ 4
    end
    return s[1:2:end, 1:2:end, 1:2:end]
end

# trilinear prolongation (res÷2 → res): zero-fill the coarse points, [1 2 1]³/8 smooth (= 8·Rᵀ)
function mg_prolong(ec::AbstractArray{T,3}, res::Int) where {T}
    z = KernelAbstractions.zeros(get_backend(ec), T, res, res, res)
    z[1:2:end, 1:2:end, 1:2:end] .= ec
    s = z
    for d in 1:3
        sh = ntuple(i -> i == d ? 1 : 0, 3)
        s = (circshift(s, sh) .+ 2 .* s .+ circshift(s, .-sh)) ./ 4
    end
    return s .* T(8)
end

# one V-cycle of ∇²φ=δ. With `mask`: smoother restricted to the active cells at THIS (fine) level;
# the coarse recursion is always global (the long-range stays on the whole box).
function mg_vcycle(φ::AbstractArray{T,3}, δ::AbstractArray{T,3}, h2::Real, ν::Int; mask=nothing) where {T}
    res = size(φ, 1)
    res <= 4 && return gs_smooth(φ, δ, T(h2), 40)
    φ = gs_smooth(φ, δ, T(h2), ν; mask = mask)
    rc = mg_restrict(δ .- laplacian7(φ, h2))
    ec = mg_vcycle(zero(rc), rc, 4 * h2, ν)              # coarse level: GLOBAL
    φ = φ .+ mg_prolong(ec, res)
    return gs_smooth(φ, δ, T(h2), ν; mask = mask)
end

"""    mg_solve(δ, h2; ncycle=6, ν=2, mask=nothing) -> φ

FFT-free multigrid solve of ∇²φ = δ (periodic 7-point, spacing²=`h2`).  `ncycle` V-cycles, `ν` pre/
post smoothing sweeps.  `mask` (Bool) → AMR footprint solve (fine grid only on the active region,
coarse global).  Differentiable w.r.t. `δ` (self-adjoint; the adjoint is the same MG solve)."""
function mg_solve(δ::AbstractArray{T,3}, h2::Real; ncycle::Int=6, ν::Int=2, mask=nothing) where {T}
    φ = zero(δ)
    for _ in 1:ncycle
        φ = mg_vcycle(φ, δ, h2, ν; mask = mask)
    end
    return φ
end

function ChainRulesCore.rrule(::typeof(mg_solve), δ::AbstractArray{T,3}, h2::Real;
                              ncycle::Int=6, ν::Int=2, mask=nothing) where {T}
    φ = mg_solve(δ, h2; ncycle = ncycle, ν = ν, mask = mask)
    function mg_solve_pullback(φ̄)
        d = φ̄ isa ChainRulesCore.AbstractZero ? zero(δ) : T.(unthunk(φ̄))
        return (NoTangent(), mg_solve(d, h2; ncycle = ncycle, ν = ν, mask = mask), NoTangent())
    end
    return φ, mg_solve_pullback
end
