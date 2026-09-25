# ── Fast de-aliased nLPT engine: shared derivative fields + fused KernelAbstractions sources ──
#
# Same mathematics as the term-by-term engine in nlpt_core.jl (a faithful port of DISCO-DJ's
# `compute_core` / `compute_core_exact`), organised for speed:
#
#   1. `ext_derivs(K, ψ)` — the nine de-aliased derivative fields Aᵢₖ = irfft_ext(ikₖ·pad(ψᵢ)),
#      computed ONCE per LPT order (3 pads + 9 inverse FFTs) and shared by every source term
#      (the term-by-term engine re-pads and re-transforms both factors of every term).
#   2. Fused KernelAbstractions kernels evaluate the μ₂, μ₂&C and μ₃ products point-wise on the
#      extended grid in one pass (CPU threads / CUDA), with hand-written adjoint kernels.
#   3. crop∘rfft is linear, so all source terms of an order are summed in real space and
#      transformed once (1 rfft for the longitudinal source, 3 for the transverse curl).
#
# Differentiability: every step is either a plain broadcast or has a hand `rrule` (ext_derivs,
# the three source kernels, and the existing `_rfftn`/`_crop3`/`_irfftn` adjoints), so Zygote
# composes analytic pullbacks and never traces into a kernel.
#
# Memory: fields of all lower orders are kept (27 extended-grid fields at 4th order), so the fast
# engine trades memory for speed; `compute_core(...; mode=:lean)` keeps the term-by-term engine.

export ext_derivs

@inline _col(i, k) = 3 * (i - 1) + k

"""
    ext_derivs(K, f::(res,res,res÷2+1,3)) -> A::(ext,ext,ext,9)

The nine de-aliased first-derivative fields of a Fourier vector field on the extended grid,
`A[:,:,:,3(i-1)+k] = irfft_ext(i k_k · pad(f_i))` (the `_Aext` fields of nlpt_core.jl)."""
function ext_derivs(K::NLPTKernels{T}, f::AbstractArray{Complex{T},4}) where {T}
    res, ext = K.res, K.ext
    A = similar(f, T, ext, ext, ext, 9)
    for i in 1:3
        P = _pad3(f[:, :, :, i], res, ext)
        for (k, d) in enumerate((K.dx_ext, K.dy_ext, K.dz_ext))
            A[:, :, :, _col(i, k)] .= _irfftn(d .* P, ext)
        end
    end
    return A
end

function ChainRulesCore.rrule(::typeof(ext_derivs), K::NLPTKernels{T}, f::AbstractArray{Complex{T},4}) where {T}
    A = ext_derivs(K, f)
    ext = K.ext; nh = ext ÷ 2 + 1; N = T(ext)^3
    function ext_derivs_pullback(Ā)
        Ā_ = ChainRulesCore.unthunk(Ā)
        f̄ = fill!(similar(f), zero(Complex{T}))
        Ā_ isa ChainRulesCore.AbstractZero && return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), f̄)
        wv = _halfdim_weights(f, nh, ext, T)
        for i in 1:3, k in 1:3
            _dext_accum!(f̄, K, Ā_[:, :, :, _col(i, k)], i, k, wv, N)
        end
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), f̄)
    end
    return A, ext_derivs_pullback
end

# ── fused source kernels (linear index p over the ext³ grid; fields as (ext³, 9)) ────────────────
@kernel function _k_mu2sym!(μ, @Const(A))
    p = @index(Global)
    @inbounds μ[p] = A[p, 1] * A[p, 5] + A[p, 1] * A[p, 9] + A[p, 5] * A[p, 9] -
                     A[p, 2] * A[p, 4] - A[p, 3] * A[p, 7] - A[p, 6] * A[p, 8]
end

@kernel function _k_mu2sym_adj!(Ā, @Const(P̄), @Const(A))
    p = @index(Global)
    @inbounds begin
        s = P̄[p]
        Ā[p, 1] = s * (A[p, 5] + A[p, 9]); Ā[p, 5] = s * (A[p, 1] + A[p, 9]); Ā[p, 9] = s * (A[p, 1] + A[p, 5])
        Ā[p, 2] = -s * A[p, 4]; Ā[p, 4] = -s * A[p, 2]
        Ā[p, 3] = -s * A[p, 7]; Ā[p, 7] = -s * A[p, 3]
        Ā[p, 6] = -s * A[p, 8]; Ā[p, 8] = -s * A[p, 6]
    end
end

# (term tables from nlpt_core.jl: _MU2_*, _C_*, _MU3_*; _cyc)
@kernel function _k_mu2C!(out, @Const(A), @Const(B))
    p = @index(Global)
    @inbounds begin
        m = zero(eltype(out))
        for n in 1:12
            m += _MU2_S[n] * A[p, _col(_MU2_I[n], _MU2_K[n])] * B[p, _col(_MU2_J[n], _MU2_L[n])]
        end
        out[p, 1] = m
        for c in 1:3
            s = zero(eltype(out))
            for n in 1:6
                s += _C_S[n] * A[p, _col(_cyc(_C_I[n], c - 1), _cyc(_C_K[n], c - 1))] *
                               B[p, _col(_cyc(_C_J[n], c - 1), _cyc(_C_L[n], c - 1))]
            end
            out[p, 1 + c] = s
        end
    end
end

@kernel function _k_mu2C_adj!(Ā, B̄, @Const(Ō), @Const(A), @Const(B))
    p = @index(Global)
    @inbounds begin
        for q in 1:9; Ā[p, q] = 0; B̄[p, q] = 0; end
        sm = Ō[p, 1]
        for n in 1:12
            a = _col(_MU2_I[n], _MU2_K[n]); b = _col(_MU2_J[n], _MU2_L[n])
            Ā[p, a] += _MU2_S[n] * sm * B[p, b]; B̄[p, b] += _MU2_S[n] * sm * A[p, a]
        end
        for c in 1:3
            sc = Ō[p, 1 + c]
            for n in 1:6
                a = _col(_cyc(_C_I[n], c - 1), _cyc(_C_K[n], c - 1)); b = _col(_cyc(_C_J[n], c - 1), _cyc(_C_L[n], c - 1))
                Ā[p, a] += _C_S[n] * sc * B[p, b]; B̄[p, b] += _C_S[n] * sc * A[p, a]
            end
        end
    end
end

@kernel function _k_mu3!(μ, @Const(A), @Const(B), @Const(C))
    p = @index(Global)
    @inbounds begin
        m = zero(eltype(μ))
        for n in 1:6
            m += _MU3_S[n] * A[p, _col(1, _MU3_K[n])] * B[p, _col(2, _MU3_L[n])] * C[p, _col(3, _MU3_M[n])]
        end
        μ[p] = m
    end
end

@kernel function _k_mu3_adj!(Ā, B̄, C̄, @Const(P̄), @Const(A), @Const(B), @Const(C))
    p = @index(Global)
    @inbounds begin
        for q in 1:9; Ā[p, q] = 0; B̄[p, q] = 0; C̄[p, q] = 0; end
        s0 = P̄[p]
        for n in 1:6
            a = _col(1, _MU3_K[n]); b = _col(2, _MU3_L[n]); c = _col(3, _MU3_M[n]); s = _MU3_S[n] * s0
            Ā[p, a] += s * B[p, b] * C[p, c]; B̄[p, b] += s * A[p, a] * C[p, c]; C̄[p, c] += s * A[p, a] * B[p, b]
        end
    end
end

_flat9(A) = reshape(A, :, 9)
function _run!(k, args...; n)
    be = get_backend(args[1]); k(be)(args...; ndrange=n); synchronize(be)
end

"""Real μ₂ source (symmetric, j = i−j) of the derivative fields `A = ext_derivs(K, f)`."""
function src_mu2sym(A::AbstractArray{T,4}) where {T}
    e = size(A, 1); μ = similar(A, T, e, e, e)
    _run!(_k_mu2sym!, reshape(μ, :), _flat9(A); n=e^3); μ
end
"""Real μ₂ (asymmetric) and transverse C sources: (ext,ext,ext,4) = [μ₂ C_x C_y C_z]."""
function src_mu2C(A::AbstractArray{T,4}, B::AbstractArray{T,4}) where {T}
    e = size(A, 1); o = similar(A, T, e, e, e, 4)
    _run!(_k_mu2C!, reshape(o, :, 4), _flat9(A), _flat9(B); n=e^3); o
end
"""Real μ₃ source (Levi-Civita triple product of rows 1, 2, 3 of A, B, C)."""
function src_mu3(A::AbstractArray{T,4}, B::AbstractArray{T,4}, C::AbstractArray{T,4}) where {T}
    e = size(A, 1); μ = similar(A, T, e, e, e)
    _run!(_k_mu3!, reshape(μ, :), _flat9(A), _flat9(B), _flat9(C); n=e^3); μ
end

function ChainRulesCore.rrule(::typeof(src_mu2sym), A::AbstractArray{T,4}) where {T}
    μ = src_mu2sym(A)
    function pb(μ̄)
        P = ChainRulesCore.unthunk(μ̄); e = size(A, 1)
        Ā = similar(A); _run!(_k_mu2sym_adj!, _flat9(Ā), reshape(collect(P), :), _flat9(A); n=e^3)
        (ChainRulesCore.NoTangent(), Ā)
    end
    return μ, pb
end
function ChainRulesCore.rrule(::typeof(src_mu2C), A::AbstractArray{T,4}, B::AbstractArray{T,4}) where {T}
    o = src_mu2C(A, B)
    function pb(ō)
        O = ChainRulesCore.unthunk(ō); e = size(A, 1)
        Ā = similar(A); B̄ = similar(B)
        _run!(_k_mu2C_adj!, _flat9(Ā), _flat9(B̄), reshape(collect(O), :, 4), _flat9(A), _flat9(B); n=e^3)
        (ChainRulesCore.NoTangent(), Ā, B̄)
    end
    return o, pb
end
function ChainRulesCore.rrule(::typeof(src_mu3), A::AbstractArray{T,4}, B::AbstractArray{T,4},
                              C::AbstractArray{T,4}) where {T}
    μ = src_mu3(A, B, C)
    function pb(μ̄)
        P = ChainRulesCore.unthunk(μ̄); e = size(A, 1)
        Ā = similar(A); B̄ = similar(B); C̄ = similar(C)
        _run!(_k_mu3_adj!, _flat9(Ā), _flat9(B̄), _flat9(C̄), reshape(collect(P), :), _flat9(A), _flat9(B), _flat9(C); n=e^3)
        (ChainRulesCore.NoTangent(), Ā, B̄, C̄)
    end
    return μ, pb
end

# crop∘rfft of an ext-real source (linear; hand rrules of _rfftn/_crop3 apply)
_to_k(K::NLPTKernels, s) = _crop3(_rfftn(s), K.res, K.ext)
_acc(a, b) = a === nothing ? b : a .+ b

function _compute_core_fast(fphi_ini::AbstractArray{Complex{T},3}, K::NLPTKernels{T};
                            n_order::Int, no_transverse::Bool=false) where {T}
    psi = (_psi1_fourier(K, fphi_ini),)
    G = n_order >= 2 ? (ext_derivs(K, psi[1]),) : ()
    for i in 2:n_order
        SL = nothing; SC = nothing
        den = (i + 3 / 2) * (i - 1)
        if iseven(i)
            h = i ÷ 2
            SL = _acc(SL, T(((3 - i) / 2 - 2h^2) / den) .* src_mu2sym(G[h]))
        end
        if i > 2
            for j in 1:((i + 1) ÷ 2 - 1)
                imj = i - j
                o = src_mu2C(G[j], G[imj])
                SL = _acc(SL, T(((3 - i) / 2 - j^2 - imj^2) / den) .* o[:, :, :, 1])
                no_transverse || (SC = _acc(SC, T(1 - 2j / i) .* o[:, :, :, 2:4]))
            end
            for k in 1:(i - 2), l in 1:(i - k - 1)
                m = i - k - l
                SL = _acc(SL, T(((3 - i) / 2 - k^2 - l^2 - m^2) / den) .* src_mu3(G[k], G[l], G[m]))
            end
        end
        fL = _to_k(K, SL)
        fT = (i > 2 && !no_transverse) ?
             cat(_to_k(K, SC[:, :, :, 1]), _to_k(K, SC[:, :, :, 2]), _to_k(K, SC[:, :, :, 3]); dims=4) : nothing
        psi = (psi..., _assemble_psi(K, fL, fT))
        i <= n_order - 1 && (G = (G..., ext_derivs(K, psi[i])))
    end
    out = Dict{String,AbstractArray{T,4}}()
    for i in 1:n_order
        out["psi_$i"] = _vec_irfftn(K, psi[i])
    end
    return out
end

function _compute_core_exact_fast(fphi_ini::AbstractArray{Complex{T},3}, K::NLPTKernels{T};
                                  n_order::Int) where {T}
    psi1 = _psi1_fourier(K, fphi_ini)
    out = Dict{String,AbstractArray{T,4}}()
    out["psi_1"] = _vec_irfftn(K, psi1)
    n_order >= 2 || return out
    G1 = ext_derivs(K, psi1)
    psi2 = _assemble_psi(K, _to_k(K, src_mu2sym(G1)), nothing)
    out["psi_2_ex"] = _vec_irfftn(K, psi2)
    n_order >= 3 || return out
    G2 = ext_derivs(K, psi2)
    o = src_mu2C(G1, G2)
    fL3a = (-one(T)) .* _to_k(K, src_mu3(G1, G1, G1))
    fL3b = (-T(0.5)) .* _to_k(K, o[:, :, :, 1])
    fT3c = (-one(T)) .* cat(_to_k(K, o[:, :, :, 2]), _to_k(K, o[:, :, :, 3]), _to_k(K, o[:, :, :, 4]); dims=4)
    out["psi_3a_ex"] = _vec_irfftn(K, _assemble_psi(K, fL3a, nothing))
    out["psi_3b_ex"] = _vec_irfftn(K, _assemble_psi(K, fL3b, nothing))
    out["psi_3c_ex"] = _vec_irfftn(K, _assemble_psi_transverse(K, fT3c))
    return out
end
