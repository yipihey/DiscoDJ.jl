# ── Fixed-mass statistics of the sheet: F(R, M), its derivatives, and the real-space mass budget ──
#
# For spheres of radius R centred on the nodes of an ng³ mesh, μ_R(x) = M_R(x)/(ρ̄V_R) = 1 + δ_R(x).
# F(R, M) = P(μ_R ≥ M/ρ̄V_R) is the continuum version of the kNN-CDF (fixed M, varying R) and of
# counts-in-cells (fixed R, varying M).  Its derivatives follow from instantaneous fields, without
# differencing in time:
#   ∂_R F = p_R(μ) E[∂_R μ_R | μ],          ∂_R μ_R = 3(s_R − μ_R)/R   (s_R: shell-mean density)
#   ∂_D F = p_R(μ) E[μ̇_R | μ],             μ̇_R = −⟨∇·p⟩_R
#   ∂²_D F = p_R E[μ̈_R | μ] − ∂_μ(p_R E[μ̇²_R | μ]),
#   μ̈_R = ⟨∂_i∂_j Π_ij⟩_R − ⟨∇·(ρ a)⟩_R                                       (exact kinematics)
# with time D (EdS: D = a), v = dx/dD and a = d²x/dD², ρ̄ = 1, ⟨·⟩_R the top-hat average.
#
# Two sheet estimators are combined:
#  * conservative (mass, momentum, stress, ρa): CIC deposit of tracers at fixed barycentric positions
#    in every tetrahedron (`sheet_tracer_deposit_periodic`).  The tracers follow the P1 sheet
#    kinematics exactly, so the budget above closes identically; point-sampling the piecewise-
#    constant tetrahedron density does NOT conserve mass (nodes jump between tetrahedra in time).
#  * pointwise, stream-resolved (`sheet_moments_mesh_periodic`): mean velocity v̄ and dispersion σ,
#    σ ≡ 0 in single-stream regions.  Π splits as  ρv̄v̄ (bulk) + ρσ (multistream dispersion)
#    + a coarse-graining remainder (the deposit kernel's spread of a single stream's velocities).
# For PM, a = (3/2D²) g − (3/2D) v with ∇·g = −δ; grav = ⟨∇·(ρ g)⟩_R uses the deposited density.
#
# Per (epoch, R) the device reduces each term X to sums over log-μ bins (conditional moments) and
# global moments; a strided sub-lattice keeps the per-point values for field-level UV−IR work.

export FixedMassPlan, fixedmass_snapshot, FM_TERMS, FM_CHANNELS, lpt_acc!

# histogram columns (sums over the nodes of each log-μ bin); flux = ⟨∇·p⟩_R = −μ̇_R, stress = ⟨∂∂Π⟩_R,
# bulk = ⟨∂∂(ρv̄v̄)⟩_R, disp = ⟨∂∂(ρσ)⟩_R, grav = ⟨∇·(ρg)⟩_R, acc = ⟨∇·(ρa)⟩_R  (μ̈_R = stress − acc)
const FM_TERMS = ("count", "mu", "shell", "flux", "flux2", "stress", "bulk", "disp", "grav", "acc")
# per-point / moment channels
const FM_CHANNELS = ("mu", "shell", "flux", "stress", "bulk", "disp", "grav", "acc")

struct FixedMassPlan{AR}
    ng::Int
    L::Float64
    R::Vector{Float64}
    nbins::Int
    stride::Int
    sub::Vector{Int}          # radius indices whose per-point values are kept on the sub-lattice
    kx::AR; ky::AR; kz::AR    # derivative wavenumbers (Nyquist zeroed), DJ._rfftn layout
    kmag::AR                  # |k| on the half grid (window argument)
    Tacc::DataType
    level::Int                # tracer level of the conservative deposit (binomial(level+3,3) per tet)
    reclaim::Function         # called between stages (return pooled device memory)
end

function FixedMassPlan(ng::Int, L::Real, R; nbins::Int=256, stride::Int=8, sub=1:length(R),
                       to_device=identity, T::Type{<:AbstractFloat}=Float32, level::Int=1,
                       reclaim::Function=() -> nothing)
    kf = DJ._kfull(ng, Float64(L)); kh = DJ._khalf(ng, Float64(L))
    kfd = copy(kf); kfd[ng ÷ 2 + 1] = 0; khd = copy(kh); khd[end] = 0     # odd derivatives: drop Nyquist
    km = sqrt.(reshape(kf, :, 1, 1) .^ 2 .+ reshape(kf, 1, :, 1) .^ 2 .+ reshape(kh, 1, 1, :) .^ 2)
    d(x) = to_device(T.(x))
    FixedMassPlan(ng, Float64(L), Float64.(collect(R)), nbins, stride, collect(Int, sub),
                  d(reshape(kfd, :, 1, 1)), d(reshape(kfd, 1, :, 1)), d(reshape(khd, 1, 1, :)), d(km), T, level, reclaim)
end

_wth_t(x::T) where {T} = x < T(1e-3) ? one(T) - x^2 / 10 : 3 * (sin(x) - x * cos(x)) / x^3
_sinc_t(x::T) where {T} = x < T(1e-3) ? one(T) - x^2 / 6 : sin(x) / x

# Σ over the cells of each bin of X^pw (pw = 0: count).  One work item per (i, j) column walking the
# slowest axis k (a warp reads consecutive i: coalesced); successive k often share a bin, so partial
# sums are flushed only on bin changes, into one of NCOPY histogram copies chosen by i (narrow log-μ
# bins change almost every cell, and a single copy serialises ~10⁸ atomics on a few hundred words).
const NCOPY = 32
@kernel function _k_binsum!(H, @Const(bin), @Const(X), ng::Int, col::Int, pw::Int)
    i, j = @index(Global, NTuple)
    @inbounds begin
        cp = (i - 1) % NCOPY + 1
        cur = bin[i, j, 1]; s = 0.0
        for k in 1:ng
            b = bin[i, j, k]
            if b != cur
                KernelAbstractions.@atomic H[cur, col, cp] += s
                cur = b; s = 0.0
            end
            x = Float64(X[i, j, k])
            s += pw == 0 ? 1.0 : (pw == 1 ? x : x * x)
        end
        KernelAbstractions.@atomic H[cur, col, cp] += s
    end
end

function _binsum!(H, bin, X, col, pw)
    ng = size(bin, 1); be = get_backend(bin)
    _k_binsum!(be)(H, bin, X, ng, col, pw; ndrange=(ng, ng)); synchronize(be)
end

"""
    lpt_acc!(A, sh, D, n) -> A

Kinematic acceleration d²Ψ/dD² = Σ_{m≤n} m(m−1) D^{m−2} ψ_m of the order-n EdS LPT map (`sh` from
`compute_core`)."""
function lpt_acc!(A, sh, D, n)
    fill!(A, zero(eltype(A)))
    for m in 2:n
        A .+= eltype(A)(m * (m - 1) * D^(m - 2)) .* sh["psi_$m"]
    end
    return A
end

"""
    fixedmass_snapshot(P, Ψ, V; acc=nothing) -> NamedTuple (host arrays)

Ψ (unwrapped displacement), V = dΨ/dD and optionally acc = d²Ψ/dD² on the (n,n,n,3) Lagrangian
lattice.  Returns
  edges  (nR, nbins+1)          log μ bin edges per radius
  hist   (nR, nbins, 10)        per-bin sums of FM_TERMS
  mom    (nR, 4)                ⟨δ_R^p⟩, p = 1..4 (volume weighted over the mesh nodes)
  tmom   (nR, 8, 3)             ⟨X⟩, ⟨X δ_R⟩, ⟨X²⟩ for X in FM_CHANNELS
  sub    (nsub, 8, Nc) Float32  FM_CHANNELS at the sub-lattice nodes (stride) for radii P.sub
  scal   Dict                   sheet diagnostics (mass, multistream volume/mass fractions, ⟨tr σ⟩) and
                                stage timings t_* (s).  `reclaim()` is called between stages (e.g. to
                                return pooled device memory: `() -> (GC.gc(false); CUDA.reclaim())`)."""
function fixedmass_snapshot(P::FixedMassPlan, Ψ::AbstractArray{T,4}, V::AbstractArray{T,4}; acc=nothing,
                            reclaim::Function=P.reclaim) where {T}
    ng = P.ng; L = P.L; TA = P.Tacc; be = get_backend(Ψ)
    tim = Dict{String,Float64}(); t0 = Ref(time())
    tick(name) = (synchronize(be); t = time(); tim["t_" * name] = t - t0[]; t0[] = t)
    m = sheet_moments_mesh_periodic(Ψ, V, L, ng; Tacc=TA)          # pointwise: v̄, σ (intensive)
    ns = m.nstream; vb = m.vbar; σ = m.sigma
    tick("moments")
    dep(mode; G=nothing, mesh=nothing) =
        sheet_tracer_deposit_periodic(Ψ, V, L, ng, mode; acc=acc, G=G, level=P.level, Tacc=TA, mesh=mesh)
    ρ = dep(:rho)                                                   # conservative mass (PCS tracers)
    # ── sheet diagnostics ──
    N3 = ng^3; msk = ns .> 1
    ρsum = sum(Float64, ρ); trσ = view(σ, :, :, :, 1) .+ view(σ, :, :, :, 2) .+ view(σ, :, :, :, 3)
    scal = Dict{String,Float64}(
        "mean_rho" => ρsum / N3,
        "mean_rho_point" => sum(Float64, m.density) / N3,
        "ms_volume_frac" => count(msk) / N3,
        "ms_mass_frac" => sum(Float64, ρ .* msk) / ρsum,
        "mean_trsigma_mass" => sum(Float64, ρ .* trσ) / ρsum,
        "max_nstream" => Float64(maximum(ns)))
    trσ = nothing; msk = nothing; ns = nothing; m = nothing
    tick("diagnostics")
    # ── sources: divergences carried by the deposit kernel (exact budget), then one FFT each ──
    F(x) = DJ._rfftn(x)
    ρh = F(ρ); ρ = nothing
    buf = KernelAbstractions.zeros(get_backend(Ψ), TA, ng, ng, ng)
    flux = F(dep(:flux; mesh=buf))
    stress = F(dep(:stress; mesh=buf))
    bulk = F(dep(:bulk; G=vb, mesh=buf)); vb = nothing
    disp = F(dep(:disp; G=σ, mesh=buf)); σ = nothing
    tick("deposits"); reclaim()
    # g = −∇φ, ∇²φ = δ  ⇒  ĝ_i = i k_i δ̂/k²; gathered to the tracers, ∇·(ρg) = Σ m g·∇W
    kk = (P.kx, P.ky, P.kz)
    ik2 = map(x -> x > 0 ? inv(x * x) : zero(x), P.kmag)
    g = similar(buf, TA, ng, ng, ng, 3)
    for i in 1:3
        g[:, :, :, i] .= DJ._irfftn(im .* kk[i] .* ρh .* ik2, ng)
    end
    ik2 = nothing
    grav = F(dep(:gdiv; G=g, mesh=buf)); g = nothing
    accf = acc === nothing ? nothing : F(dep(:acc; mesh=buf))
    buf = nothing
    tick("gravity_acc"); reclaim()
    # ── radius sweep ──
    nR = length(P.R); nb = P.nbins; s = P.stride; Nc = length(1:s:ng)^3
    edges = zeros(nR, nb + 1); hist = zeros(nR, nb, length(FM_TERMS))
    mom = zeros(nR, 4); tmom = fill(NaN, nR, length(FM_CHANNELS), 3)
    sub = fill(NaN32, length(P.sub), length(FM_CHANNELS), Nc)
    bin = KernelAbstractions.zeros(be, Int32, ng, ng, ng)
    for (r, R) in enumerate(P.R)
        W = _wth_t.(P.kmag .* T(R)); Sh = _sinc_t.(P.kmag .* T(R))
        μ = DJ._irfftn(ρh .* W, ng)
        lo, hi = extrema(μ); lo = log(max(Float64(lo), 1e-6)); hi = log(Float64(hi)) + 1e-9
        edges[r, :] .= range(lo, hi, length=nb + 1)
        a = T(nb / (hi - lo)); l0 = T(lo)
        bin .= clamp.(floor.(Int32, (log.(max.(μ, T(1e-6))) .- l0) .* a) .+ Int32(1), Int32(1), Int32(nb))
        H = KernelAbstractions.zeros(be, Float64, nb, length(FM_TERMS), NCOPY)
        _binsum!(H, bin, μ, 1, 0); _binsum!(H, bin, μ, 2, 1)
        for p in 1:4
            mom[r, p] = sum(x -> (Float64(x) - 1)^p, μ) / N3
        end
        isub = findfirst(==(r), P.sub)
        chan(c, X) = begin
            tmom[r, c, 1] = sum(Float64, X) / N3
            tmom[r, c, 2] = sum(Float64, X .* (μ .- one(T))) / N3
            tmom[r, c, 3] = sum(x -> Float64(x)^2, X) / N3
            isub === nothing || (sub[isub, c, :] .= vec(Array(X[1:s:ng, 1:s:ng, 1:s:ng])))
        end
        chan(1, μ)
        X = DJ._irfftn(ρh .* Sh, ng); _binsum!(H, bin, X, 3, 1); chan(2, X)
        X = DJ._irfftn(flux .* W, ng); _binsum!(H, bin, X, 4, 1); _binsum!(H, bin, X, 5, 2); chan(3, X)
        X = DJ._irfftn(stress .* W, ng); _binsum!(H, bin, X, 6, 1); chan(4, X)
        X = DJ._irfftn(bulk .* W, ng); _binsum!(H, bin, X, 7, 1); chan(5, X)
        X = DJ._irfftn(disp .* W, ng); _binsum!(H, bin, X, 8, 1); chan(6, X)
        X = DJ._irfftn(grav .* W, ng); _binsum!(H, bin, X, 9, 1); chan(7, X)
        if accf !== nothing
            X = DJ._irfftn(accf .* W, ng); _binsum!(H, bin, X, 10, 1); chan(8, X)
        end
        Hh = dropdims(sum(Array(H); dims=3); dims=3)
        for c in 1:9; hist[r, :, c] .= view(Hh, :, c); end
        hist[r, :, 10] .= accf === nothing ? NaN : view(Hh, :, 10)
        X = nothing; μ = nothing
    end
    tick("radius_sweep"); reclaim()
    merge!(scal, tim)
    return (R=P.R, edges=edges, hist=hist, mom=mom, tmom=tmom, sub=sub, sub_R=P.R[P.sub], stride=s, scal=scal)
end
