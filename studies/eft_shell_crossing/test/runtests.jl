# Tests of the paired-IC construction, the measurement kernels and the estimators (CPU, small N).
#   julia --project=. test/runtests.jl
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, DiscoDJNative, Test, Random, FFTW, Statistics, LinearAlgebra
const DJ = DiscoDJNative

pk = planck_pk()
L = 200.0; N = 32

@testset "EFTShell" begin

@testset "paired ICs: shared long modes, band only in UV" begin
    s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45, seed_L=3, seed_S=7)
    fu = make_fphi(s, pk, :UV); fi = make_fphi(s, pk, :IR)
    kk = kmag_half(N, L)
    lo = kk .< s.k_split; band = (kk .>= s.k_split) .& (kk .< s.k_uv)
    @test fu[lo] == fi[lo]                                   # bitwise identical long modes
    @test all(iszero, fi[.!lo]) && all(iszero, fu[kk .>= s.k_uv])
    @test count(!iszero, fu[band]) == count(band) - 0        # band populated
    # long-mode sign flip and band amplitude
    @test make_fphi(PairSpec(s; long_sign=-1), pk, :IR) == -fi
    fb = make_fphi(PairSpec(s; band_amp=2.0), pk, :UV)
    @test fb[band] ≈ 2 .* fu[band] && fb[lo] == fu[lo]
    @test make_fphi(PairSpec(s; band_amp=0.0), pk, :UV) == fi   # null pair
end

@testset "exact embedding into a finer particle grid" begin
    s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45, seed_L=3, seed_S=7)
    f1 = make_fphi(s, pk, :UV; T=Float64)
    f2 = make_fphi(PairSpec(s; N=2N), pk, :UV; T=Float64)
    x1 = irfft(f1, N, [3, 1, 2]); x2 = irfft(f2, 2N, [3, 1, 2])
    @test maximum(abs.(x2[1:2:end, 1:2:end, 1:2:end] .- x1)) < 1e-10 * maximum(abs.(x1))
    ms1 = ModeSet(N, L, 0.5); ms2 = ModeSet(2N, L, 0.5)
    @test ms1.nvec == ms2.nvec
    @test field_modes(ms1, x1) ≈ field_modes(ms2, x2) rtol = 1e-9
end

@testset "null pair gives Δψ = 0 at every LPT order" begin
    s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45, band_amp=0.0)
    K = nlpt_kernels(N, L; T=Float64)
    a = compute_core(make_fphi(s, pk, :UV; T=Float64), K; n_order=4)
    b = compute_core(make_fphi(s, pk, :IR; T=Float64), K; n_order=4)
    @test all(a["psi_$n"] == b["psi_$n"] for n in 1:4)
end

@testset "even LPT orders are even in the long modes, odd orders odd (IR member)" begin
    s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45)
    K = nlpt_kernels(N, L; T=Float64)
    p = compute_core(make_fphi(s, pk, :IR; T=Float64), K; n_order=4)
    m = compute_core(make_fphi(PairSpec(s; long_sign=-1), pk, :IR; T=Float64), K; n_order=4)
    for n in 1:4
        @test m["psi_$n"] ≈ (-1)^n .* p["psi_$n"] rtol = 1e-10
    end
end

@testset "inverted tetrahedra: identity, plane-wave Zel'dovich past crossing" begin
    Ψ0 = zeros(Float64, N, N, N, 3)
    @test inverted_fraction(Ψ0, L) == 0
    for A in (0.5, 1.5, 3.0)                      # x = q − (A/k) sin(kq): crossed where A cos(kq) > 1
        k = 2π / L * 2
        q = [(i - 1) * L / N for i in 1:N]
        Ψ = zeros(Float64, N, N, N, 3); Ψ[:, :, :, 1] .= reshape(-A / k .* sin.(k .* q), N, 1, 1)
        f = inverted_fraction(Ψ, L)
        # each cube's tets are inverted iff the finite difference of x along x is negative
        dx = [(L / N) - A / k * (sin(k * (q[i] + L / N)) - sin(k * q[i])) for i in 1:N]
        @test f ≈ count(<(0), dx) / N
        A > 1 && @test isapprox(f, acos(1 / A) / π; atol=2 / N)
        A < 1 && @test f == 0
    end
end

@testset "estimator recovers an injected −C₂k² + C₄k⁴ response and the stochastic power" begin
    rng = MersenneTwister(11)
    Lb = 1000.0; ms = ModeSet(64, Lb, 0.18)
    k = ms.kmag; M = length(ms)
    C2, C4 = 3.0, 20.0
    edges = collect(0.5:1:28) .* (2π / Lb)
    accs = BinAcc[]; pe = []
    for r in 1:16
        x = randn(rng, ComplexF64, M) .* sqrt.(1e3 .* k .^ -2.5)            # ψ_L∥ coefficients
        ε = randn(rng, ComplexF64, M) .* sqrt.(50.0 .* k .^ 2) .* 1e-3      # P_ε ∝ k² (displacement)
        y = (-C2 .* k .^ 2 .+ C4 .* k .^ 4) .* x .+ ε
        push!(accs, accum!(y, x, k, edges))
        push!(pe, binned_power(y .- (-C2 .* k .^ 2 .+ C4 .* k .^ 4) .* x, k, edges, Lb))
    end
    est(A) = (r = response(sum(A)); r.R)
    R, σR = jackknife(est, accs)
    r = response(sum(accs))
    f1 = fit_counterterm(r.k2, r.k4, R, σR; kmax=0.05, order=1)
    f2 = fit_counterterm(r.k2, r.k4, R, σR; kmax=0.17, order=2)
    @test abs(f1.C2 - C2) < 0.2 * C2           # leading-only fit, short range: small k⁴ bias
    @test abs(f2.C2 - C2) < 4 * f2.σC2 + 0.02 * C2
    @test abs(f2.C4 - C4) < 4 * f2.σC4 + 0.05 * C4
    @test f2.chi2 / f2.dof < 2.5
    Pε = mean(p.P for p in pe); ke = pe[1].k
    good = isfinite.(Pε) .& (ke .> 0)
    slope = cov(log.(ke[good]), log.(Pε[good])) / var(log.(ke[good]))
    @test abs(slope - 2) < 0.15                  # P_ε ∝ k² recovered
end

@testset "Gaussian-filter pair: spurious IC-level k² response; sharp split: none" begin
    s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45, seed_L=5)
    R1, R2 = 4.0, 8.0
    f1 = make_fphi_gauss(s, pk, R1; T=Float64); f2 = make_fphi_gauss(s, pk, R2; T=Float64)
    K = nlpt_kernels(N, L; T=Float64)
    ms = ModeSet(N, L, 0.2)
    ψ1 = vector_modes(ms, compute_core(f1, K; n_order=1)["psi_1"])
    ψ2 = vector_modes(ms, compute_core(f2, K; n_order=1)["psi_1"])
    y = longitudinal(ψ1 .- ψ2, ms.kvec); x = longitudinal(ψ2, ms.kvec)
    edges = collect(0.5:1:6.5) .* (2π / L)
    r = response(accum!(y, x, ms.kmag, edges))
    exact = exp.(-r.k2 .* (R1^2 - R2^2) ./ 2) .- 1             # per-mode ratio is exact
    @test maximum(abs.(r.R .- exact) ./ abs.(exact)) < 0.1
    fit = fit_counterterm(r.k2, r.k4, r.R, fill(1.0, length(r.R)); kmax=0.1)
    @test isapprox(fit.C2, (R1^2 - R2^2) / 2; rtol=0.25)          # a "C₂" of −24 (Mpc/h)² from the filters alone
    # sharp split: 1LPT long modes identical → exactly zero
    a = compute_core(make_fphi(s, pk, :UV; T=Float64), K; n_order=1)["psi_1"]
    b = compute_core(make_fphi(s, pk, :IR; T=Float64), K; n_order=1)["psi_1"]
    lo = ms.kmag .< s.k_split
    @test maximum(abs.(vector_modes(ms, a)[lo, :] .- vector_modes(ms, b)[lo, :])) < 1e-12
end


@testset "counts in spheres: deposited density → linear field for a small displacement" begin
    # IR member (|k| < 0.25, well below the N = 32 Nyquist frequency 0.5): δ_R and ∇²δ_R of the deposited
    # density reproduce D × the linear field.  (∇² of a sharp top-hat average is a surface term whose k²W(kR)
    # tail does not decay, so it is only meaningful for band-limited fields — hence the IR member.)
    s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45, seed_L=3, seed_S=7)
    f = make_fphi(s, pk, :IR; T=Float64)
    K = nlpt_kernels(N, L; T=Float64)
    ψ1 = compute_core(f, K; n_order=1)["psi_1"]
    D = 1e-3
    P = CiSPlan(N, L, [30.0, 50.0]; stride=4, T=Float64)
    q = lattice_q(ψ1, N, L)
    _, cs = density_modes(ModeSet(N, L, 0.2), D .* ψ1, q; mesh=zeros(N, N, N), X=similar(ψ1), cis=P)
    lin = cis_linear(P, f)
    for r in 1:2
        a = Float64.(cs[r, 1, :]); b = D .* Float64.(lin[r, 1, :])
        @test cor(a, b) > 0.995
        @test isapprox(std(a), std(b); rtol=0.01)
        # ∇²δ_R: k²-weighted, so the deposit window of a near-lattice distribution (∝ sinc, not the sinc²
        # that is deconvolved) shows up at the (kΔ/2 ≈ 0.8) level of this coarse test grid: ≤ 20 % here,
        # < 1 % for the production grid at k ≤ 0.2
        a = Float64.(cs[r, 2, :]); b = D .* Float64.(lin[r, 2, :])
        @test cor(a, b) > 0.8
        @test isapprox(std(a), std(b); rtol=0.25)
    end
    # identical centres for every member: UV − IR of the linear fields is the band field only
    lu = cis_linear(P, make_fphi(s, pk, :UV; T=Float64))
    @test 0 < std(lu[1, 1, :] .- lin[1, 1, :]) < std(lin[1, 1, :])
end


@testset "redshift-space sheet crosses before the real-space sheet (Zel'dovich plane wave along the LOS)" begin
    # EdS: u = dΨ/dln a = D Π = Ψ for the growing mode, so s_z = q_z + 2Ψ_z; amplitude A in real space is 2A
    # in redshift space: crossed in s for A > 1/2 while uncrossed in x for A < 1.
    k = 2π / L * 2; q = [(i - 1) * L / N for i in 1:N]
    for A in (0.4, 0.7, 0.9)
        Ψ = zeros(Float64, N, N, N, 3); Ψ[:, :, :, 3] .= reshape(-A / k .* sin.(k .* q), 1, 1, N)
        Ψs = copy(Ψ); Ψs[:, :, :, 3] .*= 2
        @test inverted_fraction(Ψ, L) == 0
        fs = inverted_fraction(Ψs, L)
        dz = [(L / N) - 2A / k * (sin(k * (q[i] + L / N)) - sin(k * q[i])) for i in 1:N]   # exact lattice criterion
        @test fs ≈ count(<(0), dz) / N
        A > 0.5 ? (@test isapprox(fs, acos(1 / (2A)) / π; atol=3 / N)) : (@test fs == 0)
    end
end


@testset "kinetic stress: zero for bulk flow, smooth-flow artifact removed, dispersion in crossed cells" begin
    q = lattice_q(zeros(Float64, N, N, N, 3), N, L)
    Ψ0 = zeros(Float64, N, N, N, 3); U = zeros(Float64, N, N, N, 3); U[:, :, :, 1] .= 3.7; U[:, :, :, 3] .= -1.2
    r = stress_modes(ModeSet(8, L, 0.12), Ψ0, U, q, L, 8)
    @test maximum(abs.(r.modes)) < 1e-9 && r.mass_frac == 0 && abs(r.mean_trace) < 1e-9
    qq = [(i - 1) * L / N for i in 1:N]
    # smooth single stream (wavelength = box, 64 particles per cell): the per-cell linear-flow fit removes the
    # velocity-gradient "dispersion" that a plain cell variance reports
    k1 = 2π / L
    Ψ = zeros(Float64, N, N, N, 3); Ψ[:, :, :, 3] .= reshape(-0.1 / k1 .* sin.(k1 .* qq), 1, 1, N)
    r8 = stress_modes(ModeSet(8, L, 0.06), Ψ, copy(Ψ), q, L, 8); r4 = stress_modes(ModeSet(4, L, 0.06), Ψ, copy(Ψ), q, L, 4)
    @test r8.mass_frac == 0 && r8.mean_trace < 0.02 * r8.mean_trace_raw
    @test r8.mean_trace < 0.1 * r4.mean_trace          # remainder is curvature, falling faster than Δ³
    # crossed plane wave: multistream cells flagged, and they carry the dispersion
    k = 2π / L * 2
    Ψ = zeros(Float64, N, N, N, 3); Ψ[:, :, :, 3] .= reshape(-1.5 / k .* sin.(k .* qq), 1, 1, N)
    r = stress_modes(ModeSet(8, L, 0.12), Ψ, copy(Ψ), q, L, 8)
    @test r.mass_frac > 0.3 && r.sigma2_ms > r.mean_trace > 0
end


@testset "full-range power spectra: linear density and displacement for a small Zel'dovich displacement" begin
    s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45, seed_L=3, seed_S=7)
    f = make_fphi(s, pk, :UV; T=Float64)
    ψ1 = compute_core(f, nlpt_kernels(N, L; T=Float64); n_order=1)["psi_1"]
    D = 1e-3; q = lattice_q(ψ1, N, L)
    P = PkPlan(L, N; nbins=8, kmax=0.4)
    k, Pd, Pp = pk_full(P, D .* ψ1, q)
    # reference: the same binning of the exact linear field (δ̂ = −k² φ̂, Σ|Ψ̂|² = |δ̂|²/k²)
    kk = kmag_half(N, L); b, w, _ = EFTShell._pk_bins(N, L, P.edges)
    ref = zeros(8); refp = zeros(8); cnt = zeros(8)
    for i in eachindex(kk)
        b[i] == 0 && continue
        v = abs2(kk[i]^2 * f[i]) * D^2
        ref[b[i]] += w[i] * v; refp[b[i]] += w[i] * v / kk[i]^2; cnt[b[i]] += w[i]
    end
    ok = cnt .> 0
    # density: CIC deposit of a near-lattice distribution has an effective window ∝ sinc (not the deconvolved
    # sinc²), so only check where kΔ/2 ≤ 0.5 (production: Δ = 0.98 Mpc/h, i.e. k ≤ 1)
    lo = ok .& (k .* (L / N) ./ 2 .<= 0.5)
    @test maximum(abs.(Pd[lo] ./ (L^3 .* ref[lo] ./ cnt[lo] ./ N^6) .- 1)) < 0.1
    @test maximum(abs.(Pp[ok] ./ (L^3 .* refp[ok] ./ cnt[ok] ./ N^6) .- 1)) < 1e-6
end

end
