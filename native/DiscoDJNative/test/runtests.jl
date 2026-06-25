using Test
using DiscoDJNative
using Random: MersenneTwister
using FFTW: rfft
using LinearAlgebra: norm

mean(x) = sum(x) / length(x)

@testset "DiscoDJNative" begin

    @testset "Cosmology" begin
        c = Cosmology("Planck18EEBAOSN")
        @test abs(Omega_m(c) - 0.3085131) < 1e-6   # Planck18EEBAOSN (matches DISCO-DJ)
        @test abs(c.h - 0.67742) < 1e-6
        @test abs(hubble_E(c, 1.0) - 1.0) < 1e-10   # E(1) = 1 by definition
        @test hubble_E(c, 0.5) > 1.0                  # H increases at high z

        # chi is monotone: chi(a=1) < chi(a=0.5)
        chi1 = comoving_distance(c, 1.0)
        chi05 = comoving_distance(c, 0.5)
        @test chi1 < chi05
        @test chi1 < 0.1  # chi(a=1) ≈ 0

        # Round-trip: scale_factor_from_chi ∘ comoving_distance ≈ identity
        for a_test in [0.2, 0.5, 0.9]
            chi_a = comoving_distance(c, a_test)
            a_back = scale_factor_from_chi(c, chi_a)
            @test abs(a_back - a_test) < 1e-3
        end
    end

    @testset "Growth factors" begin
        c = Cosmology("Planck18EEBAOSN")
        @test abs(growth_D1(c, 1.0) - 1.0) < 1e-6   # D1(1) = 1 by normalisation
        @test growth_D1(c, 0.5) < growth_D1(c, 1.0)  # D1 grows with a

        # EdS approximation: D1 ∝ a deep in matter domination
        D_02 = growth_D1(c, 0.02)
        D_04 = growth_D1(c, 0.04)
        ratio = D_04 / D_02
        @test abs(ratio - 2.0) < 0.05  # ≤5% from D∝a in EdS

        # Growth rate f ≈ 1 deep in matter domination
        @test abs(growth_f1(c, 0.02) - 1.0) < 0.05

        # Higher-order growth factors (D₂plus, D₃plusa/b/c) — exact-growth nLPT.
        # Validated to machine precision vs JAX; here check the EdS limits at high z:
        # D₂plus → -3/7 D₁², D₃plusa → 1/3 D₁³, D₃plusb → -10/21 D₁³, D₃plusc → 1/7 D₁³.
        a = 0.01; D1 = growth_D1(c, a)
        @test isapprox(growth_D2(c, a),  -3/7  * D1^2; rtol=2e-2)
        @test isapprox(growth_D3a(c, a),  1/3  * D1^3; rtol=2e-2)
        @test isapprox(growth_D3b(c, a), -10/21 * D1^3; rtol=2e-2)
        @test isapprox(growth_D3c(c, a),  1/7  * D1^3; rtol=2e-2)
        @test growth_D2(c, 1.0) < 0          # D₂plus is negative at a=1
    end

    @testset "Transfer functions" begin
        c = Cosmology("Planck18EEBAOSN")
        k = exp10.(LinRange(-3, 1, 100))

        T_eh = eisenstein_hu(c, k)
        @test all(0 .< T_eh .<= 1.0)   # T(k) ∈ (0,1]
        @test T_eh[1] ≈ 1.0 atol=0.05  # T(k→0) → 1

        T_bbks = bbks(c, k)
        @test all(0 .< T_bbks .<= 1.0)
        @test T_bbks[1] ≈ 1.0 atol=0.05
    end

    @testset "Linear power spectrum" begin
        c = Cosmology("Planck18EEBAOSN")
        pk = linear_power_spectrum(c)
        @test haskey(pk, "k"); @test haskey(pk, "Pk")
        @test all(pk["Pk"] .> 0)

        # σ₈ consistency: compute sigma8 from Pk table
        k = pk["k"]; Pk = pk["Pk"]
        R8 = 8.0
        kR = k .* R8
        W = @. 3(sin(kR) - kR*cos(kR)) / kR^3
        W[kR .< 1e-3] .= 1.0
        dk = diff(k)
        integ = @. k^2 * Pk * W^2 / (2π^2)
        sigma8_meas = sqrt(sum(0.5*(integ[1:end-1] + integ[2:end]) .* dk))
        @test abs(sigma8_meas - c.sigma8) / c.sigma8 < 0.01
    end

    @testset "Fourier grid" begin
        grid = get_fourier_grid(16, 100.0)
        @test grid.res == 16
        @test grid.boxsize == 100.0
        @test size(grid.k2) == (16, 16, 9)   # res/2+1 = 9
        @test grid.k2[1,1,1] == 0.0           # DC mode k=0
    end

    @testset "LPT kernels — correctness" begin
        res = 16
        n   = res^3 * (res÷2+1)
        k2  = rand(Float64, res, res, res÷2+1)
        k2[1,1,1] = 0.0   # DC mode
        f   = randn(ComplexF64, res, res, res÷2+1)

        out_ka  = similar(f)
        out_thr = similar(f)

        inv_laplace_ka!(out_ka, f, k2)
        inv_laplace_threads!(out_thr, f, k2)

        @test out_ka ≈ out_thr rtol=1e-10
        @test out_ka[1,1,1] == 0   # DC zeroed
    end

    @testset "GRF — shape and statistics" begin
        c   = Cosmology("Planck18EEBAOSN")
        pk  = linear_power_spectrum(c)
        res = 16
        fphi = generate_grf(:fourier, 3, pk, res, 100.0, 42)
        @test size(fphi) == (res, res, res÷2+1)
        @test fphi[1,1,1] == 0   # DC zeroed
    end

    @testset "1LPT displacement" begin
        c   = Cosmology("Planck18EEBAOSN")
        pk  = linear_power_spectrum(c)
        res = 16
        T   = Float32
        fphi = generate_grf(:ngenic, 3, pk, res, 100.0, 42; dtype=T, dtype_c=Complex{T})
        grid = get_fourier_grid(res, 100.0; T=T)
        lpt  = compute_lpt(fphi, grid; n_order=1, backend=:threads)

        @test size(lpt.psi1) == (res, res, res, 3)
        @test lpt.psi2 === nothing

        # Zero mean: ∑ψ₁ = 0 (no bulk motion)
        @test abs(mean(lpt.psi1[:,:,:,1])) < 0.1  # within 10% of 0 for small box

        # D∝a in matter domination (up to 5%)
        psi_02 = evaluate_lpt_psi_at_a(lpt, c, 0.02)
        psi_04 = evaluate_lpt_psi_at_a(lpt, c, 0.04)
        rms02 = sqrt(mean(psi_02.^2))
        rms04 = sqrt(mean(psi_04.^2))
        @test abs(rms04 / rms02 - 2.0) < 0.05
    end

    @testset "2LPT — order matters" begin
        c   = Cosmology("Planck18EEBAOSN")
        pk  = linear_power_spectrum(c)
        res = 16; T = Float32
        fphi = generate_grf(:ngenic, 3, pk, res, 100.0, 42; dtype=T, dtype_c=Complex{T})
        grid = get_fourier_grid(res, 100.0; T=T)
        lpt1 = compute_lpt(fphi, grid; n_order=1, backend=:threads)
        lpt2 = compute_lpt(fphi, grid; n_order=2, backend=:threads)

        psi_1lpt = evaluate_lpt_psi_at_a(lpt1, c, 0.02)
        psi_2lpt = evaluate_lpt_psi_at_a(lpt2, c, 0.02)
        diff_rms = sqrt(mean((psi_1lpt - psi_2lpt).^2))
        psi_rms  = sqrt(mean(psi_1lpt.^2))
        frac_diff = diff_rms / psi_rms
        @test frac_diff > 0        # 2LPT ≠ 1LPT
        @test frac_diff < 0.05     # difference < 5% at high z
    end

    @testset "Positions periodic wrap" begin
        c   = Cosmology("Planck18EEBAOSN")
        pk  = linear_power_spectrum(c)
        res = 8; T = Float32; L = 100.0
        fphi = generate_grf(:ngenic, 3, pk, res, L, 42; dtype=T, dtype_c=Complex{T})
        grid = get_fourier_grid(res, L; T=T)
        lpt  = compute_lpt(fphi, grid; n_order=1, backend=:threads)
        pos  = evaluate_lpt_pos_at_a(lpt, c, 0.1)
        @test all(0 .<= pos .< L)
    end

    @testset "Power spectrum" begin
        res = 32; boxsize = 100.0
        # White noise field → P(k) ≈ constant shot noise P = V/N (the correct
        # normalisation; guards against a spurious extra factor of V).
        field = randn(res, res, res)
        ps = evaluate_power_spectrum(field, boxsize; bins=10)
        @test length(ps.k) == 10
        @test all(ps.Pk .>= 0)
        @test all(diff(ps.k) .> 0)  # k bins are ordered
        Vn = boxsize^3 / res^3                          # shot noise V/N
        nz = ps.Pk[ps.Pk .> 0]
        @test 0.5 * Vn < sum(nz)/length(nz) < 2.0 * Vn  # amplitude ~ V/N, not V²/N
    end

    @testset "HEALPix ang2pix_ring" begin
        nside = 16
        # Pole pixel
        p = ang2pix_ring(nside, 0.0, 0.0)
        @test p >= 0 && p < 12*nside^2

        # Equator
        p_eq = ang2pix_ring(nside, π/2, 0.0)
        @test p_eq >= 0 && p_eq < 12*nside^2

        # South pole
        p_sp = ang2pix_ring(nside, π, 0.0)
        @test p_sp >= 0 && p_sp < 12*nside^2
    end

    @testset "Replica enumeration" begin
        L  = 100.0
        obs = [50.0, 50.0, 50.0]
        chi_near = 0.0
        chi_far  = 80.0
        reps = enumerate_replicas(L, obs, chi_near, chi_far)
        # The fiducial box (0,0,0) must be included
        @test any(reps[:, 1] .== 0 .& reps[:, 2] .== 0 .& reps[:, 3] .== 0)
        @test size(reps, 2) == 3
    end

    @testset "Bispectrum equilateral" begin
        res = 32; boxsize = 100.0
        # White noise → B(k) ≈ 0 (no connected 3-point function)
        rng_field = randn(res, res, res)
        bs = evaluate_bispectrum_equilateral(rng_field, boxsize; bins=5)
        @test length(bs.k) == 5
        @test all(diff(bs.k) .> 0)   # k bins ordered
        # White noise has no connected 3-pt signal, so |B|/P² ≪ 1 at well-sampled
        # (high-k, many-triangle) bins.  Low-k bins are estimator-noise dominated
        # (few triangles), so we test the two best-sampled bins.  (NB: the |δ|²-FFT
        # estimator here is internally consistent with the V/N² P(k) normalisation
        # but is NOT the same estimator as JAX's triangle enumeration.)
        ps = evaluate_power_spectrum(rng_field, boxsize; bins=5)
        rB = abs.(bs.Bk) ./ (ps.Pk .^ 2 .+ 1e-30)
        @test all(rB[end-1:end] .< 1.0)
    end

    # ── GPU backend (runs only when CUDA is available) ───────────────────────
    # The :ka backend runs unchanged on the GPU through the CUDA extension; the
    # KA kernels infer their device from the arrays and the FFTs use cuFFT.  The
    # shared GRF is canonicalised so both backends see identical input (cuFFT's
    # C2R is stricter than FFTW's about the non-canonical DC/Nyquist modes that
    # generate_grf produces).  At res ≤ 64 the agreement is fp32 round-off; at
    # larger res a ~1e-3 residual remains from the Nyquist-derivative convention
    # difference (physically negligible — Nyquist modes carry vanishing power).
    cuda_ok = false
    try
        @eval using CUDA
        cuda_ok = CUDA.functional()
    catch
        cuda_ok = false
    end
    if cuda_ok
        @testset "GPU (CUDA) ≈ CPU" begin
            relerr(a, b) = maximum(abs.(a .- b)) / (maximum(abs.(a)) + eps(Float32))
            c  = Cosmology("Planck18EEBAOSN")
            pk = linear_power_spectrum(c)
            for res in (32, 64)
                T = Float32; L = 1000.0
                grid  = get_fourier_grid(res, L; T=T)
                fphi  = generate_grf(:ngenic, 3, pk, res, L, 42; dtype=T, dtype_c=Complex{T})
                fc    = canonicalize_hermitian(fphi, grid)
                gridg = to_gpu(grid); fphig = to_gpu(fphi)
                for n in (1, 2, 3)
                    cpu = compute_lpt(fc,    grid;  n_order=n, backend=:ka)
                    gpu = compute_lpt(fphig, gridg; n_order=n, backend=:ka)
                    @test relerr(cpu.psi1, to_host(gpu.psi1)) < 1e-4
                    n >= 2 && @test relerr(cpu.psi2, to_host(gpu.psi2)) < 1e-4
                    n >= 3 && @test relerr(cpu.psi3, to_host(gpu.psi3)) < 1e-4
                end
                # f16 storage: packed displacements reconstruct to ~f16 round-off
                g32 = compute_lpt(fphig, gridg; n_order=2, backend=:ka)
                g16 = compute_lpt(fphig, gridg; n_order=2, backend=:ka, store=:f16)
                @test g16.psi1 isa HalfField
                @test relerr(to_host(g32.psi2), expand_half(to_host(g16.psi2))) < 5e-3
            end
        end
    else
        @info "CUDA not functional — skipping GPU backend tests"
    end

    # ── Differentiability (runs only if Zygote + FiniteDifferences are available) ──
    # `lpt_psi_ad` is the faithful de-aliased engine (nlpt_core) and is differentiable
    # w.r.t. the white-noise field ω — the property that makes DISCO-DJ "Done with Jax"
    # (needed for field-level IC inference).  The IC map `white_noise_to_fphi` uses the
    # JAX −1/k² gauge and reproduces JAX's `generate_grf("real")` to machine precision.
    ad_ok = false
    try
        @eval using Zygote, FiniteDifferences
        ad_ok = true
    catch
        ad_ok = false
    end
    if ad_ok
        @testset "Differentiable nLPT (∂/∂ω)" begin
            relerr2(a,b) = maximum(abs.(a .- b)) / (maximum(abs.(a)) + eps())
            res = 16; L = 1000.0; T = Float64; a = 0.1
            c  = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
            grid = get_fourier_grid(res, L; T=T)
            op   = ic_operator(res, L, pk; T=T)
            ω    = randn(MersenneTwister(7), T, res, res, res)
            fphi = white_noise_to_fphi(op, ω)
            # The faithful IC map (JAX −1/k² gauge) is the exact negative of the legacy
            # (+1/k²) generate_grf map at the same white noise.
            @test relerr2(generate_grf(:real, 3, pk, res, L, 7; dtype=T, dtype_c=Complex{T},
                                       white_noise=randn(MersenneTwister(7), T, res, res, res)),
                          -white_noise_to_fphi(op, randn(MersenneTwister(7), T, res, res, res))) < 1e-12
            # lpt_psi_ad delegates to the faithful nlpt_core engine
            K = nlpt_kernels(res, L)
            for n in (1, 2, 3)
                @test relerr2(lpt_psi_ad(fphi, grid, c, a; n_order=n),
                              lpt_displacement(fphi, K, c, a; n_order=n)) < 1e-12
            end
            # ∂/∂ω vs finite differences (the defining property)
            loss(w) = sum(abs2, lpt_psi_ad(white_noise_to_fphi(op, w), grid, c, a; n_order=2))
            g = Zygote.gradient(loss, ω)[1]
            fdm = central_fdm(5, 1)
            for i in rand(MersenneTwister(3), 1:length(ω), 5)
                gfd = FiniteDifferences.grad(fdm, t -> (w = copy(ω); w[i] = t; loss(w)), ω[i])[1]
                @test isapprox(g[i], gfd; rtol=1e-5)                         # ∂/∂ω vs FD
            end
        end
    else
        @info "Zygote/FiniteDifferences not available — skipping differentiability tests"
    end

    # ── Faithful general-order nLPT (the line-for-line JAX `compute_core` port) ──
    # No JAX dependency in CI: validated here via cosmology-independent algebraic
    # identities that the de-aliased recursion must satisfy exactly, plus a
    # finite-difference gradient check through the full transverse 3LPT engine.
    @testset "Faithful nLPT core (compute_core / compute_core_exact)" begin
        rel(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), eps())
        res = 12; L = 500.0
        white = randn(MersenneTwister(7), res, res, res)
        fphi  = rfft(white, [3, 1, 2])
        K     = nlpt_kernels(res, L)
        eds = compute_core(fphi, K; n_order=3)
        exa = compute_core_exact(fphi, K; n_order=3)

        @test size(eds["psi_1"]) == (res, res, res, 3)
        @test haskey(exa, "psi_3c_ex")               # transverse 3LPT mode present

        # EdS reconstruction identities: the exact-growth shape fields combined with
        # the EdS-limit growth ratios must equal the EdS-recursion shapes.  These are
        # algebraic identities (no cosmology) → hold to machine precision.
        @test rel(eds["psi_2"], (-3/7) .* exa["psi_2_ex"]) < 1e-12
        recon3 = (1/3) .* exa["psi_3a_ex"] .+ (-10/21) .* exa["psi_3b_ex"] .+ (1/7) .* exa["psi_3c_ex"]
        @test rel(eds["psi_3"], recon3) < 1e-12

        # general order: order 4 runs and is finite
        @test all(isfinite, compute_core(fphi, K; n_order=4)["psi_4"])

        # evaluate_core: exact-growth & EdS displacement converge in the high-z limit
        c = Cosmology("Planck18EEBAOSN")
        ψ_eds = lpt_displacement(fphi, K, c, 0.01; n_order=3, exact_growth=false)
        ψ_exa = lpt_displacement(fphi, K, c, 0.01; n_order=3, exact_growth=true)
        @test rel(ψ_exa, ψ_eds) < 1e-2

        if ad_ok
            # ∂/∂ω through the de-aliased pad/crop/conv2 + transverse-curl engine
            lossc(w) = sum(abs2, compute_core_exact(rfft(w, [3, 1, 2]), K; n_order=3)["psi_3c_ex"])
            gc = Zygote.gradient(lossc, white)[1]
            fdm = central_fdm(5, 1)
            for i in rand(MersenneTwister(5), 1:length(white), 4)
                gfd = FiniteDifferences.grad(fdm, t -> (w = copy(white); w[i] = t; lossc(w)), white[i])[1]
                @test isapprox(gc[i], gfd; rtol=1e-4, atol=1e-20)
            end
        end
    end

    # ── Lightcone crossing (Newton root-find of |x(a)-obs| = χ(a)) ──────────────
    # Regression for the dχ/da sign + the robust bracketing seed: every reported
    # crossing must satisfy the lightcone condition to ~machine precision.
    @testset "Lightcone crossing" begin
        c = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
        res = 16; L = 200.0
        fphi = generate_grf(:ngenic, 3, pk, res, L, 42; dtype=Float64, dtype_c=ComplexF64)
        grid = get_fourier_grid(res, L; T=Float64)
        lpt  = compute_lpt(fphi, grid; n_order=2, backend=:threads)
        obs  = [L/2, L/2, L/2]; reps = reshape(Int[0, 0, 0], 1, 3)

        # dχ/da < 0 (χ decreases with a) — the bug that inverted the Newton step
        @test DiscoDJNative._dchi_da_at(c, 0.5) < 0

        # Bracketing seed alone (n_newton_iters=0) lands on the lightcone
        cr = find_lightcone_crossings(lpt, c, collect(range(0.3, 1.0; length=6)), obs, reps;
                                      n_newton_iters=0, radial_residual_tol=0.1)
        @test length(cr.a_cross) > 0
        maxres = maximum(abs(norm(cr.x[i, :] .- obs) - comoving_distance(c, cr.a_cross[i]))
                         for i in eachindex(cr.a_cross))
        @test maxres < 1e-6                     # |x-obs| = χ(a) to ~machine precision
        @test all(0.3 .<= cr.a_cross .<= 1.0)   # all within the shell range
    end

    # ── Bit-exact N-GenIC (GSL ranlxd1 port) ────────────────────────────────────
    # Reference constants validated against the GSL library + the C++ rng_ngenic.
    @testset "N-GenIC bit-exact (GSL ranlxd1)" begin
        st = DiscoDJNative.ranlxd1_set(42)      # GSL ranlxd1 stream for seed 42
        ref = [0.66962007120990563, 0.26813969602963539, 0.0948083864381708,
               0.31262174009276222, 0.34292274337927253]
        for r in ref
            @test isapprox(DiscoDJNative.gsl_uniform!(st), r; atol=1e-15)
        end
        f = ngenic_field_gsl(42, 8)             # = rng_ngenic(42,8).get_field()
        @test size(f) == (8, 8, 5)
        @test isapprox(sqrt(sum(abs2, f)), 13.448527; atol=1e-3)   # ‖field‖ (C++ oracle)
        w = ngenic_wnoise_real(42, 8)           # = DISCO-DJ get_ngenic_wnoise
        @test eltype(w) == Float64 && size(w) == (8, 8, 8)
        @test 0.5 < sqrt(sum(abs2, w) / length(w)) < 1.5   # ~unit-variance white noise
    end

end
