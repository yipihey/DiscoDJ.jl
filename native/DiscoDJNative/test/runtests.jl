using Test
using DiscoDJNative
using Random: MersenneTwister

mean(x) = sum(x) / length(x)

@testset "DiscoDJNative" begin

    @testset "Cosmology" begin
        c = Cosmology("Planck18EEBAOSN")
        @test abs(Omega_m(c) - 0.3075) < 1e-3
        @test abs(c.h - 0.6774) < 1e-6
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
        # White noise field → P(k) ≈ constant (shot noise P = V/N)
        field = randn(res, res, res)
        ps = evaluate_power_spectrum(field, boxsize; bins=10)
        @test length(ps.k) == 10
        @test all(ps.Pk .>= 0)
        @test all(diff(ps.k) .> 0)  # k bins are ordered
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
        # B(k) for white noise should be small relative to P(k)²
        ps = evaluate_power_spectrum(rng_field, boxsize; bins=5)
        # |B| / P² ≪ 1 for white noise (no signal)
        @test all(abs.(bs.Bk) ./ (ps.Pk .^ 2 .+ 1e-30) .< 1.0)
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
    # The functional `lpt_psi_ad` reproduces compute_lpt+evaluate exactly and is
    # differentiable w.r.t. the initial white-noise field ω (the property that makes
    # DISCO-DJ "Done with Jax" — needed for field-level IC inference).
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
            # ω → φ(k) reproduces generate_grf(:real)
            @test relerr2(generate_grf(:real, 3, pk, res, L, 7; dtype=T, dtype_c=Complex{T}),
                          white_noise_to_fphi(op, randn(MersenneTwister(7), T, res, res, res))) < 1e-12
            for n in (1, 2, 3)
                psi_ad  = lpt_psi_ad(fphi, grid, c, a; n_order=n)
                psi_ref = evaluate_lpt_psi_at_a(compute_lpt(fphi, grid; n_order=n, backend=:threads), c, a)
                @test relerr2(psi_ref, psi_ad) < 1e-10                       # forward parity
            end
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

end
