# The memory-lean PM solver (PMSolverLean / run_nbody(lean=true)) reproduces PMSolver to round-off.
#   julia --project=. test/pm_lean_tests.jl      (GPU part runs when CUDA is functional)
using Test, DiscoDJNative, Random

function _lean_case(dev, T)
    n, res, L = 24, 48, 120.0
    rng = MersenneTwister(5)
    ψ = dev(T.(3 .* randn(rng, n, n, n, 3)))
    for (W, g) in ((2, 4), (3, 2), (4, 6))
        cfg = PMConfig(res_pm=res, n_part=n, boxsize=L, worder=W, grad_order=g)
        a = Array(pm_acceleration!(PMSolver(cfg, ψ), ψ))
        b = Array(pm_acceleration!(PMSolverLean(cfg, ψ), ψ))
        @test maximum(abs.(a .- b)) / maximum(abs.(a)) < (T == Float64 ? 1e-12 : 2e-5)
    end
    cfg = PMConfig(res_pm=res, n_part=n, boxsize=L, worder=2, deconvolve=true)
    @test Array(pm_acceleration!(PMSolverLean(cfg, ψ), ψ)) ≈ Array(pm_acceleration!(PMSolver(cfg, ψ), ψ)) rtol = 2e-5
    @test_throws ErrorException PMSolverLean(PMConfig(res_pm=res, n_part=n, boxsize=L, antialias=1), ψ)
end

@testset "PMSolverLean ≡ PMSolver" begin
    @testset "CPU Float64" begin _lean_case(identity, Float64) end
    @testset "CPU Float32" begin _lean_case(identity, Float32) end
    @testset "run_nbody(lean=true) trajectory" begin
        c = Cosmology("Planck18EEBAOSN"); n = 16; L = 100.0
        rng = MersenneTwister(2); ψ = 0.05 .* randn(rng, n, n, n, 3); Π = ψ ./ 0.05
        kw = (boxsize=L, a_ini=0.05, a_end=0.5, n_steps=10, res_pm=32, return_displacement=true)
        X1, P1, _ = run_nbody(c, ψ, Π; kw...)
        X2, P2, _ = run_nbody(c, ψ, Π; lean=true, kw...)
        @test maximum(abs.(X1 .- X2)) < 1e-10 && maximum(abs.(P1 .- P2)) < 1e-10 * maximum(abs.(P1))
        seen = Int[]
        run_nbody(c, ψ, Π; lean=true, step_callback=(k, a, X, P) -> push!(seen, k),
                  callback_when=(k, a) -> k in (0, 4, 10), kw...)
        @test seen == [0, 4, 10]
    end
    cuda = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end
    if cuda
        @testset "CUDA Float32 (cuFFT native layout)" begin _lean_case(CuArray, Float32) end
    else
        @info "CUDA not functional — skipping GPU lean-PM test"
    end
end
