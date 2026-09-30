# Tests of the exact sheet deposit (sheet_deposit.jl):
#   julia -t auto --project=<this directory> test_sheet_deposit.jl
using Test, DiscoDJNative, Random, Statistics
include(joinpath(@__DIR__, "sheet_deposit.jl"))

L = 300.0
@testset "exact sheet deposit" begin
    # 1. unperturbed lattice and a uniform shift that is not grid aligned: rho = 1 everywhere
    for n in (16, 32), ng in (16, 32, 48)
        ψ = zeros(n, n, n, 3)
        ρ, _ = sheet_density_exact(ψ, L, ng)
        @test maximum(abs.(ρ .- 1)) < 1e-9
        ψ .= reshape([0.37, -1.91, 5.23] .* (L / n), 1, 1, 1, 3)
        ρ, _ = sheet_density_exact(ψ, L, ng)
        @test maximum(abs.(ρ .- 1)) < 1e-9
    end
    # 2. mass conservation and positivity for a strongly nonlinear, shell-crossed flow
    n = 32; ng = 40
    rng = MersenneTwister(3)
    q = [(i - 1) * L / n for i in 1:n]
    ψ = zeros(n, n, n, 3)
    for d in 1:3, (A, kk) in ((9.0, 1), (4.0, 3), (2.0, 5))
        ph = 2π * rand(rng, 3)
        k = 2π * kk / L
        for i in 1:n, j in 1:n, l in 1:n
            ψ[i, j, l, d] += A * sin(k * (d == 1 ? q[i] : d == 2 ? q[j] : q[l]) + ph[d]) +
                             0.5A * sin(k * (q[i] + q[j] + q[l]) + ph[mod1(d + 1, 3)])
        end
    end
    ρ, st = sheet_density_exact(ψ, L, ng)
    @test abs(sum(ρ) / ng^3 - 1) < 1e-11
    @test minimum(ρ) >= 0
    @test st.tets == 6n^3
    # 3. agreement with DiscoDJNative's point-sampled sheet density, averaged over sub-samples
    #    of each cell (a smooth single-stream flow: Zel'dovich plane waves of small amplitude)
    ψs = 0.25 .* ψ
    ng = 16
    ρx, _ = sheet_density_exact(ψs, L, ng)
    # point samples at the sub-cell centres (midpoint rule): nodes sit at (j−1)·L/(ng·sub), so
    # translate the sheet by −½ sub-cell, which puts the original field's sub-cell centres on them.
    # The sheet density is piecewise constant (jumps at tetrahedron faces), so the sampled cell
    # average converges to the exact one like 1/sub.
    err = Float64[]
    for sub in (4, 8, 16)
        m = sheet_mesh_periodic(ψs .- L / (ng * sub) / 2, L, ng * sub)
        ρp = zeros(ng, ng, ng)
        for i in 1:ng, j in 1:ng, l in 1:ng
            ρp[i, j, l] = mean(@view m.density[(i-1)*sub+1:i*sub, (j-1)*sub+1:j*sub, (l-1)*sub+1:l*sub])
        end
        push!(err, mean(abs.(ρp .- ρx)))
    end
    @info "exact vs point-sampled cell averages, mean |Δρ| for sub = 4, 8, 16: $err (std ρ = $(std(ρx)))"
    @test err[1] / err[2] > 1.6 && err[2] / err[3] > 1.6     # first-order convergence to the exact value
    @test err[3] < 0.02 * std(ρx)
end
