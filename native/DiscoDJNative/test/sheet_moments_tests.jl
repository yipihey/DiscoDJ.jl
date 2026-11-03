# Tests for src/field/sheet_moments.jl — exact answers only (run standalone:
#   julia --project=. test/sheet_moments_tests.jl   or include from runtests.jl).
using Test, DiscoDJNative, Random

@testset "sheet velocity moments" begin
    L = 100.0
    function wavy(n; amp=0.6, seed=1, T=Float64)
        rng = MersenneTwister(seed); ψ = zeros(n, n, n, 3); q = (0:n-1) .* (L / n)
        for c in 1:3, _ in 1:3
            kv = rand(rng, -2:2, 3); ph = 2π * rand(rng)
            for k in 1:n, j in 1:n, i in 1:n
                ψ[i, j, k, c] += amp * cos(2π * (kv[1]*q[i] + kv[2]*q[j] + kv[3]*q[k]) / L + ph)
            end
        end
        T.(ψ)
    end
    # shell-crossed plane wave along x (generic shifts keep nodes off tetrahedron faces / vertices)
    function crossed(n; T=Float64, B=1.0, U=0.0, Vy=0.0)
        kw = 2π * 2 / L; A = 1.6 / kw; q = (0:n-1) .* (L / n)
        ψ = zeros(n, n, n, 3); v = zeros(n, n, n, 3)
        for i in 1:n
            ψ[i, :, :, 1] .= A * sin(kw * q[i]); v[i, :, :, 1] .= B * sin(kw * q[i]) + U
        end
        v[:, :, :, 2] .= Vy
        sx = 0.1537 * L / n
        ψ[:, :, :, 1] .+= sx; ψ[:, :, :, 2] .+= 0.3719 * L / n; ψ[:, :, :, 3] .+= 0.2113 * L / n
        xq = q .+ A .* sin.(kw .* q) .+ sx; vq = B .* sin.(kw .* q) .+ U
        return T.(ψ), T.(v), xq, vq
    end
    # exact 1-D stream sums at node X: (count, ρ, v̄, σ_xx) from the linear segments [q_i, q_{i+1}]
    function exact1d(X, xq, vq, n)
        xq1 = circshift(xq, -1); xq1[end] += L; vq1 = circshift(vq, -1)
        cnt = 0; ρ = 0.0; ρs = Float64[]; vs = Float64[]
        for i in 1:n, s in (-L, 0.0, L)
            a, b = xq[i] + s, xq1[i] + s
            if min(a, b) <= X <= max(a, b) && a != b
                λ = (X - a) / (b - a); r = (L / n) / abs(b - a)
                cnt += 1; push!(ρs, r); push!(vs, vq[i] + λ * (vq1[i] - vq[i]))
            end
        end
        ρ = sum(ρs); vb = sum(ρs .* vs) / ρ
        return cnt, ρ, vb, sum(ρs .* (vs .- vb) .^ 2) / ρ
    end

    @testset "single stream: σ ≡ 0, density as sheet_mesh_periodic" begin
        n = 24; ng = 48; ψ = wavy(n); v = wavy(n; amp=3.0, seed=7)
        m = sheet_moments_mesh_periodic(ψ, v, L, ng)
        @test all(m.nstream .== 1)
        @test m.density == sheet_mesh_periodic(ψ, L, ng).density
        @test all(iszero, m.sigma)
        @test m.abar === nothing
    end

    @testset "uniform velocity: v̄ = u₀, σ = 0 through multistream regions; ā" begin
        n = 64; ng = 64; ψ, _, _, _ = crossed(n)
        u0 = (3.5, -1.25, 0.75); v = similar(ψ); for c in 1:3; v[:, :, :, c] .= u0[c]; end
        m = sheet_moments_mesh_periodic(ψ, v, L, ng; acc=2 .* v)
        @test any(m.nstream .== 3)
        @test all(isapprox(m.vbar[:, :, :, c], fill(u0[c], ng, ng, ng); atol=1e-12) for c in 1:3)
        @test maximum(abs, m.sigma) < 1e-24
        @test m.abar ≈ 2 .* m.vbar atol = 1e-12
    end

    @testset "shell-crossed plane wave: exact v̄ and σ" begin
        n = 64; ng = 64
        ψ, v, xq, vq = crossed(n; B=1.0, U=0.0, Vy=2.0)
        m = sheet_moments_mesh_periodic(ψ, v, L, ng)
        ok = true
        for I in 1:ng
            X = (I - 1) * L / ng
            cnt, ρ, vb, sxx = exact1d(X, xq, vq, n)
            ok &= all(m.nstream[I, :, :] .== cnt)
            ok &= all(isapprox.(m.vbar[I, :, :, 1], vb; atol=1e-10))
            ok &= all(isapprox.(m.sigma[I, :, :, 1], sxx; atol=1e-10))
        end
        @test ok
        @test maximum(abs, m.sigma[:, :, :, 2:6]) < 1e-14         # no transverse dispersion (round-off)
        @test all(isapprox.(m.vbar[:, :, :, 2], 2.0; atol=1e-12))
        @test maximum(m.sigma[:, :, :, 1]) > 1e-3                  # multistream dispersion present
        @test all(m.sigma[:, :, :, 1][m.nstream .== 1] .== 0)
    end

    @testset "Float32: two-pass σ survives a bulk flow 10³× the dispersion" begin
        n = 64; ng = 64; U = 1000.0
        ψ, v, xq, vq = crossed(n; T=Float32, B=1.0, U=U)
        m = sheet_moments_mesh_periodic(ψ, v, L, ng; Tacc=Float32)
        @test eltype(m.sigma) == Float32
        err = 0.0; smax = 0.0
        for I in 1:ng
            _, _, _, sxx = exact1d((I - 1) * L / ng, Float64.(xq), Float64.(vq), n)
            err = max(err, maximum(abs.(Float64.(m.sigma[I, :, :, 1]) .- sxx))); smax = max(smax, sxx)
        end
        @test err < 2e-3 * smax       # one-pass Σρvv/ρ − v̄² would carry ~eps·U² ≈ 0.1 absolute error
    end
end

@testset "sheet tracer deposit (PCS, exact kernel derivatives)" begin
    L = 100.0; n = 16; ng = 24
    rng = MersenneTwister(3)
    ψ1 = 0.6 .* randn(rng, n, n, n, 3); ψ2 = 0.3 .* randn(rng, n, n, n, 3); ψ3 = 0.2 .* randn(rng, n, n, n, 3)
    Ψ(D) = D .* ψ1 .+ D^2 .* ψ2 .+ D^3 .* ψ3; V(D) = ψ1 .+ 2D .* ψ2 .+ 3D^2 .* ψ3; A(D) = 2 .* ψ2 .+ 6D .* ψ3
    @test size(sheet_tracer_bary(0), 1) == 1 && size(sheet_tracer_bary(2), 1) == 10
    @test all(sum(sheet_tracer_bary(3); dims=2) .≈ 1)
    for lev in (0, 1, 2)
        ρ = sheet_tracer_deposit_periodic(Ψ(1.0), V(1.0), L, ng, :rho; level=lev)
        @test sum(ρ) ≈ ng^3 rtol = 1e-12                        # mass conservation (ρ̄ = 1)
        @test all(ρ .>= 0)
        @test abs(sum(sheet_tracer_deposit_periodic(Ψ(1.0), V(1.0), L, ng, :flux; level=lev))) < 1e-8 * ng^3
    end
    # node-by-node kinematic identities (random multistream ψ: arbitrary crossings)
    D = 1.0; h = 1e-4
    ρ(D) = sheet_tracer_deposit_periodic(Ψ(D), V(D), L, ng, :rho)
    fl = sheet_tracer_deposit_periodic(Ψ(D), V(D), L, ng, :flux)
    st = sheet_tracer_deposit_periodic(Ψ(D), V(D), L, ng, :stress)
    ac = sheet_tracer_deposit_periodic(Ψ(D), V(D), L, ng, :acc; acc=A(D))
    @test (ρ(D + h) .- ρ(D - h)) ./ 2h ≈ .-fl rtol = 1e-6
    @test (ρ(D + h) .- 2 .* ρ(D) .+ ρ(D - h)) ./ h^2 ≈ st .- ac rtol = 1e-4
    # uniform velocity u₀: flux = u₀·∇ρ (deposit of W), bulk with G = u₀ equals stress, :vx = u₀ₓρ
    u0 = (1.5, -0.5, 2.0); vu = similar(ψ1); for c in 1:3; vu[:, :, :, c] .= u0[c]; end
    Gu = zeros(ng, ng, ng, 3); for c in 1:3; Gu[:, :, :, c] .= u0[c]; end
    @test sheet_tracer_deposit_periodic(Ψ(D), vu, L, ng, :bulk; G=Gu) ≈ sheet_tracer_deposit_periodic(Ψ(D), vu, L, ng, :stress) rtol = 1e-10
    @test sheet_tracer_deposit_periodic(Ψ(D), vu, L, ng, :gdiv; G=Gu) ≈ sheet_tracer_deposit_periodic(Ψ(D), vu, L, ng, :flux) rtol = 1e-10
    @test sheet_tracer_deposit_periodic(Ψ(D), vu, L, ng, :vy) ≈ u0[2] .* ρ(D) rtol = 1e-12
    Gs = zeros(ng, ng, ng, 6); Gs[:, :, :, 1] .= u0[1]^2; Gs[:, :, :, 2] .= u0[2]^2; Gs[:, :, :, 3] .= u0[3]^2
    Gs[:, :, :, 4] .= u0[1] * u0[2]; Gs[:, :, :, 5] .= u0[1] * u0[3]; Gs[:, :, :, 6] .= u0[2] * u0[3]
    @test sheet_tracer_deposit_periodic(Ψ(D), vu, L, ng, :disp; G=Gs) ≈ sheet_tracer_deposit_periodic(Ψ(D), vu, L, ng, :stress) rtol = 1e-10
end
