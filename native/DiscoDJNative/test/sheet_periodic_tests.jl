# Tests for src/field/sheet_periodic.jl — exact answers only (run standalone:
#   julia --project=. test/sheet_periodic_tests.jl   or include from runtests.jl).
using Test, DiscoDJNative, Random

@testset "periodic phase-space sheet" begin
    L = 100.0
    # smooth periodic single-stream displacement: a few long waves
    function wavy(n; amp=0.6, seed=1)
        rng = MersenneTwister(seed); ψ = zeros(n, n, n, 3); q = (0:n-1) .* (L / n)
        for c in 1:3, _ in 1:3
            kv = rand(rng, -2:2, 3); ph = 2π * rand(rng)
            for k in 1:n, j in 1:n, i in 1:n
                ψ[i, j, k, c] += amp * cos(2π * (kv[1]*q[i] + kv[2]*q[j] + kv[3]*q[k]) / L + ph)
            end
        end
        ψ
    end

    @testset "orientation / element volumes" begin
        s = tet_orientation_signs()
        @test all(abs.(s) .== 1)
        n = 24; ψ = wavy(n)
        el = sheet_elements_periodic(ψ, L)
        @test isapprox(sum(el.V), L^3; rtol=1e-12)          # periodic: total volume exact
        @test all(el.nflip .== 0)
        el0 = sheet_elements_periodic(zeros(n, n, n, 3), L)
        @test all(isapprox.(el0.V, (L / n)^3; rtol=1e-12))
        @test all(el0.nflip .== 0)
    end

    @testset "single-stream sheet covers every node exactly once" begin
        n = 24; ng = 48; ψ = wavy(n)
        m = sheet_mesh_periodic(ψ, L, ng)
        @test all(m.nstream .== 1)
        @test isapprox(sum(m.density) / ng^3, 1.0; atol=5e-3)   # point sampling of a smooth field
    end

    @testset "shell-crossed plane wave: exact stream count and density" begin
        n = 64; ng = 64; kw = 2π * 2 / L; A = 1.6 / kw
        q = (0:n-1) .* (L / n)
        ψ = zeros(n, n, n, 3)
        for i in 1:n; ψ[i, :, :, 1] .= A * sin(kw * q[i]); end
        # uniform generic shift in y, z so mesh nodes are not on the (axis-aligned) tet faces,
        # where the closed-tetrahedron test would count a node once per touching tetrahedron
        ψ[:, :, :, 2] .+= 0.3719 * L / n; ψ[:, :, :, 3] .+= 0.2113 * L / n
        # (and in x: sin(kq) = 0 at q = 0, L/4, … would put vertices exactly on nodes)
        sx = 0.1537 * L / n; ψ[:, :, :, 1] .+= sx
        m = sheet_mesh_periodic(ψ, L, ng)
        # exact: along x the sheet is the piecewise-linear interpolant of x(q) through the lattice;
        # count / density at node X from the linear segments [q_i, q_{i+1}]
        xq = q .+ A .* sin.(kw .* q) .+ sx; xq1 = circshift(xq, -1); xq1[end] += L
        ok_n = true; ok_d = true
        for I in 1:ng
            X = (I - 1) * L / ng; cnt = 0; dens = 0.0
            for i in 1:n, s in (-L, 0.0, L)
                a, b = xq[i] + s, xq1[i] + s
                if min(a, b) <= X <= max(a, b) && a != b
                    cnt += 1; dens += (L / n) / abs(b - a)
                end
            end
            ok_n &= all(m.nstream[I, :, :] .== cnt)
            ok_d &= all(isapprox.(m.density[I, :, :], dens; rtol=1e-10))
        end
        @test ok_n
        @test ok_d
        @test any(m.nstream .== 3)
        el = sheet_elements_periodic(ψ, L)
        @test isapprox(sum(el.V), L^3; rtol=1e-12)
        @test any(el.nflip .> 0)
    end

    @testset "point query == mesh query; periodic images" begin
        n = 32; ng = 32; kw = 2π / L; A = 1.3 / kw
        q = (0:n-1) .* (L / n); ψ = wavy(n; amp=0.8)
        for i in 1:n; ψ[i, :, :, 1] .+= A * sin(kw * q[i]); end
        m = sheet_mesh_periodic(ψ, L, ng)
        pts = zeros(ng^3, 3); g = 0
        for K in 1:ng, J in 1:ng, I in 1:ng
            g += 1; pts[g, :] .= ((I - 1), (J - 1), (K - 1)) .* (L / ng)
        end
        cl = PeriodicCellList(pts, L, L / n)
        r = sheet_query_periodic(ψ, L, pts, cl)
        @test r.nstream == vec(m.nstream)
        @test isapprox(r.density, vec(m.density); rtol=1e-12)
        @test all(isodd.(r.nstream))                      # generic points: odd stream count
    end

    @testset "element location at mesh nodes" begin
        n = 16; ng = 32; ψ = wavy(n; amp=0.4)
        loc = sheet_locate_mesh_periodic(ψ, L, ng)
        m = sheet_mesh_periodic(ψ, L, ng)
        @test loc.nstream == m.nstream
        el = sheet_elements_periodic(ψ, L)
        # the located element's stream density equals the sheet density at single-stream nodes
        ρe = (L / n)^3 ./ el.V
        sel = findall(==(1), loc.nstream)
        # sheet density at a node = that tetrahedron's density; element density = its volume average,
        # so compare loosely: same element ⇒ density within the element's tet spread
        @test all(1 .<= loc.element[sel] .<= n^3)
        @test isapprox(sum(ρe[loc.element[sel]]) / length(sel), sum(m.density[sel]) / length(sel); rtol=2e-2)
    end

    @testset "band-limited refinement is exact" begin
        n = 16; L2 = 50.0; kv = (1, 2, 3)
        f(x, y, z) = 0.3 * sin(2π * (kv[1]*x + kv[2]*y + kv[3]*z) / L2)
        ψ = zeros(n, n, n, 3)
        for k in 1:n, j in 1:n, i in 1:n
            ψ[i, j, k, 2] = f((i-1)*L2/n, (j-1)*L2/n, (k-1)*L2/n)
        end
        r = refine_displacement(ψ, 1); e = 2n
        ref = [f((i-1)*L2/e, (j-1)*L2/e, (k-1)*L2/e) for i in 1:e, j in 1:e, k in 1:e]
        @test maximum(abs.(r[:, :, :, 2] .- ref)) < 1e-12
        @test maximum(abs.(r[:, :, :, 1])) < 1e-14
    end

    @testset "trilinear sampling" begin
        ng = 8; Lb = 8.0
        f = [Float64(i + 2j + 3k) for i in 0:ng-1, j in 0:ng-1, k in 0:ng-1]
        v = sample_trilinear_periodic(f, [1.25 2.5 3.75], Lb)
        @test v[1] ≈ 1.25 + 2 * 2.5 + 3 * 3.75
    end
end
