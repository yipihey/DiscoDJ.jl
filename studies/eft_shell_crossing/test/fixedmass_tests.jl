# Fixed-mass sheet statistics (src/fixedmass.jl): the real-space mass budget of spheres closes exactly
# (μ̇_R = −flux_R, μ̈_R = stress_R − acc_R, up to the finite difference in D) before and after shell
# crossing, on a small LPT realization (CPU).   julia --project=. test/fixedmass_tests.jl
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, DiscoDJNative, Test, Statistics
const DJ = DiscoDJNative

pk = planck_pk()
L = 200.0; N = 32; nord = 3
s = PairSpec(N=N, N_ic=N, L=L, k_split=0.25, k_uv=0.45)
sh = compute_core(make_fphi(s, pk, :UV; T=Float64), nlpt_kernels(N, L; T=Float64); n_order=nord)
fields(D) = (Ψ = sum(D^m .* sh["psi_$m"] for m in 1:nord), V = sum(m * D^(m - 1) .* sh["psi_$m"] for m in 1:nord),
             A = lpt_acc!(similar(sh["psi_1"]), sh, D, nord))
crossD = [D for D in 0.1:0.1:2.0 if inverted_fraction(fields(D).Ψ, L) > 0]
D_pre = 0.5 * minimum(crossD); D_post = minimum(crossD) + 0.3
rmsrel(a, b) = sqrt(mean(abs2, a .- b) / mean(abs2, b))

@testset "fixed-mass budget (LPT, N=$N)" begin
    @test inverted_fraction(fields(D_pre).Ψ, L) == 0
    @test inverted_fraction(fields(D_post).Ψ, L) > 0
    for (ng, D, tol1, tol2, label) in ((64, D_pre, 5e-4, 2e-2, "pre-crossing"), (64, D_post, 5e-4, 2e-2, "post-crossing"))
        P = FixedMassPlan(ng, L, [12.0, 20.0, 40.0]; nbins=32, stride=1, T=Float64)
        h = 1e-2                                            # FD error vs round-off amplification (μ̈ ~ ε/h²)
        rp = (f = fields(D + h); fixedmass_snapshot(P, f.Ψ, f.V; acc=f.A))
        rm = (f = fields(D - h); fixedmass_snapshot(P, f.Ψ, f.V; acc=f.A))
        r0 = (f = fields(D); fixedmass_snapshot(P, f.Ψ, f.V; acc=f.A))
        μ(r) = r.sub[:, 1, :]
        dμ = (μ(rp) .- μ(rm)) ./ 2h                         # finite difference in D
        d2μ = (μ(rp) .- 2 .* μ(r0) .+ μ(rm)) ./ h^2
        flux = r0.sub[:, 3, :]; budget = r0.sub[:, 4, :] .- r0.sub[:, 8, :]
        e1 = rmsrel(dμ, .-flux); e2 = rmsrel(d2μ, budget)
        println("  $label D=$(round(D, digits=3)) ng=$ng: closure μ̇ rel.rms = $(round(e1, sigdigits=3)), μ̈ rel.rms = $(round(e2, sigdigits=3)), ",
                "ms mass frac = $(round(r0.scal["ms_mass_frac"], sigdigits=3)), ⟨disp⟩/⟨bulk⟩ rms = ",
                round(sqrt(mean(abs2, r0.sub[:, 6, :]) / mean(abs2, r0.sub[:, 5, :])), sigdigits=3),
                ", (stress−bulk−disp)/stress rms = ", round(rmsrel(r0.sub[:, 5, :] .+ r0.sub[:, 6, :], r0.sub[:, 4, :]), sigdigits=3))
        @test e1 < tol1
        @test e2 < tol2
        D == D_pre && @test all(iszero, r0.sub[:, 6, :])  # single stream: no dispersion stress
        @test isapprox(r0.scal["mean_rho"], 1; rtol=1e-10)
        # histogram bookkeeping: counts sum to the node count; Σμ over bins equals the global sum
        @test all(sum(r0.hist[:, :, 1]; dims=2) .== ng^3)
        @test vec(sum(r0.hist[:, :, 2]; dims=2)) ≈ r0.tmom[:, 1, 1] .* ng^3 rtol = 1e-10
        @test vec(sum(r0.hist[:, :, 4]; dims=2)) ≈ r0.tmom[:, 3, 1] .* ng^3 rtol = 1e-8 atol = 1e-8
    end
end
