# Parity of the N-body port (src/nbody/) with the JAX DISCO-DJ `run_nbody`.
#
# Reference data: test/reference/jax_nbody_reference.h5, produced by
# test/reference/export_jax_nbody_reference.py from yipihey/disco-dj (float64, 16³ particles,
# L = 100 Mpc/h, 12 run_nbody configurations).  Standalone:
#   julia --project=. test/nbody_parity_tests.jl
using Test, DiscoDJNative, HDF5

const _REF = joinpath(@__DIR__, "reference", "jax_nbody_reference.h5")
_fromjax(a) = permutedims(a, (4, 3, 2, 1))           # h5py C-order (n,n,n,3) → Julia [i,j,k,c]
_rel(x, y) = maximum(abs.(x .- y)) / maximum(abs.(y))

@testset "N-body port vs JAX DISCO-DJ" begin
    c = Cosmology("Planck18EEBAOSN")
    f = h5open(_REF)
    try
        @testset "timetables" begin
            @test _rel(c._D1_table, read(f["cosmo/Dplus"])) < 1e-13
            @test _rel(DiscoDJNative._Dplusda_tab(c), read(f["cosmo/Dplusda"])) < 1e-13
            @test _rel(c._D2_table, read(f["cosmo/D2plus"])) < 1e-13
            @test _rel(DiscoDJNative._D2plusda_tab(c), read(f["cosmo/D2plusda"])) < 1e-13
            @test _rel(c._superconft_table, read(f["cosmo/superconft"])) < 1e-13
            @test c._D1_unnormed_at_1 ≈ read(attributes(f["cosmo"])["Dplus_unnormed_at_1"]) rtol = 1e-13
        end

        psi = _fromjax(read(f["ics/psi_ini"])); pii = _fromjax(read(f["ics/pi_ini"]))
        psi1 = _fromjax(read(f["ics/psi1"]))
        @testset "LPT initial conditions (compute_core → nbody_ics_lpt)" begin
            fphi = permutedims(read(f["ics/fphi_re"]) .+ im .* read(f["ics/fphi_im"]), (3, 2, 1))
            sh = compute_core(fphi, nlpt_kernels(16, 100.0); n_order=2)
            Ψ, Π = nbody_ics_lpt(c, sh, 0.05; n_order=2)
            @test _rel(Ψ, psi) < 1e-12
            @test _rel(Π, pii) < 1e-12
        end

        for name in keys(f)
            name in ("cosmo", "ics") && continue
            name == "bullfrog_resample_linear" && continue   # upstream bug evidence, see export script
            @testset "$name" begin
                at = attributes(f[name]); g(k) = read(at[k])
                tv = g("time_var"); tv = tv == "array" ? read(f["$name/time_var_array"]) : Symbol(tv)
                st = Symbol(g("stepper")); ns = g("n_steps")
                co = stepper_coefficients(c, st, g("a_ini"), g("a_end"), ns; time_var=tv)
                for k in (:alpha, :beta, :ddrift1, :ddrift2, :all_a)
                    @test _rel(getproperty(co, k), read(f["$name/$k"])) < 1e-12
                end
                cfg = PMConfig(res_pm=g("res_pm"), n_part=16, boxsize=100.0, worder=g("worder"),
                               deconvolve=Bool(g("deconvolve")), antialias=g("antialias"),
                               grad_order=g("grad_kernel_order"), lap_order=g("laplace_kernel_order"),
                               n_resample=g("n_resample"), resampling=Symbol(g("resampling_method")))
                @test _rel(pm_acceleration(psi, cfg), _fromjax(read(f["$name/acc_ini"]))) < 1e-12
                kw = (boxsize=100.0, a_ini=g("a_ini"), a_end=g("a_end"), n_steps=ns, res_pm=g("res_pm"),
                      time_var=tv, stepper=st, antialias=g("antialias"), grad_kernel_order=g("grad_kernel_order"),
                      laplace_kernel_order=g("laplace_kernel_order"), worder=g("worder"),
                      deconvolve=Bool(g("deconvolve")), n_resample=g("n_resample"),
                      resampling_method=Symbol(g("resampling_method")))
                p0, m0 = g("ic_method") == "bullfrog" ?
                    bullfrog_ics(c, psi1, g("a_ini"); boxsize=100.0, res_pm=g("res_pm")) : (psi, pii)
                Ψ, P, _ = run_nbody(c, p0, m0; return_displacement=true, kw...)
                @test _rel(Ψ, _fromjax(read(f["$name/psi_out"]))) < 1e-11
                @test _rel(P, _fromjax(read(f["$name/p_out"]))) < 1e-11
            end
        end
    finally
        close(f)
    end

    @testset "linear growth of a plane wave (independent of JAX)" begin
        # tiny-amplitude fundamental plane wave: every stepper must follow D(a) up to lattice
        # discreteness, which converges as (kΔq)² (measured: 16³ → 32³ reduces the error ≈ 4×).
        # (An ik gradient with deconvolution on a 2× mesh is unstable on a cold lattice — in the
        # JAX code as well — so the DISCO-DJ default force settings are used.)
        L = 100.0; a0 = 0.05
        D0, D1 = DiscoDJNative._Dplus(c, a0), DiscoDJNative._Dplus(c, 1.0)
        err = Dict{Tuple{Int,Symbol},Float64}()
        for n in (16, 32), (st, ns) in ((:bullfrog, 1), (:bullfrog, 16), (:fastpm, 16), (:symplectic, 64))
            q = [(i - 1) * L / n for i in 1:n]
            psi1 = zeros(n, n, n, 3)
            for i in 1:n; psi1[i, :, :, 1] .= 1e-3 * sin(2π * q[i] / L); end
            Ψ, _, _ = run_nbody(c, D0 .* psi1, copy(psi1); boxsize=L, a_ini=a0, a_end=1.0, n_steps=ns,
                                res_pm=2n, stepper=st, return_displacement=true)
            @test maximum(abs.(Ψ[:, :, :, 2:3])) < 1e-12                 # stays one-dimensional
            e = abs(sum(Ψ[:, :, :, 1] .* psi1[:, :, :, 1]) / sum(psi1[:, :, :, 1] .^ 2) / D1 - 1)
            err[(n, Symbol(st, ns))] = e
        end
        for k in (:bullfrog1, :bullfrog16, :fastpm16, :symplectic64)
            @test err[(32, k)] < 1e-2
            @test err[(16, k)] / err[(32, k)] > 3
        end
    end
end
