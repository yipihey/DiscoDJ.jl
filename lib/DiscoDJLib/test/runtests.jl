# DiscoDJLib tests. Live tests run DISCO-DJ in-process via the recorded Python
# (gated on available()); the physics gates are deterministic on CPU JAX.
# Run:  <julia> --project=test test/runtests.jl

# PythonCall resolves its interpreter at ITS load time — set the env before
# `using DiscoDJLib` (module-body ENV writes don't re-run from precompile cache).
let py = get(ENV, "DISCODJ_PYTHON",
             normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    if isfile(py)
        get!(ENV, "JULIA_PYTHONCALL_EXE", py)
        ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
        get!(ENV, "JAX_PLATFORMS", "cpu")
    end
end

using DiscoDJLib, Test
const DL = DiscoDJLib

@testset "DiscoDJLib" begin

    @testset "spec defaults" begin
        s = DiscoSpec()
        @test s.dim == 3 && s.res == 64 && s.n_order == 2
        @test DiscoSpec(res = 32, n_order = 3).n_order == 3
    end

    if !DL.available()
        @info "discodj python env not found — skipping live tests" python = DL.pypath()
    else
        spec = DiscoSpec(res = 32, boxsize = 100.0, n_order = 2, seed = 42)
        b = build(spec)

        @testset "2LPT ICs: shapes, units, zero mean" begin
            ic = lpt_ics(b, 0.02)
            @test size(ic.psi) == (32, 32, 32, 3)
            @test size(ic.pos) == size(ic.psi) && size(ic.vel) == size(ic.psi)
            @test all(p -> 0 <= p < 100.0, ic.pos)                # periodic box units
            @test abs(sum(ic.psi) / length(ic.psi)) < 1e-7        # no bulk displacement
            @test sqrt(sum(abs2, ic.vel) / length(ic.vel)) > 0    # growing mode kicked
        end

        @testset "linear growth gate (near-EdS at high z)" begin
            # ψ scales with D(a); deep in matter domination D ∝ a to ~0.1%:
            rms(x) = sqrt(sum(abs2, x) / length(x))
            r = rms(lpt_ics(b, 0.04).psi) / rms(lpt_ics(b, 0.02).psi)
            @test isapprox(r, 2.0; rtol = 5e-3)
        end

        @testset "LPT order matters (2LPT ≠ Zel'dovich, small at high z)" begin
            a = 0.02
            ψ2 = lpt_ics(b, a).psi
            ψ1 = lpt_ics(b, a; n_order = 1).psi
            d = sqrt(sum(abs2, ψ2 .- ψ1) / length(ψ1)) / sqrt(sum(abs2, ψ1) / length(ψ1))
            @test 0 < d < 0.05                                    # present but perturbative
        end

        @testset "reproducibility (same seed ⇒ same fields to f32 round-off)" begin
            # single-precision JAX doesn't promise bit-identical re-execution;
            # the same seed must still reproduce the field to float32 round-off.
            b2 = build(spec)
            ψa = lpt_ics(b, 0.02).psi; ψb = lpt_ics(b2, 0.02).psi
            rms(x) = sqrt(sum(abs2, x) / length(x))
            @test rms(ψb .- ψa) / rms(ψa) < 1e-5
        end
    end
end
