# Part 1: Gaussian prediction of the cube/parent correlations, validated against realizations.
#   julia -t 4 --project=studies/mass_cdf multiscale/bin/scale_correlations.jl
include(joinpath(@__DIR__, "_env.jl"))
using .MS, DataFrames
resdir = joinpath(M.STUDY_ROOT, "results", "multiscale"); mkpath(resdir)
datadir = joinpath(M.STUDY_ROOT, "data", "multiscale")
Ωb = 0.0488911
eds = Dict("Omega_c" => 1.0 - Ωb, "Omega_b" => Ωb, "Omega_k" => 0.0, "h" => 0.67742, "n_s" => 0.96822, "sigma8" => 0.8105, "w0" => -1.0, "wa" => 0.0)
pk_file = joinpath(M.STUDY_ROOT, "data/pk/planck18_pk_z0.txt")

# (L, N, R_f, cosmology kind, measured seeds)
cases = Any[]
for N in (64, 128, 256), R in (14.0, 18.0, 22.0)
    ms = N == 128 ? collect(1:8) : N == 64 ? collect(1:8) : (R == 18.0 ? collect(1:4) : Int[])
    push!(cases, (L = 1000.0, N = N, R = R, kind = :lcdm, seeds = ms))
end
for N in (64, 128)
    push!(cases, (L = 500.0, N = N, R = 18.0, kind = :eds, seeds = collect(1:8)))
end

function measured_sectors(c, dj0)
    # N=128, L=1000: reuse the stored cube files if present; otherwise generate with 1LPT
    secs = Dict("CC" => NamedTuple[], "PP" => NamedTuple[], "CP" => NamedTuple[])
    for s in c.seeds
        fn = joinpath(datadir, @sprintf("cubes_L%d_N%d_R%d_seed%d.jld2", c.L, c.N, c.R, s))
        if c.kind == :lcdm && isfile(fn)
            sec = JLD2.load(fn, "sec")
        else
            w = M.white_noise(s, c.N; N_fine = 256)
            dj = M.with_white_noise(dj0, w).with_lpt(; n_order = 1)
            A6 = MS.tensor6(M.deformation_gradient(dj, M.lpt_psi(dj, "psi_1")), c.N)
            AC, AP = MS.cube_and_parent(A6, c.N, c.L)
            sec = Dict("CC" => MS.sector_moment(AC, AC), "PP" => MS.sector_moment(AP, AP), "CP" => MS.sector_moment(AC, AP))
        end
        for k in keys(secs); push!(secs[k], sec[k]); end
    end
    avg(v, f) = mean(getfield.(v, f))
    return Dict(k => (S0 = avg(v, :S0), SE = avg(v, :SE), ST = avg(v, :ST)) for (k, v) in secs),
           secs
end

rows = Dict{Symbol,Any}[]
function addrow!(c, method, S, meas_r, h; nmeas = 0)
    r = MS.ratios(S)
    row = Dict{Symbol,Any}(:R_f => c.R, :L => c.L, :N => c.N, :h => h, :method => method, :h_over_Rf => h / c.R, :cosmo => String(c.kind))
    for (sn, s) in zip(("0", "E", "T"), (:S0, :SE, :ST))
        row[Symbol("S", sn, "_CC")] = S.CC[s]; row[Symbol("S", sn, "_PP")] = S.PP[s]; row[Symbol("S", sn, "_CP")] = S.CP[s]
    end
    for (i, sn) in enumerate(("0", "E", "T"))
        row[Symbol("r", sn)] = r.r[i]; row[Symbol("f", sn, "_new")] = r.f[i]; row[Symbol("beta", sn)] = r.β[i]
    end
    for (i, sn) in enumerate(("0", "E", "T"))
        row[Symbol("r", sn, "_meas")] = meas_r === nothing ? NaN : meas_r.r[i]
        row[Symbol("f", sn, "_meas")] = meas_r === nothing ? NaN : meas_r.f[i]
    end
    row[:n_seeds_meas] = nmeas
    push!(rows, row)
end

for c in cases
    t0 = time()
    h = c.L / c.N
    dj0 = c.kind == :lcdm ? M.make_dj(; N = c.N, L = c.L, R_f = c.R) : M.make_dj(; N = c.N, L = c.L, R_f = c.R, cosmo = eds, pk_file)
    kt, Pt = M.pk_table(dj0)
    Sbox = MS.predict_sectors(kt, Pt, c.N, c.L; method = :box)
    Scon = MS.predict_sectors(kt, Pt, c.N, c.L; method = :continuum)
    meas = nothing; Smeas = nothing; secs = nothing
    if !isempty(c.seeds)
        Smeas, secs = measured_sectors(c, dj0)
        Smeas = (CC = Smeas["CC"], PP = Smeas["PP"], CP = Smeas["CP"])
        meas = MS.ratios(Smeas)
    end
    addrow!(c, "box", Sbox, meas, h; nmeas = length(c.seeds))
    addrow!(c, "continuum", Scon, meas, h; nmeas = length(c.seeds))
    meas === nothing || addrow!(c, "measured", Smeas, meas, h; nmeas = length(c.seeds))
    @printf("L=%g N=%d R_f=%g h/R=%.3f: box f=(%.4f,%.4f,%.4f) cont f=(%.4f,%.4f,%.4f)", c.L, c.N, c.R, h / c.R,
            MS.ratios(Sbox).f..., MS.ratios(Scon).f...)
    if meas !== nothing
        @printf(" meas f=(%.4f,%.4f,%.4f)  S_CC meas/box = (%.3f,%.3f,%.3f)  S_PP meas/box = (%.3f,%.3f,%.3f)", meas.f...,
                Smeas.CC.S0 / Sbox.CC.S0, Smeas.CC.SE / Sbox.CC.SE, Smeas.CC.ST / Sbox.CC.ST, Smeas.PP.S0 / Sbox.PP.S0, Smeas.PP.SE / Sbox.PP.SE, Smeas.PP.ST / Sbox.PP.ST)
    end
    @printf("  [%.0fs]\n", time() - t0); flush(stdout)
end

# FEM linear response (psi_lin gradient) measured sector correlations
for N in (64, 128)
    fn = joinpath(datadir, "cubes_FEM_L500_N$N.jld2")
    isfile(fn) || continue
    sec = JLD2.load(fn, "sec")
    for (tag, key) in (("measured_FEMlin_seed1", "lin"), ("measured_input_seed1", "in"))
        sc = sec[key]; S = (CC = sc["CC"], PP = sc["PP"], CP = sc["CP"]); r = MS.ratios(S)
        c = (L = 500.0, N = N, R = 18.0, kind = :eds)
        addrow!(c, tag, S, r, 500.0 / N; nmeas = 1)
    end
end

df = DataFrame(rows)
cols = [:R_f, :L, :N, :h, :method, :S0_CC, :S0_PP, :S0_CP, :SE_CC, :SE_PP, :SE_CP, :ST_CC, :ST_PP, :ST_CP, :r0, :rE, :rT, :f0_new, :fE_new, :fT_new,
        :beta0, :betaE, :betaT, :r0_meas, :rE_meas, :rT_meas, :f0_meas, :fE_meas, :fT_meas, :h_over_Rf, :cosmo, :n_seeds_meas]
M.write_csv(joinpath(resdir, "scale_correlations.csv"), df[:, cols])
println("wrote scale_correlations.csv (", nrow(df), " rows)")

# scaling check f_new ~ (h/R_f)^p : fit on the box predictions
bx = df[df.method .== "box" .&& df.cosmo .== "lcdm", :]
for (lab, sel) in (("R_f scan at N=128, h=7.8", bx.N .== 128), ("h scan at R_f=18 (N=64,128,256)", bx.R_f .== 18.0), ("R_f scan at N=64", bx.N .== 64), ("R_f scan at N=256", bx.N .== 256))
    b = bx[sel, :]
    for s in (:f0_new, :fE_new, :fT_new)
        x = log.(b.h ./ b.R_f); y = log.(b[:, s])
        A = [ones(length(x)) x]; coef = A \ y
        @printf("  %-34s %-7s fit f ∝ (h/R_f)^p : p = %.2f\n", lab, s, coef[2])
    end
end
