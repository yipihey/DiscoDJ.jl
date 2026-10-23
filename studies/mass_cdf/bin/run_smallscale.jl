# Small smoothing scales and the post-shell-crossing regime: nLPT (vertex and Kuhn-tet Jacobians), PM N-body (DISCO-DJ
# fastpm, Kuhn-tet Jacobians of the same Lagrangian sheet) and the single-cube second-order theory extended to all thresholds.
#   julia -t 4 --project=studies/mass_cdf studies/mass_cdf/bin/run_smallscale.jl <seed> [Rf,list] [eps,list] [N] [n_steps] [res_pm] [tag]
# EdS background with the Planck18 P(k) shape (D^n LPT exact in time, g = 2/7), L = 500 Mpc/h.
# Element-level J:  vertices (FFT gradient, LPT only meaningful pre-crossing) and the 6 Kuhn tetrahedra per cube,
# J_tet = V_tet/(h³/6), signed.  Crossed elements (J ≤ 0) are kept and counted as above every density threshold
# (`signed` variant); an `abs` variant uses |J|.  Nothing is discarded.
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, LinearAlgebra, StaticArrays, JLD2, PythonCall, Dates
const M = MassCDF
seed = parse(Int, ARGS[1])
Rfs = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [4.0, 6.0, 8.0, 10.0, 14.0]
εs = length(ARGS) >= 3 ? parse.(Float64, split(ARGS[3], ",")) : [0.35, 0.5, 0.7, 1.0]
N = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 256
nsteps = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 40
res_pm = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2N
ε_lin = 0.05
PMKW = (grad_kernel_order = 0, worder = 3, deconvolve = true, n_resample = 2)   # PM force settings with 0.26 % rms linear-response error (see SUMMARY §16)
tag = length(ARGS) >= 7 ? ARGS[7] : "smallscale"
L = 500.0; a_ini = 0.02; N_fine = 256
outdir = joinpath(M.STUDY_ROOT, "data", tag); mkpath(outdir)
out = joinpath(outdir, "part_seed$(seed)_N$(N)_steps$(nsteps)_pm$(res_pm).jld2")
logf = open(replace(out, ".jld2" => ".log"), "a")
scfg = M.StatsConfig(sort(unique(vcat(M.default_rho_star(), [0.2, 0.25]))))
Ωb = 0.0488911
eds = Dict("Omega_c" => 1.0 - Ωb, "Omega_b" => Ωb, "Omega_k" => 0.0, "h" => 0.67742, "n_s" => 0.96822, "sigma8" => 0.8105, "w0" => -1.0, "wa" => 0.0)
pk_file = joinpath(M.STUDY_ROOT, "data/pk/planck18_pk_z0.txt")
pss = pyimport("discodj.analysis.phase_space_sheet")
h = L / N
q_py = M.grid_positions_py(N, L)
erf(x::Float64) = ccall(:erf, Float64, (Float64,), x); Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))

# ── single-cube second-order theory for arbitrary thresholds ──────────────────
# −ln J = δ + (13/42)δ² + g S,  S = T:T = (2σ²/15) u,  u ~ χ²₅.  F_M(>ρ★) = P(−ln J > ln ρ★).
const UG = range(0.0, 80.0, length = 20001)
const PU = [u > 0 ? u^1.5 * exp(-u / 2) / (2^2.5 * ccall(:tgamma, Float64, (Float64,), 2.5)) : 0.0 for u in UG]
function F2_theory(ρ★::Float64, σ::Float64, g::Float64; quad::Bool = true)
    f = similar(PU)
    for (i, u) in enumerate(UG)
        c = log(ρ★) - g * (2σ^2 / 15) * u             # need δ + (13/42)δ² > c
        if !quad
            f[i] = PU[i] * Φ(-c / σ)
        else
            a = 13 / 42; disc = 1 + 4a * c
            if disc <= 0
                f[i] = PU[i]                             # inequality holds for every δ
            else
                δp = (-1 + sqrt(disc)) / (2a); δm = (-1 - sqrt(disc)) / (2a)
                f[i] = PU[i] * (Φ(-δp / σ) + Φ(δm / σ))
            end
        end
    end
    return sum((f[1:end-1] .+ f[2:end]) ./ 2 .* step(UG))
end

"Signed Kuhn-tet Jacobians (6N³, layout t*N³ + cube) from unwrapped flattened positions."
function tet_J(x_flat_py)
    V = pss._kuhn_tet_volumes_jit(M.jnp.asarray(x_flat_py).reshape(N, N, N, 3), L, 1, true)
    return pyconvert(Vector{Float32}, M.np.ascontiguousarray(M.np.ravel(M.np.asarray(V)))) .* Float32(6 / h^3)
end
"Cube-mean of a vertex field (8 corners), C-order flat, repeated for the 6 tets (layout t*N³ + cube)."
function cube_mean_for_tets(f::Vector{Float32})
    A = reshape(f, N, N, N)                 # Julia dims = (k, j, i) of the C-order grid; neighbours are +1 along each dim
    S = zeros(Float32, N, N, N)
    for dz in 0:1, dy in 0:1, dx in 0:1
        S .+= circshift(A, (-dx, -dy, -dz))
    end
    c = vec(S) ./ 8f0
    return repeat(c, 6)
end
function unwrap_psi(x_py)
    psi = M.jnp.asarray(x_py).reshape(-1, 3) - q_py
    psi = psi - L * M.jnp.round(psi / L)
    return psi
end

rows = Dict{Symbol,Any}[]; cdf = NamedTuple[]
function record!(tagd::Dict, st, jtype, variant, extra = Dict{Symbol,Any}())
    row = copy(tagd); row[:jtype] = jtype; row[:variant] = variant
    for (k, v) in pairs((P_dpos = st.P_dpos, F_M1 = st.F_M1, Y = st.F_M1 - st.P_dpos, meanJ = st.meanJ, meanJinv = st.meanJinv, minJ = st.minJ,
                         maxJ = st.maxJ, frac_Jle0 = st.frac_Jle0, min_sv = st.min_sv, F_pp = st.F_pp, F_pm = st.F_pm, F_mp = st.F_mp, F_mm = st.F_mm))
        row[k] = v
    end
    for (i, p) in enumerate(scfg.pct); row[Symbol("J_pct_", p)] = st.pct[i]; end
    for ρ in [0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
        i = findfirst(==(ρ), scfg.rho_star); i === nothing && continue
        row[Symbol("F_M_rho", ρ)] = st.F_M[i]; row[Symbol("F_V_rho", ρ)] = st.F_V[i]
    end
    merge!(row, extra)
    push!(rows, row)
    push!(cdf, (seed = seed, R_f = tagd[:R_f], eps = tagd[:eps], method = tagd[:method], jtype = jtype, variant = variant, F_M = Float32.(st.F_M), F_V = Float32.(st.F_V)))
end

println(logf, "smallscale seed=$seed N=$N L=$L h=$h n_steps=$nsteps res_pm=$res_pm R_f=$Rfs eps=$εs  started $(now())"); flush(logf)
for R_f in Rfs
    dj0 = M.make_dj(; N, L, R_f, cosmo = eds, pk_file)
    σ2_box = M.sigma2_box(dj0; N, L)
    w = M.white_noise(seed, N; N_fine)
    dj1 = M.with_white_noise(dj0, w)
    linrow = let
        ics = dj1._ics
        djl = dj1.update(; ics = pydict(Dict("fphi" => ics["fphi"] * ε_lin, "key" => ics["key"]))).with_lpt(; n_order = 2)
        tl = time()
        resl = djl.run_nbody(; a_ini = a_ini, a_end = 1.0, n_steps = nsteps, stepper = "fastpm", method = "pm", res_pm = res_pm,
                             time_var = "log_a", ic_method = "lpt", nlpt_order_ics = 2, PMKW...)
        psil = pyconvert(Matrix{Float32}, M.np.asarray(unwrap_psi(resl[0])))
        psi1 = pyconvert(Matrix{Float32}, M.np.asarray(djl.evaluate_lpt_psi_at_a(1.0; n_order = 1)).reshape(-1, 3))
        psi2 = pyconvert(Matrix{Float32}, M.np.asarray(djl.evaluate_lpt_psi_at_a(1.0; n_order = 2)).reshape(-1, 3))
        rmsf(a) = sqrt(mean(abs2, a))
        Gl = M.deformation_gradient(djl, M.jnp.asarray(M.np.asarray(psil)).reshape(N, N, N, 3))
        δl = Float32[-tr(gg) for gg in Gl]; δ1 = M.delta_ini(djl)
        Dict(:pm_lin_rel_rms_vs_1LPT => rmsf(psil .- psi1) / rmsf(psi1), :pm_lin_rel_rms_vs_2LPT => rmsf(psil .- psi2) / rmsf(psi2),
             :pm_lin_std_delta_ratio => rmsf(δl) / rmsf(δ1), :pm_lin_corr_delta => sum(Float64.(δl) .* Float64.(δ1)) / sqrt(sum(abs2, Float64.(δl)) * sum(abs2, Float64.(δ1))), :t_pm_lin => time() - tl)
    end
    @printf(logf, "R_f=%4.1f PM linear response (ε=%.2f): rms(ψ_PM−ψ_1LPT)/rms=%.4f (vs 2LPT %.4f) std(δ_PM)/std(δ)=%.4f corr=%.5f [%.0fs]\n", R_f, ε_lin,
            linrow[:pm_lin_rel_rms_vs_1LPT], linrow[:pm_lin_rel_rms_vs_2LPT], linrow[:pm_lin_std_delta_ratio], linrow[:pm_lin_corr_delta], linrow[:t_pm_lin]); flush(logf)
    for ε in εs
        t0 = time()
        ics = dj1._ics
        dj = dj1.update(; ics = pydict(Dict("fphi" => ics["fphi"] * ε, "key" => ics["key"]))).with_lpt(; n_order = 4)
        δL = M.delta_ini(dj); σ = sqrt(mean(abs2, Float64.(δL)))
        G = Dict("psi_$n" => M.deformation_gradient(dj, M.lpt_psi(dj, "psi_$n")) for n in 1:4)
        TT = M.taylor_coefficients([G["psi_1"], G["psi_2"]]).TT
        δtet = cube_mean_for_tets(δL); TTtet = cube_mean_for_tets(TT)
        g = 2 / 7
        th = Dict{Symbol,Any}(:A1sigma => M.A1_EDS * σ, :series3 => 0.5 + M.A1_EDS * σ + M.A3_PRED * σ^3)
        for ρ in [0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
            th[Symbol("F2quad_rho", ρ)] = F2_theory(ρ, σ, g; quad = true); th[Symbol("F2lin_rho", ρ)] = F2_theory(ρ, σ, g; quad = false)
        end
        F2curve = Float32[F2_theory(ρ, σ, g; quad = true) for ρ in scfg.rho_star]
        push!(cdf, (seed = seed, R_f = R_f, eps = ε, method = "theory_F2quad", jtype = "theory", variant = "theory", F_M = F2curve, F_V = fill(NaN32, length(F2curve))))
        base = Dict{Symbol,Any}(:seed => seed, :L => L, :N => N, :h => h, :R_f => R_f, :eps => ε, :sigma => σ, :sigma_box => ε * sqrt(σ2_box), :n_steps => nsteps, :res_pm => res_pm, :a_ini => a_ini)
        # nLPT: vertex Jacobians and tet Jacobians
        for n in 1:4
            tagd = merge(base, Dict(:method => "$(n)LPT"))
            terms = Tuple{Float64,Vector{M.SM3}}[(1.0, G["psi_$m"]) for m in 1:n]
            record!(tagd, M.jacobian_stats(terms, δL, TT, σ, scfg), "vertex", "signed", th)
            x = q_py + M.jnp.asarray(dj.evaluate_lpt_psi_at_a(1.0; n_order = n)).reshape(-1, 3)
            Jt = tet_J(x)
            record!(tagd, M.jacobian_stats_from_J(Jt, δtet, TTtet, σ, scfg), "tet", "signed", th)
            record!(tagd, M.jacobian_stats_from_J(abs.(Jt), δtet, TTtet, σ, scfg), "tet", "abs", th)
            n == 4 && seed == 1 && ε == 1.0 && jldsave(replace(out, ".jld2" => "_Jtet4LPT_R$(R_f).jld2"), true; J_tet = Jt, delta_cube = δtet[1:N^3])
        end
        # PM N-body (2LPT ICs at a_ini, fastpm, log-a steps)
        tpm = time()
        res = dj.run_nbody(; a_ini = a_ini, a_end = 1.0, n_steps = nsteps, stepper = "fastpm", method = "pm", res_pm = res_pm,
                           time_var = "log_a", ic_method = "lpt", nlpt_order_ics = 2, PMKW...)
        x_pm = res[0]; M.jax.block_until_ready(x_pm); t_pm = time() - tpm
        psi_pm = unwrap_psi(x_pm)
        x_unw = q_py + psi_pm
        Jt = tet_J(x_unw)
        tagd = merge(base, Dict(:method => "PM", :t_pm => t_pm), linrow)
        record!(tagd, M.jacobian_stats_from_J(Jt, δtet, TTtet, σ, scfg), "tet", "signed", th)
        record!(tagd, M.jacobian_stats_from_J(abs.(Jt), δtet, TTtet, σ, scfg), "tet", "abs", th)
        Gpm = M.deformation_gradient(dj, M.jnp.asarray(psi_pm).reshape(N, N, N, 3))
        record!(tagd, M.jacobian_stats([(1.0, Gpm)], δL, TT, σ, scfg), "vertex", "signed", th)
        rel4 = pyconvert(Float64, pybuiltins.float(M.jnp.sqrt(M.jnp.mean(M.jnp.sum((psi_pm - M.jnp.asarray(dj.evaluate_lpt_psi_at_a(1.0; n_order = 4)).reshape(-1, 3))^2; axis = -1)))))
        rows[end][:rel_rms_pm_minus_4lpt] = rel4 / sqrt(mean(abs2, pyconvert(Matrix{Float32}, M.np.asarray(psi_pm))))
        seed == 1 && ε == 1.0 && jldsave(replace(out, ".jld2" => "_JtetPM_R$(R_f).jld2"), true; J_tet = Jt, delta_cube = δtet[1:N^3], psi_pm = pyconvert(Matrix{Float32}, M.np.asarray(psi_pm)))
        r4 = rows[findlast(r -> r[:method] == "4LPT" && r[:jtype] == "tet" && r[:variant] == "signed", rows)]
        rp = rows[findlast(r -> r[:method] == "PM" && r[:jtype] == "tet" && r[:variant] == "signed", rows)]
        @printf(logf, "R_f=%4.1f ε=%.2f σ=%.4f | crossed: 4LPT %.2e PM %.2e | F_M(ρ̄): 4LPT %.5f PM %.5f series %.5f F2 %.5f | F_M(ρ★=0.5): 4LPT %.5f PM %.5f F2 %.5f | F_M(0.25): 4LPT %.5f PM %.5f F2 %.5f | rms(ψ_PM−ψ_4LPT)/rms=%.4f [pm %.0fs, total %.0fs]\n",
                R_f, ε, σ, r4[:frac_Jle0], rp[:frac_Jle0], r4[:F_M1], rp[:F_M1], th[:series3], th[Symbol("F2quad_rho1.0")],
                r4[Symbol("F_M_rho0.5")], rp[Symbol("F_M_rho0.5")], th[Symbol("F2quad_rho0.5")], r4[Symbol("F_M_rho0.25")], rp[Symbol("F_M_rho0.25")], th[Symbol("F2quad_rho0.25")],
                rows[end][:rel_rms_pm_minus_4lpt], t_pm, time() - t0)
        flush(logf)
        G = nothing; Gpm = nothing; dj = nothing; GC.gc()
    end
end
jldsave(out; rows = DataFrame(rows), cdf = cdf, meta = Dict("seed" => seed, "N" => N, "L" => L, "R_f" => Rfs, "eps" => εs, "n_steps" => nsteps, "res_pm" => res_pm,
        "a_ini" => a_ini, "cosmo" => eds, "rho_star" => scfg.rho_star, "finished" => string(now())))
println(logf, "wrote $out"); close(logf)
