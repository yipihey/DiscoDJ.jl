# Diagnostics 2 and 3: fourth-order residual at initially mean density, and shear-shape dependence.
#   julia -t 4 --project=studies/mass_cdf studies/mass_cdf/bin/diag_local.jl <config.toml> <seed_lo> <seed_hi> [eps,list] [outdir]
# Recomputes the LPT fields (1–4LPT D^n, 1–3LPT exact growth) for a few seeds (cheap) and, per method/epoch/amplitude,
#   R ≡ −ln J − g D² S      with S = T:T, g = C/D² ((1 + D₂/D₁²)/2; EdS 2/7), D = D₁(a),
#   R₂ ≡ −ln J − [δ_L + (13/42)δ_L² + g D² S]  (exact second-order remainder, same thing with the δ_L terms removed)
# in bins |δ_L|/σ < w, binned in S/σ², with the fits R = c₀σ⁴ + c₁σ²S + c₂S² (+ c₃ S³) and the exact ε→0 limit from the Taylor
# coefficient j₄ (which is the degree-4 part of −ln J, so ⟨j₄ | S, δ_L=0⟩ = c₀σ⁴ + c₁σ²S + c₂S² at ε = 1 when the Taylor series holds).
# Diagnostic 3: within narrow (δ_L, S) cells, dependence of ⟨ln J⟩ on the shape χ₃ = tr(T³)/(T:T)^{3/2} ∈ [−6^{-1/2}, 6^{-1/2}].
# FEM: optional element dumps (J_FEM, psi_lin_fem) are processed with the FEM's own linear tensor (matched) when `--fem <dir> <N>` is given.
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, LinearAlgebra, StaticArrays, JLD2, PythonCall
const M = MassCDF
const SM3 = M.SM3; const SM3d = M.SM3d

cfgpath = ARGS[1]; lo = parse(Int, ARGS[2]); hi = parse(Int, ARGS[3])
εlist = length(ARGS) >= 4 && !startswith(ARGS[4], "--") ? parse.(Float64, split(ARGS[4], ",")) : [0.2, 0.3, 0.5, 0.65, 0.8, 1.0]
outdir = joinpath(M.STUDY_ROOT, "results", "local_diagnostics"); mkpath(outdir)
cfg = load_config(cfgpath)
N = cfg["N"]; L = Float64(cfg["L"]); R_f = Float64(cfg["R_f"]); N_fine = cfg["N_fine"]
WIDTHS = [0.2, 0.1, 0.05, 0.02]
SEDGES = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]      # S/σ² bins (⟨S⟩ = 2σ²/3)
CHI_EDGES = range(-1 / sqrt(6), 1 / sqrt(6), length = 7)                              # χ₃ bins
SBINS3 = [0.2, 0.4, 0.6, 0.9, 1.3, 2.0]                                                  # S/σ² bins for the shape test

"Per-element linear shear invariants from G₁: S = T:T, q₃ = tr T³, χ₃ = q₃/S^{3/2}."
function shear_invariants(G1::Vector{SM3})
    n = length(G1); S = Vector{Float32}(undef, n); q3 = Vector{Float32}(undef, n); χ = Vector{Float32}(undef, n)
    Threads.@threads for e in 1:n
        @inbounds begin
            A = SM3d(G1[e]); A = (A + A') / 2
            T = A - (tr(A) / 3) * SM3d(I)
            s = sum(abs2, T); c = tr(T * T * T)
            S[e] = s; q3[e] = c; χ[e] = s > 0 ? c / s^1.5 : 0
        end
    end
    return S, q3, χ
end

"Weighted least squares with full covariance (plain OLS on the given design)."
function ols(X::Matrix{Float64}, y::Vector{Float64})
    A = X' * X; coef = A \ (X' * y); resid = y - X * coef
    dof = max(length(y) - size(X, 2), 1); cov = inv(Symmetric(A)) * (sum(abs2, resid) / dof)
    return coef, cov, resid
end

resid_rows = Dict{Symbol,Any}[]; fit_rows = Dict{Symbol,Any}[]; shape_rows = Dict{Symbol,Any}[]; shape_fit_rows = Dict{Symbol,Any}[]

"""
Run both diagnostics for one (method, epoch, ε) given the vertex fields.  `tag` is a Dict of identifying columns.
lnJ: −ln J per element; δL, S, χ (ε-scaled where appropriate); σ; g the finite-strain coefficient (gD² already in S units of ε²D²).
"""
function diagnose!(tag::Dict, lnJ::Vector{Float64}, δL::Vector{Float32}, S::Vector{Float32}, χ::Vector{Float32}, σ::Float64, gD2::Float64; j4 = nothing)
    R = -lnJ .- gD2 .* S
    R2 = R .- Float64.(δL) .- (13 / 42) .* Float64.(δL) .^ 2
    for w in WIDTHS
        sel = findall(abs.(δL) .< Float32(w * σ))
        isempty(sel) && continue
        s = Float64.(S[sel]) ./ σ^2; r = R[sel]; r2 = R2[sel]
        # binned means in S/σ²
        for b in 1:length(SEDGES)-1
            m = (s .>= SEDGES[b]) .& (s .< SEDGES[b+1]); nb = count(m)
            nb < 20 && continue
            push!(resid_rows, merge(tag, Dict(:width => w, :S_lo => SEDGES[b], :S_hi => SEDGES[b+1], :n => nb, :S_mean => mean(s[m]) * σ^2, :S_over_sigma2 => mean(s[m]),
                :R_mean => mean(r[m]), :R_se => std(r[m]) / sqrt(nb), :R2_mean => mean(r2[m]), :R2_se => std(r2[m]) / sqrt(nb), :R_over_sigma4 => mean(r[m]) / σ^4,
                :R2_over_sigma4 => mean(r2[m]) / σ^4, :j4_over_sigma4 => j4 === nothing ? NaN : mean(Float64.(j4[sel][m])) / σ^4)))
        end
        # fits R = c₀σ⁴ + c₁σ²S + c₂S² (+ c₃S³) on elements (dimensionless c's; design in units of σ⁴)
        Sσ = Float64.(S[sel]) ./ σ^2
        for (name, y) in (("R", r), ("R2", r2), ("j4", j4 === nothing ? nothing : Float64.(j4[sel])))
            y === nothing && continue
            for (fitname, cols) in (("quad", [ones(length(Sσ)) Sσ Sσ .^ 2]), ("cubic", [ones(length(Sσ)) Sσ Sσ .^ 2 Sσ .^ 3]))
                coef, cov, resid = ols(cols, y ./ σ^4)
                d = merge(tag, Dict(:width => w, :target => name, :fit => fitname, :n => length(sel), :c0 => coef[1], :c1 => coef[2], :c2 => coef[3],
                    :c3 => length(coef) >= 4 ? coef[4] : 0.0, :c0_se => sqrt(cov[1, 1]), :c1_se => sqrt(cov[2, 2]), :c2_se => sqrt(cov[3, 3]), :c3_se => length(coef) >= 4 ? sqrt(cov[4, 4]) : 0.0,
                    :cov_c0c1 => cov[1, 2], :cov_c0c2 => cov[1, 3], :cov_c1c2 => cov[2, 3], :resid_rms_over_sigma4 => sqrt(mean(abs2, resid)),
                    :mean_y_over_sigma4 => mean(y) / σ^4, :mean_delta_over_sigma => mean(Float64.(δL[sel])) / σ))
                push!(fit_rows, d)
            end
        end
    end
    # Diagnostic 3: shape dependence within (δ_L, S) cells
    # target: the exact second-order remainder R₂ = −ln J − δ_L − (13/42)δ_L² − g D² S (δ_L and S trends removed), so that the
    # χ₃ variance fraction is measured against the beyond-second-order content and not against the δ_L spread inside the bin
    for w in (0.1, 0.05)
        sel = findall(abs.(δL) .< Float32(w * σ))
        s = Float64.(S[sel]) ./ σ^2; y = R2[sel]; c = Float64.(χ[sel])
        tot_within = 0.0; tot_within_chi = 0.0; ntot = 0
        for b in 1:length(SBINS3)-1
            m = findall((s .>= SBINS3[b]) .& (s .< SBINS3[b+1])); nb = length(m)
            nb < 200 && continue
            yb = y[m]; cb = c[m]; sb = s[m]
            # remove the within-bin linear S trend, then test χ₃
            X0 = [ones(nb) sb]; coef0, _, res0 = ols(X0, yb)
            X1 = [ones(nb) sb cb cb .^ 2]; coef1, cov1, res1 = ols(X1, yb)
            v0 = var(res0); v1 = var(res1)
            tot_within += v0 * nb; tot_within_chi += v1 * nb; ntot += nb
            push!(shape_fit_rows, merge(tag, Dict(:width => w, :S_lo => SBINS3[b], :S_hi => SBINS3[b+1], :n => nb, :S_over_sigma2 => mean(sb),
                :slope_chi3 => coef1[3], :slope_chi3_se => sqrt(cov1[3, 3]), :curv_chi3 => coef1[4], :curv_chi3_se => sqrt(cov1[4, 4]),
                :slope_chi3_over_sigma4 => coef1[3] / σ^4, :slope_chi3_over_sigma4_se => sqrt(cov1[3, 3]) / σ^4,
                :slope_chi3_over_sigma3 => coef1[3] / σ^3, :slope_chi3_over_sigma3_se => sqrt(cov1[3, 3]) / σ^3, :var_R2 => var(yb),
                :var_resid_S => v0, :var_resid_S_chi => v1, :frac_var_explained_by_chi3 => 1 - v1 / v0,
                :mean_lnJ => mean(yb), :std_lnJ => std(yb))))
            for k in 1:length(CHI_EDGES)-1
                mk = (cb .>= CHI_EDGES[k]) .& (cb .< CHI_EDGES[k+1]); nk = count(mk)
                nk < 20 && continue
                push!(shape_rows, merge(tag, Dict(:width => w, :S_lo => SBINS3[b], :S_hi => SBINS3[b+1], :chi3_lo => CHI_EDGES[k], :chi3_hi => CHI_EDGES[k+1], :n => nk,
                    :chi3_mean => mean(cb[mk]), :S_over_sigma2 => mean(sb[mk]), :lnJ_mean => mean(lnJ[sel][m][mk]), :lnJ_se => std(lnJ[sel][m][mk]) / sqrt(nk), :R2_mean => mean(yb[mk]), :R2_se => std(yb[mk]) / sqrt(nk),
                    :lnJ_minus_binmean => mean(yb[mk]) - mean(yb), :lnJ_resid_S_mean => mean(res0[mk]), :lnJ_resid_S_se => std(res0[mk]) / sqrt(nk))))
            end
        end
        ntot > 0 && push!(shape_fit_rows, merge(tag, Dict(:width => w, :S_lo => SBINS3[1], :S_hi => SBINS3[end], :n => ntot, :S_over_sigma2 => NaN,
            :slope_chi3 => NaN, :slope_chi3_se => NaN, :curv_chi3 => NaN, :curv_chi3_se => NaN, :slope_chi3_over_sigma4 => NaN, :slope_chi3_over_sigma4_se => NaN,
            :slope_chi3_over_sigma3 => NaN, :slope_chi3_over_sigma3_se => NaN, :var_R2 => NaN,
            :var_resid_S => tot_within / ntot, :var_resid_S_chi => tot_within_chi / ntot, :frac_var_explained_by_chi3 => 1 - tot_within_chi / tot_within,
            :mean_lnJ => NaN, :std_lnJ => NaN)))
    end
end

dj0 = M.make_dj(; N, L, R_f, cosmo = cfg["cosmo"], precision = cfg["precision"], pk_file = cfg["pk_file"])
epochs = [("Dn", 1.0)]
cfg["exact_growth"] && append!(epochs, [("exact", a) for a in cfg["a_exact"]])
for seed in lo:hi
    t0 = time()
    dj = M.with_white_noise(dj0, M.white_noise(seed, N; N_fine)).with_lpt(; n_order = 4)
    cfg["exact_growth"] && (dj = dj.with_lpt(; n_order = 3, exact_growth = true))
    δ0 = M.delta_ini(dj); σ0 = sqrt(mean(abs2, Float64.(δ0)))
    keys_needed = ["psi_1", "psi_2", "psi_3", "psi_4"]; cfg["exact_growth"] && append!(keys_needed, ["psi_2_ex", "psi_3a_ex", "psi_3b_ex", "psi_3c_ex"])
    G = Dict(k => M.deformation_gradient(dj, M.lpt_psi(dj, k)) for k in keys_needed)
    S1, q31, χ1 = shear_invariants(G["psi_1"])
    tc = M.taylor_coefficients([G["psi_$n"] for n in 1:4])
    for (growth_tag, a) in epochs
        g = M.growth(dj, a)
        gcoef = growth_tag == "Dn" ? 2 / 7 : (1 + g.D2 / g.D1^2) / 2
        methods = growth_tag == "Dn" ? ["2LPT", "4LPT"] : ["3LPTx"]
        for ε in εlist, method in methods
            terms = M.method_terms(G, method, ε, g)
            J = M.jacobian_field(terms); lnJ = -log.(Float64.(J))          # stores −ln J
            δL = Float32(ε * g.D1) .* δ0; S = Float32((ε * g.D1)^2) .* S1; σ = ε * g.D1 * σ0
            tag = Dict{Symbol,Any}(:ensemble => cfg["name"], :seed => seed, :R_f => R_f, :L => L, :N => N, :method => method, :growth => growth_tag, :a => a,
                                   :eps => ε, :sigma => σ, :g => gcoef, :D1 => g.D1)
            diagnose!(tag, -lnJ, δL, S, χ1, σ, gcoef * g.D1^2 * ε^2 / ε^2; j4 = (growth_tag == "Dn" && method == "4LPT" && ε == 1.0) ? tc.j4 : nothing)
        end
    end
    @printf("seed %d done (%.0f s)\n", seed, time() - t0); flush(stdout)
    G = nothing; dj = nothing; GC.gc()
end

# FEM element dumps (matched linear tensor from psi_lin_fem), EdS: g = 2/7, D₁ = 1
femargs = findfirst(==("--fem"), ARGS)
if femargs !== nothing
    femdir = ARGS[femargs+1]; Nf = parse(Int, ARGS[femargs+2]); Lf = 500.0
    files = filter(f -> occursin("_elements_seed", f) && endswith(f, ".jld2"), readdir(femdir; join = true))
    Ωb = 0.0488911
    eds = Dict("Omega_c" => 1.0 - Ωb, "Omega_b" => Ωb, "Omega_k" => 0.0, "h" => 0.67742, "n_s" => 0.96822, "sigma8" => 0.8105, "w0" => -1.0, "wa" => 0.0)
    djf = M.make_dj(; N = Nf, L = Lf, R_f = 18.0, cosmo = eds, pk_file = joinpath(M.STUDY_ROOT, "data/pk/planck18_pk_z0.txt"))
    for f in files
        el = load(f)
        ε = parse(Float64, match(r"eps(-?[0-9.]+)\.jld2", basename(f)).captures[1])
        seed = parse(Int, match(r"seed(\d+)_", basename(f)).captures[1])
        ψlin = el["psi_lin_fem"]                        # per unit ε, (N³,3)
        Glin = M.deformation_gradient(djf, M.jnp.asarray(M.np.asarray(ψlin)).reshape(Nf, Nf, Nf, 3))
        Sf, q3f, χf = shear_invariants(Glin)
        δf = Float32[-tr(gg) for gg in Glin]; σf = abs(ε) * sqrt(mean(abs2, Float64.(δf)))
        for (meth, key) in (("FEM", "J_FEM"), ("4LPT_matched", "J_4LPT_matched"))
            haskey(el, key) || continue
            lnJ = -log.(Float64.(el[key]))
            tag = Dict{Symbol,Any}(:ensemble => basename(femdir), :seed => seed, :R_f => 18.0, :L => Lf, :N => Nf, :method => meth, :growth => "EdS", :a => 1.0,
                                   :eps => ε, :sigma => σf, :g => 2 / 7, :D1 => 1.0)
            diagnose!(tag, -lnJ, Float32(ε) .* δf, Float32(ε^2) .* Sf, χf, σf, 2 / 7)
        end
        println("FEM $(basename(f)) done"); flush(stdout)
    end
end

suffix = "_" * cfg["name"] * "_s$(lo)-$(hi)" * (femargs === nothing ? "" : "_fem")
M.write_csv(joinpath(outdir, "mean_density_residual_bins$suffix.csv"), DataFrame(resid_rows))
M.write_csv(joinpath(outdir, "mean_density_residual_fits$suffix.csv"), DataFrame(fit_rows))
M.write_csv(joinpath(outdir, "shear_shape_test_cells$suffix.csv"), DataFrame(shape_rows))
M.write_csv(joinpath(outdir, "shear_shape_test_fits$suffix.csv"), DataFrame(shape_fit_rows))
println("wrote $outdir ($suffix)")
