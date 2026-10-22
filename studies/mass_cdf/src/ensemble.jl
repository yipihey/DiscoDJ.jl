# ── ensemble driver ──────────────────────────────────────────────────────────

const STUDY_ROOT = normpath(joinpath(@__DIR__, ".."))

"Read a TOML ensemble configuration (see `config/*.toml`) and fill defaults."
function load_config(path::AbstractString)
    cfg = TOML.parsefile(path)
    get!(cfg, "cosmo", "Planck18EEBAOSN")
    get!(cfg, "precision", "single")
    get!(cfg, "N_fine", cfg["N"])
    get!(cfg, "eps", [0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0])
    get!(cfg, "lpt_order", 4)
    get!(cfg, "exact_growth", true)
    get!(cfg, "a_exact", [0.5, 1.0])
    get!(cfg, "ic_types", ["gaussian", "fixed"])
    get!(cfg, "save_elements_seeds", Int[])
    get!(cfg, "save_elements_eps", [0.5, 1.0])
    get!(cfg, "shear_widths", [0.02, 0.05, 0.1, 0.2])
    get!(cfg, "m_widths", [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5])
    get!(cfg, "pk_file", nothing)
    cfg["config_path"] = abspath(path)
    return cfg
end

study_config(name::AbstractString) = load_config(joinpath(STUDY_ROOT, "config", name * ".toml"))

function git_commit(dir)
    try
        return strip(read(Cmd(`git rev-parse HEAD`; dir = dir), String))
    catch
        return "unknown"
    end
end

"Growth-weighted deformation-gradient terms for a method at signed amplitude ε."
function method_terms(G::Dict{String,Vector{SM3}}, method::String, ε::Float64, g)
    if endswith(method, "LPT")               # D^n family, keys psi_1..psi_n, coefficient (εD₁)^n
        n = parse(Int, method[1:1])
        return Tuple{Float64,Vector{SM3}}[((ε * g.D1)^m, G["psi_$m"]) for m in 1:n]
    elseif endswith(method, "LPTx")          # exact growth: εD₁ψ₁ + ε²D₂ψ₂ᵉˣ + ε³(D₃ₐψ₃ₐ + D₃bψ₃b + D₃cψ₃c)
        n = parse(Int, method[1:1])
        t = Tuple{Float64,Vector{SM3}}[(ε * g.D1, G["psi_1"])]
        n >= 2 && push!(t, (ε^2 * g.D2, G["psi_2_ex"]))
        if n >= 3
            push!(t, (ε^3 * g.D3a, G["psi_3a_ex"])); push!(t, (ε^3 * g.D3b, G["psi_3b_ex"])); push!(t, (ε^3 * g.D3c, G["psi_3c_ex"]))
        end
        return t
    else
        error("unknown method $method")
    end
end

"""
    run_ensemble(cfg; seeds, ic_type, out, log=stdout)

Run the realizations `seeds` of ensemble `cfg` with `ic_type ∈ ("gaussian","fixed")` and
write `out` (JLD2) with: `rows` (DataFrame, one row per seed × method × epoch × ε),
`cdf` (F_M, F_V on the threshold grid per row), `shear` (regressions per bin), `mcond`
(shrinking-bin conditional moments of j₂,j₃,j₄ per seed), `meta`.
"""
function run_ensemble(cfg::Dict; seeds, ic_type::String, out::AbstractString, log = stdout)
    N = cfg["N"]; L = Float64(cfg["L"]); R_f = Float64(cfg["R_f"]); N_fine = cfg["N_fine"]
    fixed = ic_type == "fixed"
    scfg = StatsConfig(; shear_widths = Float64.(cfg["shear_widths"]))
    dj0 = make_dj(; N, L, R_f, cosmo = cfg["cosmo"], precision = cfg["precision"], pk_file = cfg["pk_file"])
    σ2_box = sigma2_box(dj0; N, L); σ2_cont = sigma2_continuum(dj0)
    g1 = growth(dj0, 1.0)
    epochs = [("Dn", 1.0, g1)]
    methods_Dn = ["$(n)LPT" for n in 1:cfg["lpt_order"]]
    methods_ex = cfg["exact_growth"] ? ["$(n)LPTx" for n in 1:3] : String[]
    for a in (cfg["exact_growth"] ? cfg["a_exact"] : Float64[])
        push!(epochs, ("exact", Float64(a), growth(dj0, a)))
    end
    εs = sort(vcat(-Float64.(cfg["eps"]), Float64.(cfg["eps"])))
    rows = Dict{Symbol,Any}[]; cdf = NamedTuple[]; shear = Dict{Symbol,Any}[]; mcond = Dict{Symbol,Any}[]
    elements = Dict{String,Any}()
    meta = Dict("config" => cfg, "ic_type" => ic_type, "seeds" => collect(seeds), "sigma2_box" => σ2_box,
                "sigma2_cont" => σ2_cont, "k0_cutoff" => k0_cutoff(R_f), "rho_star" => scfg.rho_star,
                "shear_widths" => scfg.shear_widths, "pct" => scfg.pct, "growth_a1" => g1,
                "A1_EdS" => A1_EDS, "A1_exact_a1" => A1_exact(g1),
                "commit_DiscoDJ.jl" => git_commit(STUDY_ROOT), "commit_disco-dj-fem" => git_commit(normpath(joinpath(dirname(DiscoDJLib.pypath()), "..", ".."))),
                "julia" => string(VERSION), "started" => string(now()), "nthreads" => Threads.nthreads())
    println(log, "ensemble $(cfg["name"]) [$ic_type] N=$N L=$L R_f=$R_f  σ_box=$(sqrt(σ2_box))  σ_cont=$(sqrt(σ2_cont))  seeds=$(seeds)")
    flush(log)
    for seed in seeds
        t0 = time()
        w = white_noise(seed, N; N_fine, fixed)
        dj = with_white_noise(dj0, w)
        dj = dj.with_lpt(; n_order = cfg["lpt_order"])
        cfg["exact_growth"] && (dj = dj.with_lpt(; n_order = 3, exact_growth = true))
        δ0 = delta_ini(dj)
        σ0 = sqrt(mean(abs2, Float64.(δ0)))
        keys_needed = ["psi_$n" for n in 1:cfg["lpt_order"]]
        cfg["exact_growth"] && append!(keys_needed, ["psi_2_ex", "psi_3a_ex", "psi_3b_ex", "psi_3c_ex"])
        G = Dict{String,Vector{SM3}}()
        for k in keys_needed
            G[k] = deformation_gradient(dj, lpt_psi(dj, k))
        end
        t_lpt = time() - t0
        # exact Taylor coefficients of −ln J (Dn family at ε = 1, a = 1)
        tc = taylor_coefficients([G["psi_$n"] for n in 1:cfg["lpt_order"]])
        TT1 = tc.TT
        # degree-k parts of −ln J (jₖ, variant "lnJ"), of J = exp(−Σεᵏjₖ) (variant "J") and of 1/J (variant "invJ")
        Jk = expansion_variants(δ0, tc.j2, tc.j3, tc.j4)
        for (variant, series) in (("lnJ", (tc.j2, tc.j3, tc.j4)), ("J", Jk.J), ("invJ", Jk.invJ)), (k, jk) in zip((2, 3, 4), series)
            k > cfg["lpt_order"] && continue
            cm = conditional_moments(jk, δ0, σ0, Float64.(cfg["m_widths"]); k)
            for b in cm.bins
                push!(mcond, Dict(:seed => seed, :ic_type => ic_type, :variant => variant, :k => k, :width => b.width, :n => b.n,
                                  :mean => b.mean, :std => b.std, :sigma0 => σ0, :sigma0_box => sqrt(σ2_box),
                                  :quad_intercept => cm.quad_intercept))
            end
        end
        # TT and j2 regression at exact second order (ε→0 limit of the shear-focusing test)
        for (bi, wdt) in enumerate(scfg.shear_widths)
            m = abs.(δ0) .< Float32(wdt * σ0)
            x = Float64.(TT1[m]); y = Float64.(tc.j2[m])
            mx = mean(x); my = mean(y); vxx = mean(abs2, x) - mx^2; vxy = mean(x .* y) - mx * my
            s = vxy / vxx
            push!(shear, Dict(:seed => seed, :ic_type => ic_type, :method => "j2exact", :growth => "Dn", :a => 1.0,
                              :eps => 0.0, :width => wdt, :n => count(m), :n_excl => 0, :slope => s,
                              :intercept => my - s * mx, :rms => sqrt(max(mean(abs2, y) - my^2 - s * vxy, 0.0)), :sigma => σ0))
        end
        nrow_seed = 0
        for (growth_tag, a, g) in epochs, ε in εs
            δL = Float32(ε * g.D1) .* δ0
            TT = Float32((ε * g.D1)^2) .* TT1
            σ = abs(ε) * g.D1 * σ0
            for method in (growth_tag == "Dn" ? methods_Dn : methods_ex)
                terms = method_terms(G, method, ε, g)
                st = jacobian_stats(terms, δL, TT, σ, scfg)
                row = Dict{Symbol,Any}(:ensemble => cfg["name"], :seed => seed, :ic_type => ic_type, :L => L, :N => N,
                    :N_fine => N_fine, :R_f => R_f, :method => method, :growth => growth_tag, :a => a, :D1 => g.D1,
                    :eps => ε, :sigma => σ, :sigma_box => abs(ε) * g.D1 * sqrt(σ2_box), :sigma_cont => abs(ε) * g.D1 * sqrt(σ2_cont),
                    :P_dpos => st.P_dpos, :F_M1 => st.F_M1, :Y => st.F_M1 - st.P_dpos,
                    :meanJ => st.meanJ, :meanJinv => st.meanJinv, :mean_lnJ => st.mean_lnJ, :minJ => st.minJ, :maxJ => st.maxJ,
                    :frac_Jle0 => st.frac_Jle0, :min_sv => st.min_sv,
                    :F_pp => st.F_pp, :F_pm => st.F_pm, :F_mp => st.F_mp, :F_mm => st.F_mm)
                for (i, p) in enumerate(scfg.pct)
                    row[Symbol("J_pct_", p)] = st.pct[i]
                end
                for ρ in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
                    i = findfirst(==(ρ), scfg.rho_star)
                    row[Symbol("F_M_rho", ρ)] = st.F_M[i]; row[Symbol("F_V_rho", ρ)] = st.F_V[i]
                end
                push!(rows, row)
                push!(cdf, (seed = seed, ic_type = ic_type, method = method, growth = growth_tag, a = a, eps = ε,
                            F_M = Float32.(st.F_M), F_V = Float32.(st.F_V)))
                for (bi, wdt) in enumerate(scfg.shear_widths)
                    push!(shear, Dict(:seed => seed, :ic_type => ic_type, :method => method, :growth => growth_tag, :a => a,
                                      :eps => ε, :width => wdt, :n => st.shear_n[bi], :n_excl => st.shear_excl[bi],
                                      :slope => st.shear_slope[bi], :intercept => st.shear_icpt[bi], :rms => st.shear_rms[bi], :sigma => σ))
                end
                nrow_seed += 1
                # element-level dump for a subset of representative runs
                if seed in cfg["save_elements_seeds"] && growth_tag == "Dn" && abs(ε) in cfg["save_elements_eps"]
                    J = jacobian_field(terms)
                    elements["J_$(method)_eps$(ε)"] = J
                end
            end
        end
        if seed in cfg["save_elements_seeds"]
            elements["delta_L"] = δ0; elements["I1"] = tc.I1; elements["I2"] = tc.I2; elements["I3"] = tc.I3
            elements["TT"] = TT1; elements["j2"] = tc.j2; elements["j3"] = tc.j3; elements["j4"] = tc.j4
            elements["sigma0"] = σ0
            epath = replace(out, ".jld2" => "_elements_seed$(seed).jld2")
            jldopen(epath, "w"; compress = true) do f
                for (k, v) in elements
                    f[k] = v
                end
            end
            empty!(elements)
            println(log, "   element arrays → $epath"); flush(log)
        end
        yrow = rows[findfirst(r -> r[:seed] == seed && r[:method] == "$(cfg["lpt_order"])LPT" && r[:eps] == 1.0, rows)]
        @printf(log, "seed %5d  σ0=%.5f  Y(4LPT,ε=1)=%+.6f  A1σ=%.6f  minJ=%.3f  [lpt %.1fs, total %.1fs, %d rows]\n",
                seed, σ0, yrow[:Y], A1_EDS * σ0, yrow[:minJ], t_lpt, time() - t0, nrow_seed)
        flush(log)
        G = nothing; dj = nothing; GC.gc()
    end
    meta["finished"] = string(now())
    df = DataFrame(rows); shear_df = DataFrame(shear); mcond_df = DataFrame(mcond)
    jldsave(out; rows = df, cdf = cdf, shear = shear_df, mcond = mcond_df, meta = meta)
    println(log, "wrote $out ($(nrow(df)) rows)"); flush(log)
    return df
end

"J = det(I + Σ cᵢGᵢ) per element (Float32)."
function jacobian_field(terms::Vector{Tuple{Float64,Vector{SM3}}})
    n = length(terms[1][2]); J = Vector{Float32}(undef, n); I3 = SM3d(I)
    coefs = Float64[c for (c, _) in terms]; Gs = [G for (_, G) in terms]
    Threads.@threads for e in 1:n
        F = I3
        @inbounds for t in eachindex(coefs)
            F += coefs[t] * SM3d(Gs[t][e])
        end
        @inbounds J[e] = det(F)
    end
    return J
end

"""
    expansion_variants(j1, j2, j3, j4) -> (J = (J2, J3, J4), invJ = (D2, D3, D4))

Degree-k parts of J = exp(−u) and of 1/J = exp(u), u = Σ εᵏ jₖ (j₁ = δ_L):
  J₂ = −j₂ + j₁²/2,  J₃ = −j₃ + j₁j₂ − j₁³/6,  J₄ = −j₄ + j₁j₃ + j₂²/2 − j₁²j₂/2 + j₁⁴/24,
  D₂ = j₂ + j₁²/2,   D₃ = j₃ + j₁j₂ + j₁³/6,   D₄ = j₄ + j₁j₃ + j₂²/2 + j₁²j₂/2 + j₁⁴/24.
"""
function expansion_variants(j1::Vector{Float32}, j2::Vector{Float32}, j3::Vector{Float32}, j4::Vector{Float32})
    n = length(j1)
    J2 = similar(j1); J3 = similar(j1); J4 = similar(j1); D2 = similar(j1); D3 = similar(j1); D4 = similar(j1)
    Threads.@threads for e in 1:n
        @inbounds begin
            a = Float64(j1[e]); b = Float64(j2[e]); c = Float64(j3[e]); d = Float64(j4[e])
            J2[e] = -b + a^2 / 2;  J3[e] = -c + a * b - a^3 / 6;  J4[e] = -d + a * c + b^2 / 2 - a^2 * b / 2 + a^4 / 24
            D2[e] = b + a^2 / 2;   D3[e] = c + a * b + a^3 / 6;   D4[e] = d + a * c + b^2 / 2 + a^2 * b / 2 + a^4 / 24
        end
    end
    return (J = (J2, J3, J4), invJ = (D2, D3, D4))
end
