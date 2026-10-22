# ── FEM (Kuhn-tessellation P1 Poisson) N-body versus nLPT on identical phases ─
#
# DISCO-DJ's `nbody_fem.fem_nbody_cosmo` integrates the vertices of the Lagrangian sheet
# with FastPM-style D-time steps, the force coming from the P1 FEM Poisson solve on the
# deformed Kuhn mesh with tet-wise mass.  The background is EdS (so that the D^n LPT
# growth is exact in time and 4LPT is the correct comparison partner) with the ΛCDM
# spectrum shape loaded from a table.
#
# The P1 solver has O((kh)²) discretisation errors in its *linear* response (measured:
# the linear density is distorted by ≈11% rms at h = 7.8 Mpc/h for the R_f = 18 field),
# and P(J<1) is extremely sensitive to the linear field near δ_L = 0.  Each seed is
# therefore run first at ±ε_lin (small amplitude); the ε-odd part gives the FEM's own
# linear displacement ψ_lin and the ε-even part its second-order displacement.  Two LPT
# comparison partners are then formed on the same phases:
#   * "raw"     : nLPT from the input linear field δ_L;
#   * "matched" : nLPT from the FEM's linear field δ_FEM = −∇·ψ_lin (DISCO-DJ
#                 `with_external_ics(delta=…)`), which isolates the nonlinear-dynamics
#                 difference from the linear-operator discretisation.
# Statistics use (i) the FFT gradient of the vertex displacement and (ii) the Kuhn-tet
# volumes (Lagrangian elements).

"""
    run_fem_ensemble(cfg; seeds, out, log=stdout)

For each seed: FEM runs at ±ε_lin, then at every signed ε in `cfg["eps_fem"]` (3LPT ICs at
`a_ini`, integrated to a = 1), with raw and matched 1–4LPT partners.  Writes `out` (JLD2):
`rows`, `cdf`, `shear`, `linear` (per-seed linear/second-order response diagnostics), `meta`.
"""
function run_fem_ensemble(cfg::Dict; seeds, out::AbstractString, log = stdout)
    N = cfg["N"]; L = Float64(cfg["L"]); R_f = Float64(cfg["R_f"]); N_fine = get(cfg, "N_fine", N)
    fixed = get(cfg, "ic_type", "gaussian") == "fixed"
    nsteps = get(cfg, "n_steps", 50); a_ini = get(cfg, "a_ini", 0.05)
    cg_tol = get(cfg, "cg_tol", 1e-7); cg_maxiter = get(cfg, "cg_maxiter", 10)
    ic_order = get(cfg, "ic_order", 3); ε_lin = get(cfg, "eps_lin", 0.1)
    scfg = StatsConfig(; shear_widths = Float64.(get(cfg, "shear_widths", [0.02, 0.05, 0.1, 0.2])))
    pk_file = cfg["pk_file"]
    isfile(pk_file) || write_pk_table(pk_file; cosmo = get(cfg, "pk_cosmo", "Planck18EEBAOSN"), L = L, Nmax = 512)
    Ωb = 0.0488911
    eds = Dict("Omega_c" => 1.0 - Ωb, "Omega_b" => Ωb, "Omega_k" => 0.0, "h" => 0.67742, "n_s" => 0.96822,
               "sigma8" => 0.8105, "w0" => -1.0, "wa" => 0.0)
    dj0 = make_dj(; N, L, R_f, cosmo = eds, precision = get(cfg, "precision", "single"), pk_file)
    σ2_box = sigma2_box(dj0; N, L); σ2_cont = sigma2_continuum(dj0)
    nbf = pyimport("discodj.nbody_fem"); geom = pyimport("discodj.nbody_fem.geometry")
    kuhn = pyimport("discodj.fem.kuhn")
    conn = kuhn.kuhn_connectivity(N)
    q_py = grid_positions_py(N, L)
    h = L / N
    rows = Dict{Symbol,Any}[]; cdf = NamedTuple[]; shear = Dict{Symbol,Any}[]; linear = Dict{Symbol,Any}[]
    meta = Dict("config" => cfg, "seeds" => collect(seeds), "sigma2_box" => σ2_box, "sigma2_cont" => σ2_cont,
                "cosmo" => eds, "rho_star" => scfg.rho_star, "shear_widths" => scfg.shear_widths, "pct" => scfg.pct,
                "commit_DiscoDJ.jl" => git_commit(STUDY_ROOT), "started" => string(now()), "h" => h, "eps_lin" => ε_lin)
    println(log, "FEM ensemble $(cfg["name"]) N=$N L=$L h=$h R_f=$R_f nsteps=$nsteps a_ini=$a_ini ε_lin=$ε_lin σ_box=$(sqrt(σ2_box))"); flush(log)
    εs = Float64.(cfg["eps_fem"])
    rmsv(v) = sqrt(mean(abs2, Float64.(v)))
    corrv(a, b) = sum(Float64.(a) .* Float64.(b)) / sqrt(sum(abs2, Float64.(a)) * sum(abs2, Float64.(b)))

    function fem_run(dj_eps)
        x0, p0 = nbf.ics_from_discodj(dj_eps, a_ini; n_order = ic_order)
        tf = time()
        res = nbf.fem_nbody_cosmo(x0, p0, dj_eps.cosmo, a_ini, 1.0; n_steps = nsteps, boxsize = L, monitor_every = 0,
                                  cg_tol = cg_tol, cg_maxiter = cg_maxiter, check_shell_crossing = false)
        jax.block_until_ready(res.state.x)
        x_fem = res.state.x
        psi = x_fem - q_py
        psi = psi - L * jnp.round(psi / L)
        return (x = x_fem, psi = pyconvert(Matrix{Float32}, np.asarray(psi)), t = time() - tf)
    end
    to_mesh(ψ::Matrix{Float32}) = jnp.asarray(np.asarray(ψ)).reshape(N, N, N, 3)
    scaled(dj, ε) = (ics = dj._ics; dj.update(; ics = pydict(Dict("fphi" => ics["fphi"] * ε, "key" => ics["key"]))))
    psi_mat(dj, key) = pyconvert(Matrix{Float32}, np.asarray(lpt_psi(dj, key)).reshape(-1, 3))

    for seed in seeds
        t_seed = time()
        w = white_noise(seed, N; N_fine, fixed)
        dj1 = with_white_noise(dj0, w).with_lpt(; n_order = 4)          # ε = 1 reference (D₁ = 1 at a = 1)
        δ0 = delta_ini(dj1); σ0 = rmsv(δ0)
        G = Dict("psi_$n" => deformation_gradient(dj1, lpt_psi(dj1, "psi_$n")) for n in 1:4)
        ψn = Dict(n => psi_mat(dj1, "psi_$n") for n in 1:4)
        TT1 = taylor_coefficients([G["psi_1"], G["psi_2"]]).TT
        # ---- linear calibration runs at ±ε_lin
        fem_cache = Dict{Float64,Any}()
        fem_cache[ε_lin] = fem_run(scaled(dj1, ε_lin).with_lpt(; n_order = 4))
        fem_cache[-ε_lin] = fem_run(scaled(dj1, -ε_lin).with_lpt(; n_order = 4))
        ψ_lin = (fem_cache[ε_lin].psi .- fem_cache[-ε_lin].psi) ./ Float32(2ε_lin)        # ≈ ψ₁,FEM (+ O(ε²) ψ₃)
        ψ_2fem = (fem_cache[ε_lin].psi .+ fem_cache[-ε_lin].psi) ./ Float32(2ε_lin^2)     # ≈ ψ₂,FEM (+ O(ε²) ψ₄)
        G_lin = deformation_gradient(dj1, to_mesh(ψ_lin))
        δ_fem = Float32[-tr(g) for g in G_lin]                                               # FEM linear density at ε = 1
        asym = mean(g -> sum(abs2, g - g'), G_lin) / mean(g -> sum(abs2, g), G_lin)
        push!(linear, Dict(:seed => seed, :N => N, :L => L, :h => h, :sigma0 => σ0, :sigma_fem_lin => rmsv(δ_fem),
                           :corr_delta => corrv(δ0, δ_fem), :rms_psi_lin_over_psi1 => rmsv(ψ_lin) / rmsv(ψn[1]), :corr_psi_lin_psi1 => corrv(ψ_lin, ψn[1]),
                           :rms_psi2fem_over_psi2 => rmsv(ψ_2fem) / rmsv(ψn[2]), :corr_psi2fem_psi2 => corrv(ψ_2fem, ψn[2]),
                           :rel_rms_psi_lin_minus_psi1 => rmsv(ψ_lin .- ψn[1]) / rmsv(ψn[1]), :antisym_fraction_Glin => asym,
                           :P_dpos => mean(δ0 .> 0), :P_dpos_fem => mean(δ_fem .> 0)))
        # ---- matched LPT: nLPT run on the FEM's own linear density
        djm = dj0.with_external_ics(; delta = np.asarray(δ_fem).reshape(N, N, N)).with_lpt(; n_order = 4)
        Gm = Dict("psi_$n" => deformation_gradient(djm, lpt_psi(djm, "psi_$n")) for n in 1:4)
        TTm = taylor_coefficients([Gm["psi_1"], Gm["psi_2"]]).TT
        for ε in εs
            t0 = time()
            fem = haskey(fem_cache, ε) ? fem_cache[ε] : fem_run(scaled(dj1, ε).with_lpt(; n_order = 4))
            δL = Float32(ε) .* δ0; σ = abs(ε) * σ0; TT = Float32(ε^2) .* TT1
            δLm = Float32(ε) .* δ_fem; TTmε = Float32(ε^2) .* TTm
            G_fem = deformation_gradient(dj1, to_mesh(fem.psi))
            ψ4 = reduce(+, (Float32(ε^m) .* ψn[m] for m in 1:4))
            rel4 = rmsv(fem.psi .- ψ4) / rmsv(ψ4)
            rel1 = rmsv(fem.psi .- Float32(ε) .* ψn[1]) / rmsv(ψ4)
            Jtet_fem = tet_jacobians(geom, fem.x, conn, L, q_py, h)
            δtet = tet_vertex_mean(δL, conn, N); TTtet = tet_vertex_mean(TT, conn, N)
            base = Dict{Symbol,Any}(:ensemble => cfg["name"], :seed => seed, :L => L, :N => N, :h => h, :R_f => R_f, :eps => ε,
                                    :sigma => σ, :sigma_box => abs(ε) * sqrt(σ2_box), :sigma_fem_lin => abs(ε) * rmsv(δ_fem), :n_steps => nsteps, :a_ini => a_ini,
                                    :rel_rms_fem_minus_4lpt => rel4, :rel_rms_fem_minus_1lpt => rel1, :t_fem => fem.t,
                                    :P_dpos_fem => mean(δLm .> 0))
            function record!(method, st, jtype, partner)
                row = copy(base); row[:method] = method; row[:jtype] = jtype; row[:partner] = partner
                for (k, v) in pairs((P_dpos = st.P_dpos, F_M1 = st.F_M1, Y = st.F_M1 - st.P_dpos, meanJ = st.meanJ,
                                     meanJinv = st.meanJinv, minJ = st.minJ, frac_Jle0 = st.frac_Jle0, min_sv = st.min_sv,
                                     F_pp = st.F_pp, F_pm = st.F_pm, F_mp = st.F_mp, F_mm = st.F_mm))
                    row[k] = v
                end
                for (i, p) in enumerate(scfg.pct)
                    row[Symbol("J_pct_", p)] = st.pct[i]
                end
                for ρ in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
                    i = findfirst(==(ρ), scfg.rho_star)
                    row[Symbol("F_M_rho", ρ)] = st.F_M[i]; row[Symbol("F_V_rho", ρ)] = st.F_V[i]
                end
                push!(rows, row)
                push!(cdf, (seed = seed, eps = ε, method = method, jtype = jtype, partner = partner, F_M = Float32.(st.F_M), F_V = Float32.(st.F_V)))
                for (bi, wdt) in enumerate(scfg.shear_widths)
                    push!(shear, Dict(:seed => seed, :eps => ε, :method => method, :jtype => jtype, :partner => partner, :width => wdt, :n => st.shear_n[bi],
                                      :n_excl => st.shear_excl[bi], :slope => st.shear_slope[bi], :intercept => st.shear_icpt[bi],
                                      :rms => st.shear_rms[bi], :sigma => σ))
                end
                return st
            end
            # FEM itself: conditioned on the input δ_L ("raw") and on its own linear density ("matched")
            st_fem = record!("FEM", jacobian_stats([(1.0, G_fem)], δL, TT, σ, scfg), "fft", "raw")
            st_femm = record!("FEM", jacobian_stats([(1.0, G_fem)], δLm, TTmε, σ, scfg), "fft", "matched")
            record!("FEM", jacobian_stats_from_J(Jtet_fem, δtet, TTtet, σ, scfg), "tet", "raw")
            st4 = nothing; st4m = nothing
            for n in 1:4
                terms = Tuple{Float64,Vector{SM3}}[(ε^m, G["psi_$m"]) for m in 1:n]
                st = record!("$(n)LPT", jacobian_stats(terms, δL, TT, σ, scfg), "fft", "raw")
                n == 4 && (st4 = st)
                x_lpt = q_py + jnp.asarray(np.asarray(reduce(+, (Float32(ε^m) .* ψn[m] for m in 1:n))))
                record!("$(n)LPT", jacobian_stats_from_J(tet_jacobians(geom, x_lpt, conn, L, q_py, h), δtet, TTtet, σ, scfg), "tet", "raw")
                termsm = Tuple{Float64,Vector{SM3}}[(ε^m, Gm["psi_$m"]) for m in 1:n]
                stm = record!("$(n)LPT", jacobian_stats(termsm, δLm, TTmε, σ, scfg), "fft", "matched")
                n == 4 && (st4m = stm)
            end
            if seed in get(cfg, "save_elements_seeds", Int[])
                epath = replace(out, ".jld2" => "_elements_seed$(seed)_eps$(ε).jld2")
                jldsave(epath, true; delta_L = δL, delta_fem_lin = δLm, TT = TT, J_FEM = jacobian_field([(1.0, G_fem)]),
                        J_4LPT = jacobian_field(Tuple{Float64,Vector{SM3}}[(ε^m, G["psi_$m"]) for m in 1:4]),
                        J_2LPT = jacobian_field(Tuple{Float64,Vector{SM3}}[(ε^m, G["psi_$m"]) for m in 1:2]),
                        J_4LPT_matched = jacobian_field(Tuple{Float64,Vector{SM3}}[(ε^m, Gm["psi_$m"]) for m in 1:4]),
                        Jtet_FEM = Jtet_fem, psi_fem = fem.psi, psi_lin_fem = ψ_lin, psi_2fem = ψ_2fem)
            end
            yf = st_fem.F_M1 - st_fem.P_dpos; y4 = st4.F_M1 - st4.P_dpos
            yfm = st_femm.F_M1 - st_femm.P_dpos; y4m = st4m.F_M1 - st4m.P_dpos
            @printf(log, "seed %4d ε=%+.2f σ=%.4f  raw: Y_FEM=%+.5f Y_4LPT=%+.5f ΔY=%+.5f | matched: Y_FEM=%+.5f Y_4LPT=%+.5f ΔY=%+.5f | rms(ψ_FEM−ψ_4LPT)/rms=%.4f minJ_fem=%.3f [fem %.0fs, step %.0fs]\n",
                    seed, ε, σ, yf, y4, yf - y4, yfm, y4m, yfm - y4m, rel4, st_fem.minJ, fem.t, time() - t0)
            flush(log)
        end
        lr = linear[end]
        @printf(log, "   seed %4d linear response: std(δ_FEM)/std(δ)=%.4f corr=%.5f ; rms(ψ_lin)/rms(ψ1)=%.4f ; ψ2,FEM/ψ2 rms ratio=%.4f corr=%.4f ; antisym=%.1e  [seed total %.0fs]\n",
                seed, lr[:sigma_fem_lin] / lr[:sigma0], lr[:corr_delta], lr[:rms_psi_lin_over_psi1], lr[:rms_psi2fem_over_psi2], lr[:corr_psi2fem_psi2], lr[:antisym_fraction_Glin], time() - t_seed)
        flush(log)
        G = nothing; Gm = nothing; dj1 = nothing; djm = nothing; fem_cache = nothing; GC.gc()
    end
    meta["finished"] = string(now())
    jldsave(out; rows = DataFrame(rows), cdf = cdf, shear = DataFrame(shear), linear = DataFrame(linear), meta = meta)
    println(log, "wrote $out"); flush(log)
end

"Signed Kuhn-tet Jacobians J = V_tet / (h³/6) from flattened vertex positions (unwrapped, Lagrangian-aware)."
function tet_jacobians(geom, x_py, conn, L, q_py, h)
    _, V = geom.positions_to_edges(x_py, conn, Float64(L); q = q_py)
    Vj = pyconvert(Vector{Float32}, np.asarray(V))
    return Vj .* Float32(6 / h^3)
end

"Mean of a vertex field over the 4 vertices of every tet (ordering of `kuhn_connectivity`)."
function tet_vertex_mean(f::Vector{Float32}, conn, N)
    c = pyconvert(Matrix{Int32}, np.asarray(conn))          # (T, 4)
    T = size(c, 1); out = Vector{Float32}(undef, T)
    Threads.@threads for t in 1:T
        @inbounds out[t] = 0.25f0 * (f[c[t, 1]+1] + f[c[t, 2]+1] + f[c[t, 3]+1] + f[c[t, 4]+1])
    end
    return out
end

"Statistics from a precomputed Jacobian vector (no deformation tensor available: min_sv = NaN)."
function jacobian_stats_from_J(J::Vector{Float32}, δL::Vector{Float32}, TT::Vector{Float32}, σ::Float64, cfg::StatsConfig)
    # reuse the kernel by passing G = diag(J^{1/3} − 1)·I  →  det(I + G) = J  (min_sv then meaningless)
    n = length(J)
    G = Vector{SM3}(undef, n)
    Threads.@threads for e in 1:n
        @inbounds begin
            j = Float64(J[e])
            s = cbrt(abs(j)) - 1.0
            G[e] = j >= 0 ? SM3(s, 0, 0, 0, s, 0, 0, 0, s) : SM3(-cbrt(abs(j)) - 1.0, 0, 0, 0, s, 0, 0, 0, s)
        end
    end
    st = jacobian_stats([(1.0, G)], δL, TT, σ, cfg)
    return merge(st, (min_sv = NaN,))
end
