# Parts 2-4: one-scale vs two-scale regressions, CDF bookkeeping, innovation response.
#   julia -t 4 --project=studies/mass_cdf multiscale/bin/regress.jl <suite>      suite = gauss | fem64 | fem128 [casefilter]
include(joinpath(@__DIR__, "_env.jl"))
using .MS
include(joinpath(@__DIR__, "..", "src", "Regress.jl"))
using DataFrames
suite = ARGS[1]
filt = length(ARGS) >= 2 ? ARGS[2] : ""
datadir = joinpath(M.STUDY_ROOT, "data", "multiscale")
partdir = joinpath(M.STUDY_ROOT, "results", "multiscale", "parts"); mkpath(partdir)
Ωb = 0.0488911
eds = Dict("Omega_c" => 1.0 - Ωb, "Omega_b" => Ωb, "Omega_k" => 0.0, "h" => 0.67742, "n_s" => 0.96822, "sigma8" => 0.8105, "w0" => -1.0, "wa" => 0.0)
pk_file = joinpath(M.STUDY_ROOT, "data/pk/planck18_pk_z0.txt")
const BETA_CACHE = Dict{Tuple,Any}()
function betas_for(L, N, R_f, kind)
    get!(BETA_CACHE, (L, N, R_f, kind)) do
        dj0 = kind == :lcdm ? M.make_dj(; N, L, R_f) : M.make_dj(; N, L, R_f, cosmo = eds, pk_file)
        kt, Pt = M.pk_table(dj0)
        MS.ratios(MS.predict_sectors(kt, Pt, N, L; method = :box)).β
    end
end

# ---------------------------------------------------------------- innovation response
function innovation(test, ε, fit, β; target_cut = 0.1)
    # tertiles of ε·trA_P over the rows used (J > cut)
    i1p = Float64[]
    for g in test, e in eachindex(g.J)
        g.J[e] > target_cut && push!(i1p, ε * (Float64(g.AP[1, e]) + Float64(g.AP[2, e]) + Float64(g.AP[3, e])))
    end
    q1, q2 = quantile(i1p, [1 / 3, 2 / 3])
    edges = [-Inf, q1, q2, Inf]
    # accumulators per tertile (index 1..3) and overall (4): sums over [1, x1, x2, x3, e]
    S = [zeros(5, 5) for _ in 1:4]; cnt = zeros(Int, 4); sumI = zeros(4)
    x = zeros(5)
    for g in test
        n = Reg.nrows(g)
        X = Matrix{Float64}(undef, n, 26); MS.design!(X, g.AC, g.AP, 1:n, ε, :inv)
        pred = X[:, 1:11] * fit.β
        for e in 1:n
            J = g.J[e]; J > target_cut || continue
            c = ntuple(i -> ε * Float64(g.AC[i, e]), 6); p = ntuple(i -> ε * Float64(g.AP[i, e]), 6)
            tC = (c[1] + c[2] + c[3]) / 3; tP = (p[1] + p[2] + p[3]) / 3
            etr = tC - β[1] * tP
            e1 = (c[1] - tC) - β[2] * (p[1] - tP); e2 = (c[2] - tC) - β[2] * (p[2] - tP); e3 = (c[3] - tC) - β[2] * (p[3] - tP)
            t1 = c[4] - β[3] * p[4]; t2 = c[5] - β[3] * p[5]; t3 = c[6] - β[3] * p[6]
            x[1] = 1; x[2] = abs(3 * etr) / sqrt(3); x[3] = e1^2 + e2^2 + e3^2; x[4] = 2 * (t1^2 + t2^2 + t3^2); x[5] = log(Float64(J)) - pred[e]
            ip = 3 * tP
            b = ip < q1 ? 1 : ip < q2 ? 2 : 3
            for t in (b, 4)
                S[t] .+= x * x'; cnt[t] += 1; sumI[t] += ip
            end
        end
    end
    out = Dict{Symbol,Any}[]
    for t in 1:4
        s = S[t]; n = cnt[t]
        row = Dict{Symbol,Any}(:tertile => t == 4 ? 0 : t, :n => n, :I1P_lo => t == 4 ? -Inf : edges[t], :I1P_hi => t == 4 ? Inf : edges[t+1], :mean_I1P => sumI[t] / n)
        mean_(i) = s[1, i] / n
        cov_(i, j) = s[i, j] / n - mean_(i) * mean_(j)
        row[:sd_resid] = sqrt(cov_(5, 5)); row[:mean_resid] = mean_(5)
        for (i, nm) in ((2, "eta0abs"), (3, "EE"), (4, "TT"))
            row[Symbol("slope_", nm)] = cov_(i, 5) / cov_(i, i)
            row[Symbol("corr_", nm)] = cov_(i, 5) / sqrt(cov_(i, i) * cov_(5, 5))
            row[Symbol("meanx_", nm)] = mean_(i)
            row[Symbol("sdx_", nm)] = sqrt(cov_(i, i))
        end
        C = [cov_(i, j) for i in 2:4, j in 2:4]; c = [cov_(i, 5) for i in 2:4]
        mv = C \ c
        row[:mv_eta0abs] = mv[1]; row[:mv_EE] = mv[2]; row[:mv_TT] = mv[3]
        row[:mv_R2] = dot(mv, c) / cov_(5, 5)
        push!(out, row)
    end
    return out
end

# ---------------------------------------------------------------- run one case & write part files
function process_case(tag, meta, train, test, ε; flex, trainlabel, testlabel, innov = true)
    println("=== case $tag  (train rows $(sum(Reg.nrows.(train))), test rows $(sum(Reg.nrows.(test))), flex=$flex)"); flush(stdout)
    t0 = time()
    fits, stats, (mons1, mons2) = Reg.run_case(train, test, ε; flex)
    println("    regression done in $(round(time() - t0)) s"); flush(stdout)
    loc = Dict{Symbol,Any}[]; cdf = Dict{Symbol,Any}[]
    sig = sqrt(mean(vcat([(ε .* (Float64.(g.AC[1, :]) .+ Float64.(g.AC[2, :]) .+ Float64.(g.AC[3, :]))) .^ 2 for g in test]...)))
    for (k3, v) in sort(collect(stats); by = x -> (string(x[1][1]), string(x[1][2]), x[1][3]))
        model, kind, cut = k3
        t, c = Reg.pooled(v)
        rg = [1 - Reg.var2(ti) / Reg.var1(ti) for (ti, _) in v]
        f1 = fits[(model, kind, cut, 1)]; f2 = fits[(model, kind, cut, 2)]
        b1 = model == :inv ? join(MS.BASIS_ONE, ";") : "monomials of degree<=3 in the 6 components of eps*A_C ($(length(mons1)) terms)"
        b2 = model == :inv ? join(MS.BASIS_TWO, ";") : "monomials of degree<=3 in the 12 components of eps*(A_C,A_P) ($(length(mons2)) terms)"
        row = merge(meta, Dict{Symbol,Any}(:target => kind == :ln ? "lnJ" : "J", :cut => Reg.cutname(cut), :train_seeds => trainlabel, :test_seeds => testlabel,
            :n_cubes_test => t.n, :n_cubes_train => f1.n, :Var_y => Reg.vary(t), :V_C => Reg.var1(t), :V_CP => Reg.var2(t),
            :R_gain => 1 - Reg.var2(t) / Reg.var1(t), :unexpl_one => Reg.var1(t) / Reg.vary(t), :unexpl_two => Reg.var2(t) / Reg.vary(t), :R_gain_err => length(rg) > 1 ? std(rg) / sqrt(length(rg)) : NaN,
            :V_C_train => f1.vtrain, :V_CP_train => f2.vtrain, :R_gain_train => 1 - f2.vtrain / f1.vtrain, :Var_y_train => f1.vy,
            :rms_res_one => sqrt(Reg.var1(t)), :rms_res_two => sqrt(Reg.var2(t)),
            :npar_one => length(f1.β), :npar_two => length(f2.β), :basis_one => b1, :basis_two => b2, :model => model == :inv ? "invariants" : "flexible"))
        push!(loc, row)
        nall = c.nall
        grp(f1_, f0_) = [(getfield(ci, f1_) - getfield(ci, f0_)) / ci.nall for (_, ci) in v]
        se(x) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN
        push!(cdf, merge(meta, Dict{Symbol,Any}(:cut => Reg.cutname(cut), :model => model == :inv ? "invariants" : "flexible", :target => kind == :ln ? "lnJ" : "J",
            :F_M_actual => c.nM / nall, :F_M_pred_one => c.nM1 / nall, :F_M_pred_two => c.nM2 / nall,
            :F_pm_actual => c.npm / nall, :F_pm_one => c.npm1 / nall, :F_pm_two => c.npm2 / nall,
            :F_mp_actual => c.nmp / nall, :F_mp_one => c.nmp1 / nall, :F_mp_two => c.nmp2 / nall,
            :dF_M_one => (c.nM1 - c.nM) / nall, :dF_M_two => (c.nM2 - c.nM) / nall,
            :dF_M_one_err => se(grp(:nM1, :nM)), :dF_M_two_err => se(grp(:nM2, :nM)),
            :dF_pm_one => (c.npm1 - c.npm) / nall, :dF_pm_two => (c.npm2 - c.npm) / nall,
            :dF_pm_one_err => se(grp(:npm1, :npm)), :dF_pm_two_err => se(grp(:npm2, :npm)),
            :P_dpos => c.ndpos / nall, :sigma_cube => sig, :A1_sigma_cube => M.A1_EDS * sig, :n_cubes_test => nall)))
    end
    # sanity: leading coefficients of the one-scale ln J fit (cut J>0.1, and none) in eps*A units
    coefs = Dict{Symbol,Any}[]
    for cut in (0.0, 0.1)
        f = fits[(:inv, :ln, cut, 1)]
        push!(coefs, merge(meta, Dict{Symbol,Any}(:cut => Reg.cutname(cut), Dict(Symbol("c_" * nm) => f.β[i] for (i, nm) in enumerate(MS.BASIS_ONE))...)))
    end
    inn = Dict{Symbol,Any}[]
    if innov
        β = betas_for(meta[:L], meta[:N], meta[:R_f], meta[:cosmo] == "lcdm" ? :lcdm : :eds)
        for r in innovation(test, ε, fits[(:inv, :ln, 0.1, 1)], β)
            push!(inn, merge(meta, r, Dict{Symbol,Any}(:beta0 => β[1], :betaE => β[2], :betaT => β[3])))
        end
    end
    for (nm, rows) in (("locality", loc), ("cdf", cdf), ("innov", inn), ("coefs", coefs))
        if !isempty(rows)
            df = DataFrame(rows); df = df[:, sort(names(df))]
            M.write_csv(joinpath(partdir, "$(nm)_$(tag).csv"), df)
        end
    end
    @printf("    wall %.0f s; one-scale vs two-scale (inv, ln, J>0.1): R_gain=%.4f ; (inv, J, J>0.1): R_gain=%.4f\n", time() - t0,
            only(r[:R_gain] for r in loc if r[:model] == "invariants" && r[:target] == "lnJ" && r[:cut] == "J>0.1"),
            only(r[:R_gain] for r in loc if r[:model] == "invariants" && r[:target] == "J" && r[:cut] == "J>0.1"))
    flush(stdout)
end

# ---------------------------------------------------------------- suites
if suite == "gauss"
    for (R_f, εs) in ((18.0, [1.0, 0.5]), (14.0, [1.0]), (22.0, [1.0])), ε in εs, n in (4, 2)
        tag = "gauss_R$(Int(R_f))_e$(ε)_$(n)LPT"
        occursin(filt, tag) || continue
        function load(seeds)
            [let d = JLD2.load(joinpath(datadir, @sprintf("cubes_L1000_N128_R%d_seed%d.jld2", R_f, s)))
                 lim = parse(Int, get(ENV, "MS_LIMIT", "0")); r = lim > 0 ? (1:lim) : (1:length(d["Js"]["J_$(n)LPT_e$(ε)"]))
                 Reg.Group(d["AC"][:, r], d["AP"][:, r], d["Js"]["J_$(n)LPT_e$(ε)"][r], "seed$s")
             end for s in seeds]
        end
        train = load(1:4); test = load(5:8)
        meta = Dict{Symbol,Any}(:method => "$(n)LPT", :R_f => R_f, :L => 1000.0, :N => 128, :h => 1000.0 / 128, :eps => ε, :A_source => "input", :cosmo => "lcdm")
        process_case(tag, meta, train, test, ε; flex = (n == 4), trainlabel = "1-4", testlabel = "5-8")
    end
elseif suite == "gaussK"
    for (R_f, εs) in ((18.0, [1.0, 0.5]), (14.0, [1.0]), (22.0, [1.0])), ε in εs, n in (4, 2)
        tag = "gaussK_R$(Int(R_f))_e$(ε)_$(n)LPT"
        occursin(filt, tag) || continue
        function load(seeds)
            [let d = JLD2.load(joinpath(datadir, @sprintf("cubes_L1000_N128_R%d_seed%d.jld2", R_f, s))),
                 dk = JLD2.load(joinpath(datadir, @sprintf("cubesK_L1000_N128_R%d_seed%d.jld2", R_f, s)))
                 Reg.Group(dk["AK"], d["AP"], d["Js"]["J_$(n)LPT_e$(ε)"], "seed$s")
             end for s in seeds]
        end
        train = load(1:4); test = load(5:8)
        meta = Dict{Symbol,Any}(:method => "$(n)LPT", :R_f => R_f, :L => 1000.0, :N => 128, :h => 1000.0 / 128, :eps => ε, :A_source => "A_K(Kuhn edge tensor)", :cosmo => "lcdm")
        process_case(tag, meta, train, test, ε; flex = false, trainlabel = "1-4", testlabel = "5-8", innov = false)
    end
elseif startswith(suite, "fem")
    N = parse(Int, suite[4:end]); L = 500.0; R_f = 18.0; h = L / N
    d = JLD2.load(joinpath(datadir, "cubes_FEM_L500_N$N.jld2"))
    marg = ceil(Int, 2R_f / h)
    ii(n) = [mod(e - 1, N) for e in 1:n]          # i index of flat (i fastest) row e
    function slabs(AC, AP, J, name)
        i = ii(length(J))
        tr_ = findall(<(N ÷ 2 - marg), i)
        te_all = findall(x -> N ÷ 2 <= x < N - marg, i)
        edges = round.(Int, range(N ÷ 2, N - marg; length = 5))
        mk(rows, nm) = Reg.Group(AC[:, rows], AP[:, rows], J[rows], nm)
        train = [mk(tr_, "train")]
        test = [mk(findall(x -> edges[s] <= x < edges[s+1], i), "test$(s)") for s in 1:4]
        return train, test
    end
    trainlabel = "slab i in [0,$(N ÷ 2 - marg)) of seed 1"; testlabel = "4 sub-slabs of i in [$(N ÷ 2),$(N - marg)) of seed 1"
    specs = [("FEM", "FEM_e%s", :lin), ("FEM", "FEM_e%s", :in), ("4LPT", "J_4LPT_raw_e%s", :in), ("4LPT", "J_4LPT_matched_e%s", :lin),
             ("2LPT", "J_2LPT_raw_e%s", :in), ("2LPT", "J_2LPT_matched_e%s", :lin)]
    for ε in (1.0, -1.0, 0.5, -0.5), (method, jf, asrc) in specs
        partner = occursin("raw", jf) ? "raw" : occursin("matched", jf) ? "matched" : "-"
        tag = "fem$(N)_$(method)_$(partner)_A$(asrc)_e$(ε)"
        occursin(filt, tag) || continue
        jname = replace(jf, "%s" => string(ε))
        AC = asrc == :lin ? d["AC_lin"] : d["AC_in"]; AP = asrc == :lin ? d["AP_lin"] : d["AP_in"]
        train, test = slabs(AC, AP, d["Js"][jname], jname)
        meta = Dict{Symbol,Any}(:method => method, :R_f => R_f, :L => L, :N => N, :h => h, :eps => ε, :A_source => asrc == :lin ? "fem_lin" : "input",
                                :partner => partner, :cosmo => "eds")
        process_case(tag, meta, train, test, ε; flex = (asrc == :lin && method != "2LPT") || (method == "FEM"), trainlabel, testlabel)
    end
end
