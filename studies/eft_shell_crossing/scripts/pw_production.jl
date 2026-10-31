# Production plane-wave analysis → results/pw_<config>.h5 and a printed summary
#
#   julia --project=. scripts/pw_production.jl prod_ks20 prod_ks20_bg05 ...
#
# For every realization s (long seed; single_field band), model m ∈ {pm, lpt1..lpt7}, epoch D and wave j:
#   odd_m(member) = [ψ_m(member, +ε) − ψ_m(member, −ε)]/2 at k_j (longitudinal)
#   Δ_j = odd_m(UV) − odd_m(IR)        (IR member: run in the config when a long background is present; otherwise
#                                       exactly x_j.  Opt-in PW_PM_IR_SIBLING=1: use PM's bare waves from <name>_ir —
#                                       NOT the default: PM's bare-wave error (+1…+7e-5, lattice-limited) does not carry
#                                       over to waves on a band (README §14.2))
#   R_j = Re(Δ_j x_j*)/|x_j|²,   x_j = D ψ₁,PW(k_j)   (linear displacement of the waves alone)
# Fits per (model, D):  R(k) = −C₂k² + C₄k⁴ (+ C₆k⁶)  over the waves, unweighted per wave (the waves carry
# identical |x|-normalised information); errors: delete-one jackknife over realizations.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, HDF5, Printf, Statistics, LinearAlgebra
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
const MODELS = ["pm", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7"]

function polyfit(k, R, order)
    A = hcat([(-1)^p .* k .^ (2p) for p in 1:order]...)       # columns −k², +k⁴, −k⁶ → (C₂, C₄, C₆) with R = −C₂k² + C₄k⁴ − C₆k⁶
    return A \ R
end

function analyse_pw(name)
    cfg, runs = load_config(joinpath(scratch(), name)); pk = planck_pk()
    L = cfg["L"]; kf = 2π / L; epochs = Float64.(cfg["epochs"]); Dini = cfg["pm"]["D_ini"]
    seeds = sort(unique(k[3] for k in keys(runs)))
    any_run = first(values(runs)); nvec = any_run["nvec"]; kvec = kf .* nvec; km = vec(sqrt.(sum(abs2, kvec; dims=2)))
    pws = [Tuple(Int.(n)) for n in cfg["planewaves"]["n"]]
    idx = [findfirst(m -> Tuple(nvec[m, :]) == n || Tuple(nvec[m, :]) == .-n, axes(nvec, 1)) for n in pws]
    any(isnothing, idx) && error("a plane wave is not among the stored modes")
    kw = km[idx]; nw = length(idx); E = length(epochs)
    ψ1 = linear_psi1_modes(cfg, 1, 1, nvec, pk)[idx, :]; kv = kvec[idx, :]
    R = fill(NaN, length(MODELS), E, nw, length(seeds))
    inv = fill(NaN, length(MODELS), E, length(seeds))                       # UV(+ε) shell-crossed fraction
    irdir = joinpath(scratch(), name * "_ir")
    irruns = (get(ENV, "PW_PM_IR_SIBLING", "0") == "1" && isdir(irdir)) ? load_config(irdir)[2] : Dict()
    function get_run(model, w, s, sg)
        r = get(runs, (model, w, s, w == :IR ? 0 : s, sg), nothing)
        (r === nothing && w == :IR && model == "pm") && (r = get(irruns, ("pm", :IR, 1, 0, sg), nothing))
        return r
    end
    pm_ir_source = haskey(runs, ("pm", :IR, seeds[1], 0, 1)) ? "own" : (haskey(irruns, ("pm", :IR, 1, 0, 1)) ? "sibling" : "analytic")
    for (is, s) in enumerate(seeds), (im_, model) in enumerate(MODELS), (e, D) in enumerate(epochs)
        mt = startswith(model, "lpt") ? "lpt" : "pm"
        (mt == "pm" && D <= Dini) && continue
        up = get_run(mt, :UV, s, 1); um = get_run(mt, :UV, s, -1)
        (up === nothing || um === nothing) && continue
        odd(a, b) = longitudinal((EFTShell._fields(a, model, e, D)[1][idx, :] .- EFTShell._fields(b, model, e, D)[1][idx, :]) ./ 2, kv)
        x = D .* longitudinal(ψ1, kv)
        ip = get_run(mt, :IR, s, 1); imn = get_run(mt, :IR, s, -1)
        ref = (ip === nothing || imn === nothing) ? x : odd(ip, imn)
        Δ = odd(up, um) .- ref
        R[im_, e, :, is] .= real.(Δ .* conj.(x)) ./ abs2.(x)
        inv[im_, e, is] = EFTShell._fields(up, model, e, D)[4]
    end
    # fits on the realization mean, jackknife over realizations
    nS = length(seeds); fits = fill(NaN, length(MODELS) + 1, E, 3, 3); sig = similar(fits)   # (model+[pm−lpt7], E, order, coeff)
    series(sel) = begin
        Rm = dropdims(mean(R[:, :, :, sel]; dims=4); dims=4)
        vcat(Rm, reshape(Rm[1, :, :] .- Rm[8, :, :], 1, E, nw))
    end
    function fitall(sel)
        Rm = series(sel); F = fill(NaN, size(Rm, 1), E, 3, 3)
        for i in axes(Rm, 1), e in 1:E, o in 1:3
            all(isfinite, Rm[i, e, :]) || continue
            F[i, e, o, 1:o] .= polyfit(kw, Rm[i, e, :], o)
        end
        F
    end
    fits .= fitall(1:nS)
    if nS >= 3
        jk = [fitall(setdiff(1:nS, i)) for i in 1:nS]; mj = sum(jk) ./ nS
        sig .= sqrt.((nS - 1) / nS .* sum(x -> (x .- mj) .^ 2, jk))
    end
    out = joinpath(@__DIR__, "..", "results", "pw_$name.h5")
    h5open(out, "w") do h
        h["R"] = R; h["k"] = kw; h["n"] = hcat(collect.(pws)...); h["epochs"] = epochs; h["seeds"] = seeds
        h["inv"] = inv; h["fits"] = fits; h["fits_sigma"] = sig
        attrs(h)["models"] = join(vcat(MODELS, ["pm-lpt7"]), ","); attrs(h)["k_split"] = cfg["k_split"]
        attrs(h)["background"] = get(cfg["planewaves"], "background", 0.0); attrs(h)["amp"] = cfg["planewaves"]["amp"]
        attrs(h)["pm_ir_source"] = pm_ir_source
        attrs(h)["fits_layout"] = "(model, epoch, fit order 1..3, coefficient C2,C4,C6)"
    end
    @printf "\n%s: k_split = %.2f, background %.1f, %d realizations × %d waves (k = %.3f–%.3f), PM IR member: %s\n" name cfg["k_split"] get(cfg["planewaves"], "background", 0.0) nS nw minimum(kw) maximum(kw) pm_ir_source
    println("    D    inv(PM)   C₂ PM (2-term)       C₄ PM (2-term)     C₂ 3LPT   C₂ 7LPT     ΔC₂ PM−7LPT          ΔC₄ PM−7LPT")
    for (e, D) in enumerate(epochs)
        D <= Dini && continue
        @printf "%5.2f  %8.1e  %+8.4f ± %.4f   %+8.2f ± %.2f   %+8.4f  %+8.4f   %+8.4f ± %.4f   %+8.2f ± %.2f\n" D mean(inv[1, e, :]) fits[1, e, 2, 1] sig[1, e, 2, 1] fits[1, e, 2, 2] sig[1, e, 2, 2] fits[4, e, 2, 1] fits[8, e, 2, 1] fits[9, e, 2, 1] sig[9, e, 2, 1] fits[9, e, 2, 2] sig[9, e, 2, 2]
    end
    @printf "  LPT shell-crossed fraction at D_ini = %.2f (7LPT, UV): %.1e\n" Dini maximum(inv[8, findfirst(≈(Dini), epochs), :])
end

foreach(analyse_pw, ARGS)
