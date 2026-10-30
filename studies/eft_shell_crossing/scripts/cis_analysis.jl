# Counts-in-spheres analysis of an RG configuration → results/cis_<name>.h5
#
#   julia --project=. scripts/cis_analysis.jl rg_ks12 rg_ks15 ...
#
# Every member stores, at identical sphere centres (32³ lattice) and radii R = 20…150 Mpc/h, the top-hat
# contrast δ_R, its Laplacian, and δ_R^{<k_c} of the field sharply low-passed at each split k_c, for the
# measured density (PM, 4LPT, 7LPT) and the exact linear field (D = 1).  With k_s the split of this config:
#
#   Δ_R        = δ_R^UV − δ_R^IR                       (what a sphere count sees)
#   Δ_R^<      = δ_R^{UV,<k_s} − δ_R^{IR,<k_s}         (only modes the two members share initially; exact zero at D → 0)
#   Δ_R^band   = Δ_R − Δ_R^<                           (the added band's own modes leaking through the top-hat window;
#                Δ_R^band,lin = D(δ_R^{lin,UV} − δ_R^{lin,IR}) is its linear part)
#   coherent   Δ_R^< = b_R δ_R^{IR,<} + ε_R,  b_R = ⟨Δ^< δ^{IR,<}⟩ / ⟨(δ^{IR,<})²⟩
#   EFT form   Δ_R^< = C2_R ∇²δ_R^L + η_R,  ∇²δ_R^L = D ∇²δ_R^{lin,IR}  (band-limited long field)
#   Δσ²_R/σ²_R(IR) = [(1+b)² − 1]·σ²_<,IR/σ²_IR   (coherent)
#                  + Var(ε_R)/σ²_IR                (stochastic, k < k_s)
#                  + remainder                     (band leakage and cross terms; its linear part reported separately)
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, HDF5, TOML, Statistics, Printf, Random
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")

function load(dir, model, member, sL)
    f = joinpath(dir, @sprintf("%s_%s_L%d_S%d_s+1.h5", model, member, sL, member == "UV" ? sL : 0))
    h5open(f) do h
        (cis=read(h["cis"]), lin=read(h["cis_lin"]), R=read(h["cis_R"]), kcut=read(h["cis_kcut"]),
         orders=haskey(h, "cis_orders") ? read(h["cis_orders"]) : Int[], epochs=read(h["cis_epochs"]))
    end
end

# sphere samples of one member: (δ_R, δ_R^{<k_s}) at cis-epoch e, radius r
function samples(M, kind, ord, e, r, ch)
    if kind == "pm"
        return Float64.(M.cis[e, r, 1, :]), Float64.(M.cis[e, r, ch, :])
    else
        o = findfirst(==(ord), M.orders)
        return Float64.(M.cis[e, o, r, 1, :]), Float64.(M.cis[e, o, r, ch, :])
    end
end

const KEYS = ("n", "ii", "uu", "LL", "lb", "ll", "Dl", "bl", "dd", "D2", "ql", "qq", "DI")

function analyse_cis(name)
    dir = joinpath(scratch(), name); cfg = TOML.parsefile(joinpath(dir, "config.toml"))
    longs = cfg["seeds"]["long"]; ks = cfg["k_split"]
    models = [("pm", "pm", 0), ("lpt4", "lpt", 4), ("lpt7", "lpt", 7)]
    f1 = load(dir, "pm", "UV", longs[1]); Rs = f1.R; E = length(f1.epochs); nR = length(Rs)
    ch = 2 + findfirst(x -> isapprox(x, ks; atol=1e-9), f1.kcut)
    nm = length(models); nL = length(longs)
    S = Dict(k => zeros(nm, E, nR, nL) for k in KEYS)
    nk = min(3000, size(f1.cis, 4))
    keep = randperm(MersenneTwister(1), size(f1.cis, 4))[1:nk]
    scatter = zeros(Float32, nm, E, nR, 3, nk * nL)          # (∇²δ_L, Δ^<, δ^{IR,<}) samples for plots
    for (il, sL) in enumerate(longs), (im_, (m, kind, ord)) in enumerate(models)
        U = load(dir, kind, "UV", sL); I = load(dir, kind, "IR", sL)
        for e in 1:E, r in 1:nR
            D = U.epochs[e]
            du, dul = samples(U, kind, ord, e, r, ch); di, dil = samples(I, kind, ord, e, r, ch)
            Δ = du .- di; Δl = dul .- dil
            blin = D .* (Float64.(U.lin[r, 1, :]) .- Float64.(I.lin[r, 1, :]))
            lap = D .* Float64.(I.lin[r, 2, :]); dL = D .* Float64.(I.lin[r, 1, :])
            for (k, v) in (("n", length(du)), ("ii", sum(abs2, di)), ("uu", sum(abs2, du)), ("LL", sum(abs2, dil)),
                           ("lb", sum(Δl .* dil)), ("ll", sum(abs2, Δl)), ("Dl", sum(Δl .* lap)), ("bl", sum(abs2, blin)),
                           ("dd", sum(abs2, lap)), ("D2", sum(abs2, Δ)), ("ql", sum(Δl .* dL)), ("qq", sum(abs2, dL)),
                           ("DI", sum(Δ .* di)))
                S[k][im_, e, r, il] += v
            end
            sl = (il-1)*nk+1:il*nk
            scatter[im_, e, r, 1, sl] .= lap[keep]; scatter[im_, e, r, 2, sl] .= Δl[keep]; scatter[im_, e, r, 3, sl] .= dil[keep]
        end
    end
    function stats(sel)
        g(k) = dropdims(sum(S[k][:, :, :, sel]; dims=4); dims=4)
        n = g("n"); σI = g("ii") ./ n
        b = g("lb") ./ g("LL")                                   # coherent slope on the shared-mode IR field
        varε = g("ll") ./ n .- b .^ 2 .* g("LL") ./ n            # stochastic, modes k < k_s
        coh = ((1 .+ b) .^ 2 .- 1) .* (g("LL") ./ n) ./ σI
        tot = g("uu") ./ g("ii") .- 1
        C2R = g("Dl") ./ g("dd")                                 # Δ^< = C2_R ∇²δ_L
        r_eft = g("Dl") ./ sqrt.(g("dd") .* g("ll"))
        bL = g("ql") ./ g("qq")                                  # slope on the linear long field
        return (; ratio_total=tot, coherent=coh, stoch=varε ./ σI, band_lin=g("bl") ./ n ./ σI,
                remainder=tot .- coh .- varε ./ σI, b, b_lin=bL, C2R, r_eft, sigma2_ir=σI,
                resid_eft=(g("ll") .- C2R .^ 2 .* g("dd")) ./ n ./ σI)
    end
    full = stats(1:nL)
    out = joinpath(@__DIR__, "..", "results", "cis_$name.h5")
    h5open(out, "w") do h
        h["R"] = Rs; h["epochs"] = f1.epochs; attrs(h)["models"] = join(getindex.(models, 1), ",")
        attrs(h)["n_long"] = nL; attrs(h)["k_split"] = ks
        for k in keys(full)
            h[String(k)] = full[k]
            if nL >= 3
                jk = [stats(setdiff(1:nL, i))[k] for i in 1:nL]; mjk = sum(jk) ./ nL
                h[String(k) * "_sigma"] = sqrt.((nL - 1) / nL .* sum(x -> (x .- mjk) .^ 2, jk))
            end
        end
        h["scatter"] = scatter
    end
    e = length(f1.epochs)
    println("$name (k_s = $ks), D = $(f1.epochs[e]), PM:  R | Δσ²/σ² total = coherent + stochastic(k<k_s) + remainder (band-lin) | b | C2_R (r)")
    for r in 1:nR
        @printf "  R = %5.0f  %+8.4f = %+8.4f + %8.5f + %+8.5f (%8.5f) | b = %+7.4f | C2_R = %6.3f (%.2f)\n" Rs[r] full.ratio_total[1, e, r] full.coherent[1, e, r] full.stoch[1, e, r] full.remainder[1, e, r] full.band_lin[1, e, r] full.b[1, e, r] full.C2R[1, e, r] full.r_eft[1, e, r]
    end
end

foreach(analyse_cis, ARGS)
