# Top-hat sphere variance of the shared long modes (|k| < k_split) from the stored Fourier modes — exact for
# the low-passed field (σ²_R = Σ_k |δ_k|² W²_TH(kR)), and a cross-check of the direct counts in spheres.
#
#   julia --project=. scripts/spheres_from_modes.jl fiducial      → results/spheres_modes_<name>.json
#
# Per radius R, model, epoch (pooled over all pairs; jackknife over long seeds):
#   Δσ²_R/σ²_R = σ²_UV/σ²_IR − 1 = [(1+b_R)² − 1]  (coherent: b_R = ΣW² Re(Δδ δ_IR*)/ΣW²|δ_IR|²)
#                                + ΣW²|Δδ − b_R δ_IR|² / ΣW²|δ_IR|²  (stochastic)
#   prediction from the Fourier response: b_R ≈ ΣW² R(k)|δ_L|² / ΣW²|δ_L|² with R(k) = −C₂k² (k ≤ 0.08 fit),
#   i.e. Δσ²_R/σ²_R ≈ −2 C₂ ⟨k²⟩_R.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, JSON3, Printf, Statistics
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
wth(x) = x < 1e-4 ? 1 - x^2 / 10 : 3 * (sin(x) - x * cos(x)) / x^3

name = ARGS[1]
cfg, runs = load_config(joinpath(scratch(), name)); pk = planck_pk()
L = cfg["L"]; ks = cfg["k_split"]; epochs = Float64.(cfg["epochs"])
Rs = [20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0]
models = ["lpt1", "lpt3", "lpt7", "pm"]
anyrun = first(values(runs)); nvec = anyrun["nvec"]
kvec = (2π / L) .* nvec; kmag = vec(sqrt.(sum(abs2, kvec; dims=2))); sel = kmag .< ks; kmag = kmag[sel]
W2 = [wth.(kmag .* R) .^ 2 for R in Rs]
longs = sort(unique(k[3] for k in keys(runs)))
# sums per (seed, model, epoch, R): uu, ii, di, dd(after b), and the linear-field moments for the prediction
acc = Dict{Tuple{Int,String,Int,Int},Vector{Float64}}()
cache = Dict()
for sL in longs, (im_, m) in enumerate(models)
    mtype = m == "pm" ? "pm" : "lpt"; n = m == "pm" ? 0 : parse(Int, m[4:end])
    for (e, D) in enumerate(epochs)
        (m == "pm" && D <= cfg["pm"]["D_ini"]) && continue
        pairs = []
        for k in keys(runs)
            (k[1] == mtype && k[2] == :UV && k[3] == sL) || continue
            ir = get(runs, (mtype, :IR, sL, 0, k[5]), nothing); ir === nothing && continue
            du = m == "pm" ? runs[k]["dens"][e, :] : runs[k]["dens"][e, n, :]
            di = m == "pm" ? ir["dens"][e, :] : ir["dens"][e, n, :]
            ψ1 = get!(() -> linear_psi1_modes(cfg, sL, k[5], nvec, pk; cache=cache), cache, (sL, k[5]))
            δL = D .* (-im .* (2π / L) .* vec(sum(nvec .* ψ1; dims=2)) ./ 1)      # −i k·ψ_L
            push!(pairs, (ComplexF64.(du[sel]), ComplexF64.(di[sel]), δL[sel]))
        end
        isempty(pairs) && continue
        for (r, w) in enumerate(W2)
            v = zeros(5)
            for (du, di, δL) in pairs
                Δ = du .- di
                v[1] += sum(w .* abs2.(du)); v[2] += sum(w .* abs2.(di)); v[3] += sum(w .* real.(Δ .* conj.(di)))
                v[4] += sum(w .* abs2.(Δ)); v[5] += sum(w .* kmag .^ 2 .* abs2.(δL)) / sum(w .* abs2.(δL)) / length(pairs)
            end
            acc[(sL, m, e, r)] = v
        end
    end
end
function stats(seeds, m, e, r)
    vs = [acc[(s, m, e, r)] for s in seeds if haskey(acc, (s, m, e, r))]
    isempty(vs) && return (tot=NaN, coh=NaN, sto=NaN, k2=NaN)
    v = sum(vs); b = v[3] / v[2]
    (tot=v[1] / v[2] - 1, coh=(1 + b)^2 - 1, sto=(v[4] - 2b * v[3] + b^2 * v[2]) / v[2], k2=v[5] / length(vs))
end
res = Dict{String,Any}("R" => Rs, "epochs" => epochs, "k_split" => ks)
fit = Result = nothing
h = joinpath(@__DIR__, "..", "results", "$name.h5")
using HDF5
C2tab = h5open(h) do f
    g = f["dens/fit1"]; j = argmin(abs.(read(g["kmax"]) .- 0.08)); ms = split(read_attribute(f, "models"), ",")
    Dict(m => read(g["C2"])[findfirst(==(m), ms), :, j] for m in models if m in ms)
end
for m in models, (e, D) in enumerate(epochs), (r, R) in enumerate(Rs)
    s = stats(longs, m, e, r); isnan(s.tot) && continue
    jk = [stats(setdiff(longs, [l]), m, e, r) for l in longs]
    σ(f) = (vals = [getfield(x, f) for x in jk if !isnan(getfield(x, f))]; length(vals) < 2 ? NaN :
            sqrt((length(vals) - 1) / length(vals) * sum(abs2, vals .- mean(vals))))
    pred = haskey(C2tab, m) ? (1 - C2tab[m][e] * s.k2)^2 - 1 : NaN
    res["$(m)_D$(D)_R$(Int(R))"] = Dict("total" => s.tot, "coherent" => s.coh, "stochastic" => s.sto,
        "sigma_total" => σ(:tot), "sigma_coherent" => σ(:coh), "predicted_from_C2" => pred, "k2_R" => s.k2)
end
clean(x) = x isa AbstractFloat && !isfinite(x) ? nothing : x isa Dict ? Dict(k => clean(v) for (k, v) in x) : x
open(io -> JSON3.write(io, clean(res)), joinpath(@__DIR__, "..", "results", "spheres_modes_$name.json"), "w")
for m in models, D in (0.2, 0.5, 1.0)
    haskey(res, "$(m)_D$(D)_R20") || continue
    println("$m, D = $D:  R | Δσ²/σ² (shared modes k < $ks) = coherent + stochastic | predicted −2C₂⟨k²⟩_R")
    for R in Rs
        x = res["$(m)_D$(D)_R$(Int(R))"]
        @printf "   R = %5.0f  %+.3e ± %.1e = %+.3e + %.3e | %+.3e\n" R x["total"] x["sigma_total"] x["coherent"] x["stochastic"] x["predicted_from_C2"]
    end
end
