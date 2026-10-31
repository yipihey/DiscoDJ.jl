# Second-order (A×B) plane-wave kernel → results/ab_kernel_<cfg>.h5
#
#   julia --project=. scripts/ab_kernel.jl prod_ks20 prod_ks20_ab
#
# Sign codes per realization: +1 = (A+,B+), −1 = (A−,B−) [first config], +2 = (A+,B−), −2 = (A−,B+) [second].
#   AB(k) = [ψ(++) − ψ(+−) − ψ(−+) + ψ(−−)]/4      (coefficient of s_A s_B: the band's own field, every term
#                                                      even or odd in the total sign, and A×A, B×B couplings cancel)
# At each pair target k_t = k_a ± k_b (a ∈ A, b ∈ B; targets hit by two pairs dropped) the tree-level kernel is the
# 2LPT value T = AB_2LPT(k_t) (independent of the band).  For model m:
#   K_m  = Re(AB_m∥ T∥*)/|T∥|²     (longitudinal kernel relative to tree; 2LPT ≡ 1)
#   V_m  = |AB_m⊥| / |T∥|           (transverse = vorticity channel; 2LPT and 3LPT ≡ 0 at this order)
# and the same for the Eulerian density (K^δ relative to the 2LPT density at k_t).
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, HDF5, Printf, Statistics, TOML
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
const MODELS = ["pm", "lpt2", "lpt3", "lpt5", "lpt7"]

c1, r1 = load_config(joinpath(scratch(), ARGS[1])); c2, r2 = load_config(joinpath(scratch(), ARGS[2]))
runs = merge(r1, r2)
L = c1["L"]; kf = 2π / L; epochs = Float64.(c1["epochs"]); Dini = c1["pm"]["D_ini"]
W = [Tuple(Int.(n)) for n in c1["planewaves"]["n"]]; B = Int.(c2["planewaves"]["group_b"]); A = setdiff(1:length(W), B)
nvec = first(values(r1))["nvec"]; kvec = kf .* nvec
lookup = Dict(Tuple(nvec[m, :]) => m for m in axes(nvec, 1))
targets = Dict{NTuple{3,Int},Vector{Tuple{Int,Int,Int}}}()
for a in A, b in B, s in (1, -1)
    t = W[a] .+ s .* W[b]; push!(get!(targets, t, Tuple{Int,Int,Int}[]), (a, b, s))
end
tl = [(t, only(v)) for (t, v) in targets if length(v) == 1 && !haskey(targets, .-t)]
idx = Int[]; cj = Bool[]; keep = Tuple{NTuple{3,Int},Tuple{Int,Int,Int}}[]
for (t, p) in tl
    if haskey(lookup, t); push!(idx, lookup[t]); push!(cj, false)
    elseif haskey(lookup, .-t); push!(idx, lookup[.-t]); push!(cj, true)
    else continue end
    push!(keep, (t, p))
end
np = length(keep); kt = [kf * sqrt(sum(abs2, t)) for (t, _) in keep]
k1 = [kf * sqrt(sum(abs2, W[p[1]])) for (_, p) in keep]; k2 = [kf * sqrt(sum(abs2, W[p[2]])) for (_, p) in keep]
mu = [p[3] * sum(W[p[1]] .* W[p[2]]) / sqrt(sum(abs2, W[p[1]]) * sum(abs2, W[p[2]])) for (_, p) in keep]
kv = kvec[idx, :]
seeds = sort(unique(k[3] for k in keys(runs)))
fix(v) = [cj[i] ? conj(v[i, :]) : v[i, :] for i in axes(v, 1)]         # rows → value at +k_t
function abfield(model, e, D, s, which)
    mt = startswith(model, "lpt") ? "lpt" : "pm"
    g(sg) = (r = runs[(mt, :UV, s, s, sg)]; f = EFTShell._fields(r, model, e, D); which == :psi ? f[1][idx, :] : reshape(f[3][idx], :, 1))
    raw = (g(1) .- g(2) .- g(-2) .+ g(-1)) ./ 4
    out = similar(raw); for i in axes(raw, 1); out[i, :] .= cj[i] ? conj.(raw[i, :]) : raw[i, :]; end
    return out
end
E = length(epochs)
K = fill(NaN, length(MODELS), E, np, length(seeds)); V = similar(K); Kd = similar(K)
for (is, s) in enumerate(seeds), (e, D) in enumerate(epochs)
    T = abfield("lpt2", e, D, s, :psi); Tl = longitudinal(T, kv); Td = abfield("lpt2", e, D, s, :dens)[:, 1]
    for (im_, m) in enumerate(MODELS)
        (m == "pm" && D <= Dini) && continue
        X = abfield(m, e, D, s, :psi); Xl = longitudinal(X, kv); Xt = transverse(X, kv); Xd = abfield(m, e, D, s, :dens)[:, 1]
        K[im_, e, :, is] .= real.(Xl .* conj.(Tl)) ./ abs2.(Tl)
        V[im_, e, :, is] .= vec(sqrt.(sum(abs2, Xt; dims=2))) ./ abs.(Tl)
        Kd[im_, e, :, is] .= real.(Xd .* conj.(Td)) ./ abs2.(Td)
    end
end
h5open(joinpath(@__DIR__, "..", "results", "ab_kernel_$(ARGS[1]).h5"), "w") do h
    h["K"] = K; h["V"] = V; h["Kdens"] = Kd; h["kt"] = kt; h["k1"] = k1; h["k2"] = k2; h["mu"] = mu; h["epochs"] = epochs
    attrs(h)["models"] = join(MODELS, ",")
end
mn(x) = mean(x; dims=ndims(x)); se(x) = std(x; dims=ndims(x)) ./ sqrt(size(x, ndims(x)))
@printf "%d pair targets (k_t = %.3f–%.3f), %d realizations\n" np minimum(kt) maximum(kt) length(seeds)
println("    D  | ⟨K−1⟩ over pairs: 3LPT     7LPT     PM        | PM−7LPT: mean ± sem  [fit c·k_t² : c]  | ⟨V⟩ 7LPT   PM   | density: PM−7LPT")
for (e, D) in enumerate(epochs)
    D <= Dini && continue
    Km = dropdims(mean(K[:, e, :, :]; dims=3); dims=3); dK = Km[1, :] .- Km[5, :]
    dKs = vec(std(K[1, e, :, :] .- K[5, e, :, :]; dims=2)) ./ sqrt(length(seeds))
    w = 1 ./ max.(dKs, 1e-6) .^ 2; c = sum(w .* dK .* kt .^ 2) / sum(w .* kt .^ 4); σc = 1 / sqrt(sum(w .* kt .^ 4))
    Vm = dropdims(mean(V[:, e, :, :]; dims=3); dims=3); Kdm = dropdims(mean(Kd[:, e, :, :]; dims=3); dims=3)
    @printf "%5.2f  |  %+8.4f %+8.4f %+8.4f  | %+8.4f ± %.4f  [%+8.2f ± %.2f]  | %.4f  %.4f | %+8.4f\n" D mean(Km[3, :]) - 1 mean(Km[5, :]) - 1 mean(Km[1, :]) - 1 mean(dK) std(dK) / sqrt(np) c σc mean(Vm[5, :]) mean(Vm[1, :]) mean(Kdm[1, :] .- Kdm[5, :])
end
